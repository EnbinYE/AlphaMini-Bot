"""
main.py — 主程序循環與風控報警

優化點：
- 敏感信息改為環境變量（TELEGRAM_TOKEN / CHAT_ID）
- 數據抓取加入重試 + 熔斷機制（連續失敗後報警並停止重試刷屏）
- 風控：橫盤過濾（價格變化 < 0.05% 不計入對錯）
- 風控：報警後重置計數器，避免無限發送通知
- 風控：模型 CV 準確率過低時主動報警
- 新增「觀望」狀態，不更新 last_prediction，避免誤計錯誤
- 統一日誌格式，每輪輸出清晰可讀的狀態摘要

部署到 GCP 香港（以 nohup 為例）：
  export TELEGRAM_TOKEN="your_token"
  export CHAT_ID="your_chat_id"
  nohup python3 main.py > alphamini.log 2>&1 &

或寫入 .env 並用 python-dotenv 加載（推薦）：
  pip install python-dotenv
  在代碼頂部加: from dotenv import load_dotenv; load_dotenv()
"""

import os
import time
import requests
from datetime import datetime

from data_agent import fetch_crypto_data, prepare_features, FEATURE_COLS
from strategy_agent import train_and_predict
from dotenv import load_dotenv
load_dotenv()

# ── 配置區（從環境變量讀取，不硬編碼敏感信息）──────────────────────────
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN", "")
CHAT_ID = os.environ.get("CHAT_ID", "")

SYMBOL = "BTCUSDT"
INTERVAL = "1m"          # 可改為 "5m" 或 "15m" 以降低噪音
LOOP_INTERVAL = 60       # 主循環間隔（秒）

# 風控參數
MAX_CONSECUTIVE_ERRORS = 3     # 連續錯誤多少次觸發報警
MIN_PRICE_CHANGE_PCT = 0.0005  # 低於此漲幅視為橫盤，不計入對錯（0.05%）

# 數據抓取重試參數
FETCH_MAX_RETRIES = 3
FETCH_RETRY_DELAY = 5  # 每次重試等待秒數
FETCH_FAIL_ALERT_THRESHOLD = 3  # 連續失敗多少次觸發報警（避免重複刷屏）

# ── 狀態變量（模塊級，供 main_loop 使用）──────────────────────────────
_consecutive_fetch_fails = 0
_fetch_alert_sent = False  # 熔斷標記：數據源故障報警只發一次


# ── Telegram 報警 ──────────────────────────────────────────────────────
def send_alert(message: str) -> None:
    """發送即時警報到手機 Telegram。Token 未配置時退化為僅打印。"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    full_message = f"[AlphaMini {timestamp}] {message}"

    if not TELEGRAM_TOKEN or not CHAT_ID:
        print(f"[Alert] （未配置 Telegram，僅本地輸出）{full_message}")
        return

    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": CHAT_ID, "text": full_message}
    try:
        requests.post(url, json=payload, timeout=3)
        print(f"[Alert] 已發送: {full_message}")
    except Exception as e:
        print(f"[Alert] Telegram 發送失敗: {e}")


# ── 帶重試的數據抓取 ────────────────────────────────────────────────────
def fetch_with_retry() -> object:
    """
    嘗試抓取數據，最多重試 FETCH_MAX_RETRIES 次。
    連續失敗超過 FETCH_FAIL_ALERT_THRESHOLD 次後觸發報警（熔斷，只報一次）。
    返回 DataFrame 或 None。
    """
    global _consecutive_fetch_fails, _fetch_alert_sent

    for attempt in range(1, FETCH_MAX_RETRIES + 1):
        df = fetch_crypto_data(symbol=SYMBOL, interval=INTERVAL, limit=120)
        if df is not None:
            # 成功：重置失敗計數器和熔斷標記
            _consecutive_fetch_fails = 0
            _fetch_alert_sent = False
            return df
        print(f"[Main] 數據抓取失敗，第 {attempt}/{FETCH_MAX_RETRIES} 次重試...")
        if attempt < FETCH_MAX_RETRIES:
            time.sleep(FETCH_RETRY_DELAY)

    # 全部重試失敗
    _consecutive_fetch_fails += 1
    if _consecutive_fetch_fails >= FETCH_FAIL_ALERT_THRESHOLD and not _fetch_alert_sent:
        send_alert(f"數據源連續失敗 {_consecutive_fetch_fails} 輪，請檢查幣安 API 或網絡狀態！")
        _fetch_alert_sent = True  # 熔斷：不重複刷屏

    return None


# ── 主循環 ─────────────────────────────────────────────────────────────
def main_loop() -> None:
    print("=" * 55)
    print("  AlphaMini-Bot 量化沙盒 — 已啟動")
    print(f"  交易對: {SYMBOL} | 周期: {INTERVAL} | 間隔: {LOOP_INTERVAL}s")
    print(f"  Telegram: {'已配置' if TELEGRAM_TOKEN else '未配置（僅本地輸出）'}")
    print("=" * 55)

    last_price: float | None = None
    last_prediction: int | None = None  # 1=看漲, 0=看跌, None=上輪觀望
    error_counter: int = 0

    while True:
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"\n{'─' * 50}")
        print(f"[{now}] 開始新一輪循環...")

        # ── Step 1: 抓取數據 ────────────────────────────────────────
        df_raw = fetch_with_retry()
        if df_raw is None:
            print("[Main] 本輪跳過，等待下一輪...")
            time.sleep(LOOP_INTERVAL)
            continue

        # ── Step 2: 特徵工程 ─────────────────────────────────────────
        df = prepare_features(df_raw)
        if df is None:
            print("[Main] 特徵工程失敗，本輪跳過...")
            time.sleep(LOOP_INTERVAL)
            continue

        current_price: float = df["close"].iloc[-1]
        print(f"[Main] 當前價格: {current_price:.2f} USDT")

        # ── Step 3: 風控驗證（對比上一輪預測）────────────────────────
        if last_price is not None and last_prediction is not None:
            price_change_pct = abs(current_price - last_price) / last_price

            if price_change_pct < MIN_PRICE_CHANGE_PCT:
                # 橫盤過濾：價格變化過小，結果不可信，不計入對錯
                print(f"[Risk] 橫盤過濾（變化 {price_change_pct*100:.4f}%），跳過本輪驗證")
            else:
                actual_up = current_price > last_price
                predicted_up = bool(last_prediction)
                is_correct = (actual_up == predicted_up)

                if is_correct:
                    error_counter = 0
                    print(f"[Risk] ✅ 預測正確，錯誤計數歸零")
                else:
                    error_counter += 1
                    print(f"[Risk] ❌ 預測錯誤，連續錯誤次數: {error_counter}")

                # 觸發風控報警（報警後重置，避免無限刷屏）
                if error_counter >= MAX_CONSECUTIVE_ERRORS:
                    send_alert(
                        f"策略連續 {error_counter} 次預測錯誤！"
                        f"市場可能趨勢反轉，當前價格 {current_price:.2f} USDT，請注意風險！"
                    )
                    error_counter = 0  # 重置，避免下輪繼續重複報警

        # ── Step 4: 模型訓練與預測 ──────────────────────────────────
        prediction, prob, cv_acc = train_and_predict(df, FEATURE_COLS)

        # CV 準確率過低時主動報警（表示市場進入難以預測的狀態）
        if cv_acc > 0 and cv_acc < 0.52:
            send_alert(f"模型本輪 CV 準確率僅 {cv_acc:.3f}，接近隨機水平，建議觀望！")

        # ── Step 5: 輸出結果並更新狀態 ──────────────────────────────
        if prediction is None:
            print(f"[Model] ⏸  觀望（CV准確率={cv_acc:.3f}, 信心度={prob:.3f}）")
            # 觀望時不更新 last_prediction，保留上一輪有效預測用於下輪驗證
        else:
            direction = "🔺 看漲" if prediction == 1 else "🔻 看跌"
            print(f"[Model] {direction} | 信心度={prob:.3f} | CV准確率={cv_acc:.3f}")
            last_prediction = prediction

        last_price = current_price

        # ── Step 6: 等待下一輪 ────────────────────────────────────────
        print(f"[Main] 等待 {LOOP_INTERVAL} 秒...")
        time.sleep(LOOP_INTERVAL)


if __name__ == "__main__":
    main_loop()
