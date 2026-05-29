"""
data_agent.py — 數據抓取與特徵工程

優化點：
- 新增 RSI、成交量比率、高低波幅三個特徵
- 加入前向偏差保護注釋，明確標記 X_current 邊界
- 特徵計算異常時返回 None，由調用方決策
"""

import requests
import pandas as pd
import numpy as np


def fetch_crypto_data(symbol: str = "BTCUSDT", interval: str = "1m", limit: int = 120) -> pd.DataFrame | None:
    """
    從幣安公開 API 抓取 K 線數據（無需 API Key）。
    limit 設為 120，為滾動指標留足預熱期。
    """
    url = "https://api.binance.com/api/v3/klines"
    params = {"symbol": symbol, "interval": interval, "limit": limit}

    try:
        response = requests.get(url, params=params, timeout=5)
        response.raise_for_status()
        data = response.json()

        df = pd.DataFrame(data, columns=[
            "open_time", "open", "high", "low", "close", "volume",
            "close_time", "quote_asset_volume", "number_of_trades",
            "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore"
        ])

        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = df[col].astype(float)

        return df

    except requests.RequestException as e:
        print(f"[DataAgent] 網絡請求失敗: {e}")
        return None
    except Exception as e:
        print(f"[DataAgent] 數據解析失敗: {e}")
        return None


def _compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """手寫 RSI，避免引入 TA-Lib 依賴。"""
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(window=period).mean()
    loss = (-delta.clip(upper=0)).rolling(window=period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def prepare_features(df: pd.DataFrame) -> pd.DataFrame | None:
    """
    特徵工程：生成六個量化因子。

    ⚠️ 前向偏差保護：
      target = close.shift(-1) > close
      這意味著最後一行的 target 是 NaN（未來未知）。
      訓練時必須用 df[:-1]，最後一行僅作為 X_current 預測用。
      此約定由 strategy_agent.py 的 train_and_predict() 負責執行。
    """
    try:
        # --- 趨勢類 ---
        df["sma_5"] = df["close"].rolling(window=5).mean()
        df["sma_10"] = df["close"].rolling(window=10).mean()
        df["feature_sma_diff"] = df["sma_5"] - df["sma_10"]

        # --- 動量類 ---
        df["feature_momentum"] = df["close"] - df["close"].shift(3)

        # --- 波動類 ---
        df["feature_volatility"] = df["close"].rolling(window=5).std()
        df["feature_hl_ratio"] = (df["high"] - df["low"]) / df["close"]  # 高低波幅

        # --- 超買超賣 ---
        df["feature_rsi"] = _compute_rsi(df["close"], period=14)

        # --- 量價類 ---
        df["vol_ma_10"] = df["volume"].rolling(window=10).mean()
        df["feature_vol_ratio"] = df["volume"] / df["vol_ma_10"].replace(0, np.nan)

        # --- 標籤（前向偏差保護：最後一行 target=NaN，訓練時必須排除）---
        df["target"] = (df["close"].shift(-1) > df["close"]).astype(float)
        df.loc[df.index[-1], "target"] = np.nan  # 顯式標記最後一行不可用於訓練

        df = df.dropna(subset=[
            "feature_sma_diff", "feature_momentum", "feature_volatility",
            "feature_hl_ratio", "feature_rsi", "feature_vol_ratio"
        ])

        if len(df) < 20:
            print("[DataAgent] 有效數據不足 20 行，跳過本輪")
            return None

        return df

    except Exception as e:
        print(f"[DataAgent] 特徵工程失敗: {e}")
        return None


FEATURE_COLS = [
    "feature_sma_diff",
    "feature_momentum",
    "feature_volatility",
    "feature_hl_ratio",
    "feature_rsi",
    "feature_vol_ratio",
]
