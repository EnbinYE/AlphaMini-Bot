# AlphaMini-Bot — 優化版部署指南

## 文件結構

```
alphamini/
├── data_agent.py      # 數據抓取 + 特徵工程
├── strategy_agent.py  # RandomForest 模型 + 預測
├── main.py            # 主循環 + 風控 + Telegram 報警
├── .env               # 敏感配置（不要上傳 Git）
└── requirements.txt
```

---

## 環境安裝

```bash
pip install requests pandas numpy scikit-learn python-dotenv
```

---

## 配置 Telegram Bot

1. 在 Telegram 搜索 `@BotFather`，發送 `/newbot`，拿到 `Token`
2. 搜索 `@userinfobot`，拿到你的 `Chat ID`
3. 創建 `.env` 文件（與 `main.py` 同目錄）：

```
TELEGRAM_TOKEN=你的Token
CHAT_ID=你的ChatID
```

4. 在 `main.py` 頂部加入（已在代碼中標注）：

```python
from dotenv import load_dotenv
load_dotenv()
```

---

## 本地運行

```bash
python3 main.py
```

---

## 部署到 GCP 香港服務器

### 方式一：nohup（快速）

```bash
export TELEGRAM_TOKEN="your_token"
export CHAT_ID="your_chat_id"
nohup python3 main.py > alphamini.log 2>&1 &
echo $! > alphamini.pid   # 保存進程 ID 方便後續停止
```

停止：
```bash
kill $(cat alphamini.pid)
```

### 方式二：systemd（推薦，服務器重啟後自動拉起）

創建 `/etc/systemd/system/alphamini.service`：

```ini
[Unit]
Description=AlphaMini Quant Bot
After=network.target

[Service]
User=your_username
WorkingDirectory=/home/your_username/alphamini
EnvironmentFile=/home/your_username/alphamini/.env
ExecStart=/usr/bin/python3 main.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

啟動：
```bash
sudo systemctl daemon-reload
sudo systemctl enable alphamini
sudo systemctl start alphamini
sudo systemctl status alphamini
```

查看日誌：
```bash
journalctl -u alphamini -f
```

---

## 優化對比

| 模塊 | 原版 | 優化版 |
|------|------|--------|
| 特徵數量 | 3（SMA差、波動率、動量）| 6（新增 RSI、成交量比率、高低波幅）|
| 前向偏差保護 | 隱式依賴調用順序 | 顯式標記最後行 target=NaN |
| 模型評估 | 無 | 5-fold CV，低於 0.52 報警 |
| 信心度過濾 | 無 | prob < 0.55 輸出「觀望」|
| 數據抓取 | 失敗直接跳過 | 重試 3 次 + 熔斷報警 |
| 橫盤過濾 | 無 | 價格變化 < 0.05% 不計入對錯 |
| 報警去重 | 連續報警不停止 | 報警後重置計數器 |
| 敏感信息 | 硬編碼在源文件 | 環境變量 / .env 文件 |

---

## 進一步迭代建議

- **換週期**：`INTERVAL = "5m"` 信噪比更高，適合作為下一步測試
- **加回測**：部署前用 `/api/v3/klines` 拉取 1000 根歷史 K 線跑一遍準確率分布
- **加「不操作」邏輯**：`prob < 0.55` 已實現，可進一步調高閾值至 0.6 觀察效果
- **日誌持久化**：可將每輪結果寫入 CSV，方便事後分析模型表現
