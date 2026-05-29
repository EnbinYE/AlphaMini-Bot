Markdown
# AlphaMini-Bot: AI-Driven Quant Monitoring & Alert System

[![Python Version](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/)
[![Docker](https://img.shields.io/badge/docker-enabled-blue.svg)](https://www.docker.com/)
[![Deployment](https://img.shields.io/badge/GCP-Hong%20Kong-orange.svg)](https://cloud.google.com/)

一個生產級別的 AI 量化盯盤與信號通知智能體。本項目基於機器學習模型對金融市場（Crypto/Stock）進行實時數據分析與走勢預測，並在觸發交易信號時，通過雲端架構將高價值警報實時精準推送到用戶的 Telegram 手機端。

項目完整實現了從**本地開發、數據清洗、模型預測、Docker 容器化**到 **GCP（Google Cloud Platform）雲端機房 24/7 自動化運維**的完整 MLOps 流程。

---

## 🚀 核心架構與技術棧

* **Data Agent (數據工程):** 基於 `Pandas` 與 `NumPy` 實現高速時序數據特徵工程，完美處理 Rolling Windows 與避免數據洩漏（Data Leakage）。
* **AI Engine (預測核心):** 集成機器學習預測模型（Random Forest / XGBoost），動態評估多維度技術指標並實時輸出方向性預測。
* **Notification Engine (通知信號):** 基於 Telegram Bot API 封鎖異步通訊模塊，打通低延遲的風控與信號通知線。
* **Infrastructure (基礎設施):**
    * **Docker:** 實現環境無縫封裝，確保本地與雲端環境高度一致。
    * **GCP (GCP Hong Kong Region):** 部署於香港雲端核心機房，保證對亞太金融市場數據源的低延遲監聽與 24/7 不間斷穩定運行。

---

## 📂 項目文件結構

```text
AlphaMini-Bot/
├── main.py             # 項目啟動主入口，調度數據與模型核心
├── data_agent.py       # 數據獲取、時序特徵工程與數據清洗模塊
├── model/              # AI / 機器學習模型訓練與預測核心
├── Dockerfile          # Docker 鏡像構建配置文件
├── .gitignore          # Git 忽略清單（嚴格隔離敏感私鑰）
├── .env.example        # 環境變量配置模板（開源引導）
└── README.md           # 項目說明文檔
🛠️ 生產環境部署指南（GCP + Docker）
本項目採用 Docker 容器化技術，支持在任何雲端虛擬機（Ubuntu/Debian）上一鍵滾動更新。

1. 配置環境變量
複製 .env.example 並命名為 .env，填入你的私鑰信息（此文件已被 .gitignore 保護）：

Bash
TELEGRAM_TOKEN=your_bot_token_here
CHAT_ID=your_chat_id_here
2. 構建 Docker 鏡像
在項目根目錄下執行：

Bash
docker build -t alphamini-bot .
3. 24/7 後台持久化運行
Bash
docker run -d --name my_quant_bot --restart always alphamini-bot
4. 運維與日誌監控
Bash
# 實時查看盯盤日誌與預測信號
docker logs -f my_quant_bot
📈 未來迭代方向
[ ] 增加多因子模型（Multi-Factor Models）特徵輸入。

[ ] 對接幣安 (Binance) / 富途 (Futu) WebSocket 實時流式數據源。

[ ] 引入動態止盈止損與基於凱利公式的倉位風控模塊。

Disclaimer: 本項目僅用於 AI 量化工程技術展示與學術交流，不構成任何實質性投資建議。