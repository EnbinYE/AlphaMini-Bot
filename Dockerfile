# 使用輕量級 Python 鏡像
FROM python:3.12-slim

# 設置工作目錄
WORKDIR /app

# 複製依賴清單並安裝
RUN pip install --no-cache-dir requests pandas scikit-learn xgboost python-dotenv

# 複製當前目錄下的所有代碼到容器中
COPY . .

# 運行主程序
CMD ["python", "main.py"]