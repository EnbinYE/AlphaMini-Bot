"""
strategy_agent.py — 機器學習模型與預測

優化點：
- 加入 5-fold 交叉驗證，低於閾值時拒絕輸出預測
- 信心度過濾：prob < 0.55 時返回 None，表示「觀望」
- 嚴格執行前向偏差保護：訓練集排除最後一行
- 返回值新增 cv_accuracy 供調用方決策
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

# 信心度閾值：低於此值視為觀望
CONFIDENCE_THRESHOLD = 0.55

# 交叉驗證準確率下限：低於此值說明模型接近隨機
CV_ACCURACY_FLOOR = 0.52


def train_and_predict(
    df,
    feature_cols: list[str],
    confidence_threshold: float = CONFIDENCE_THRESHOLD,
) -> tuple[int | None, float, float]:
    """
    滾動訓練 RandomForest 並預測下一分鐘漲跌。

    返回：
        prediction  : 1（看漲）| 0（看跌）| None（觀望/模型不可信）
        prob        : 預測概率（信心度），觀望時為 0.0
        cv_accuracy : 5-fold 交叉驗證均值，供外部風控參考

    ⚠️ 前向偏差保護：
        最後一行的 target 為 NaN（由 data_agent 顯式標記）。
        訓練集使用 df[:-1]，最後一行僅取 X 用於推理，絕不參與訓練。
    """
    # --- 分離訓練集與當前觀測值 ---
    train_df = df.dropna(subset=["target"])          # 排除 target=NaN 的最後一行
    current_row = df.iloc[[-1]]                      # 最後一行（target=NaN，僅供推理）

    X_train = train_df[feature_cols].values
    y_train = train_df["target"].values.astype(int)
    X_current = current_row[feature_cols].values

    if len(X_train) < 15:
        print("[StrategyAgent] 訓練樣本不足，跳過本輪")
        return None, 0.0, 0.0

    # --- 初始化模型（限制深度防止過擬合）---
    model = RandomForestClassifier(
        n_estimators=50,
        max_depth=3,
        min_samples_leaf=5,   # 新增：葉節點最少 5 樣本，進一步防過擬合
        random_state=42,
        n_jobs=-1,
    )

    # --- 5-fold 交叉驗證（在訓練集上評估泛化能力）---
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="accuracy")
    cv_accuracy = cv_scores.mean()

    if cv_accuracy < CV_ACCURACY_FLOOR:
        print(f"[StrategyAgent] 模型 CV 準確率 {cv_accuracy:.3f} 低於閾值，本輪觀望")
        return None, 0.0, cv_accuracy

    # --- 在全量訓練集上正式訓練，然後推理 ---
    model.fit(X_train, y_train)
    prediction = model.predict(X_current)[0]
    prob = model.predict_proba(X_current)[0][prediction]

    # --- 信心度過濾 ---
    if prob < confidence_threshold:
        print(f"[StrategyAgent] 信心度 {prob:.3f} 不足，本輪觀望")
        return None, prob, cv_accuracy

    return int(prediction), float(prob), float(cv_accuracy)
