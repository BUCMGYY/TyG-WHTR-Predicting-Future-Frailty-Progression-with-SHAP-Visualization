import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

# 加载保存的随机森林模型
model = joblib.load('RF.pkl')

# 特征范围定义（根据提供的特征范围和数据类型）
feature_ranges = {
    "tyg_whtr_w1_": {"type": "numerical", "min": 0.49, "max": 23.73, "default": 10.00},
    "Age": {"type": "numerical", "min": 45.0, "max": 101.0, "default": 50.0},
    "BMI": {"type": "numerical", "min": 5.3, "max": 542.6, "default": 100.0},
    "Education_Status": {"type": "categorical", "options": [1, 2]},
    "Regional_Category": {"type": "categorical", "options": [1, 2, 3]},
    "Cooking_Fuel_Use": {"type": "categorical", "options": [1, 2]},
    "Annual_Household_Expenditure": {"type": "numerical", "min": 98.3, "max": 158132.0, "default": 10000.0},
    "creatinine__2": {"type": "numerical", "min": 0.24, "max": 3.38, "default": 2.00},
    "c.reactive_protein___2": {"type": "numerical", "min": 0.059, "max": 110.19, "default": 50.00},
    "glycated_hemoglobin__2": {"type": "numerical", "min": 3.5, "max": 12.2, "default": 5.0},
    "cystatin_c__2": {"type": "numerical", "min": 0.40, "max": 3.57, "default": 2.00},
    "Depression_category": {"type": "categorical", "options": [1, 2]},
    "egfr_w1_": {"type": "numerical", "min": 18.17, "max": 150.33, "default": 10.00},
}

# Streamlit 界面
st.title("TyG-WHTR: Predicting Future Frailty Progression with SHAP Visualization")

# 动态生成输入项
st.header("Enter the following feature values:")
feature_values = []
for feature, properties in feature_ranges.items():
    if properties["type"] == "numerical":
        value = st.number_input(
            label=f"{feature} ({properties['min']} - {properties['max']})",
            min_value=float(properties["min"]),
            max_value=float(properties["max"]),
            value=float(properties["default"]),
        )
    elif properties["type"] == "categorical":
        value = st.selectbox(
            label=f"{feature} (Select a value)",
            options=properties["options"],
        )
    feature_values.append(value)

# 转换为模型输入格式
features = np.array([feature_values])

# 预测与 SHAP 可视化
if st.button("Predict"):
    # 模型预测
    predicted_class = model.predict(features)[0]
    predicted_proba = model.predict_proba(features)[0]

    # 提取预测的类别概率
    probability = predicted_proba[predicted_class] * 100

    # 显示预测结果，使用 Matplotlib 渲染指定字体
    text = f"Based on feature values, predicted possibility of AKI is {probability:.2f}%"
    fig, ax = plt.subplots(figsize=(8, 1))
    ax.text(
        0.5, 0.5, text,
        fontsize=16,
        ha='center', va='center',
        fontname='Times New Roman',
        transform=ax.transAxes
    )
    ax.axis('off')
    plt.savefig("prediction_text.png", bbox_inches='tight', dpi=300)
    st.image("prediction_text.png")

    # 计算 SHAP 值
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(pd.DataFrame([feature_values], columns=feature_ranges.keys()))

    # 生成 SHAP 力图
    class_index = predicted_class  # 当前预测类别
    shap_fig = shap.force_plot(
        explainer.expected_value[class_index],
        shap_values[:,:,class_index],
        pd.DataFrame([feature_values], columns=feature_ranges.keys()),
        matplotlib=True,
    )
    # 保存并显示 SHAP 图
    plt.savefig("shap_force_plot.png", bbox_inches='tight', dpi=1200)
    st.image("shap_force_plot.png")
