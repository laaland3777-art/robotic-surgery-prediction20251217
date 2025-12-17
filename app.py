import streamlit as st
import pandas as pd
import numpy as np
import joblib

# --- 页面配置 ---
st.set_page_config(
    page_title="Robotic Surgery Difficulty Prediction",
    layout="centered"
)

# --- 加载模型和标准化器 ---
@st.cache_resource
def load_model_and_scaler():
    try:
        # 注意这里的文件名要和你上传的一致
        model = joblib.load('rob_ensemble_model.pkl')
        scaler = joblib.load('rob_scaler.pkl')
        return model, scaler
    except FileNotFoundError:
        st.error("Error: Model files not found. Please ensure 'rob_ensemble_model.pkl' and 'rob_scaler.pkl' are in the same directory.")
        return None, None

model, scaler = load_model_and_scaler()

# --- 标题和介绍 ---
st.title("Robotic Surgery Difficulty Prediction Model")
st.markdown("""
This application predicts the difficulty probability of **robotic surgery** based on preoperative clinical features.
Please input the patient's parameters below.
""")

st.markdown("---")

# --- 输入表单 ---
st.subheader("Patient Features Input")

# 创建两列布局
col1, col2 = st.columns(2)

with col1:
    # 1. Abdominal wall adipose area
    f1 = st.number_input(
        "Abdominal wall adipose area (cm²)", 
        min_value=0.0, 
        value=100.0, 
        step=1.0,
        help="Area of subcutaneous fat in the abdominal wall."
    )
    
    # 2. SMV adipose thickness
    f2 = st.number_input(
        "SMV adipose thickness (mm)", 
        min_value=0.0, 
        value=10.0, 
        step=0.1,
        help="Thickness of adipose tissue around the Superior Mesenteric Vein."
    )
    
    # 3. Intra-abdominal adipose area
    f3 = st.number_input(
        "Intra-abdominal adipose area (cm²)", 
        min_value=0.0, 
        value=150.0, 
        step=1.0,
        help="Visceral fat area."
    )

with col2:
    # 4. History of abdominal surgery
    # 这是一个二分类变量 (0/1)
    f4_display = st.selectbox(
        "History of abdominal surgery",
        options=["No (0)", "Yes (1)"],
        index=0,
        help="Does the patient have a history of prior abdominal surgeries?"
    )
    f4 = 1 if "Yes" in f4_display else 0
    
    # 5. Plasma triglycerides
    f5 = st.number_input(
        "Plasma triglycerides (mmol/L)", 
        min_value=0.0, 
        value=1.5, 
        step=0.1
    )

# --- 预测逻辑 ---
if st.button("Predict Difficulty", type="primary"):
    if model is not None and scaler is not None:
        # 1. 构造输入 DataFrame (必须与训练时的列名一致)
        feature_names = [
            'Abdominal wall adipose area',
            'SMV adipose thickness',
            'Intra-abdominal adipose area',
            'History of abdominal surgery',
            'Plasma triglycerides'
        ]
        
        # 注意顺序必须严格对应 feature_names
        input_data = pd.DataFrame([[f1, f2, f3, f4, f5]], columns=feature_names)
        
        # 2. 标准化
        input_scaled = scaler.transform(input_data)
        
        # 3. 预测概率
        # model.predict_proba 返回 [[prob_class_0, prob_class_1]]
        probability = model.predict_proba(input_scaled)[0][1]
        prediction_class = 1 if probability >= 0.5 else 0
        
        # --- 显示结果 ---
        st.markdown("---")
        st.subheader("Prediction Result")
        
        # 进度条显示概率
        st.progress(probability)
        
        result_col1, result_col2 = st.columns(2)
        
        with result_col1:
            st.metric(label="Difficulty Probability", value=f"{probability:.2%}")
            
        with result_col2:
            if prediction_class == 1:
                st.error("High Difficulty Predicted")
            else:
                st.success("Low Difficulty Predicted")
                
        st.info(f"The model predicts a **{probability:.1%}** chance of the robotic surgery being difficult.")

# --- 页脚 ---
st.markdown("---")
st.caption("Model based on Weighted Ensemble (LR, AdaBoost, SVM, MLP, XGBoost).")
