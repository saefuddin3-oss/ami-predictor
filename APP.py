import streamlit as st
import numpy as np

# ==========================================
# 1. 页面配置
# ==========================================
st.set_page_config(
    page_title="AMI 院前延迟风险预测工具",
    page_icon="🚑",
    layout="centered"
)

st.title("🚑 AMI 院前延迟风险预测计算器")
st.markdown("### 基于机器学习逻辑回归模型")
st.info("数据来源：307例急性心肌梗死患者真实临床数据 | 准确率(AUC): 0.741")

# ==========================================
# 2. 侧边栏：特征录入 (严格对应 Table 1 & 2)
# ==========================================
st.sidebar.header("📝 患者特征录入")

# --- 危险因素 (Coef > 0) ---
st.sidebar.subheader("⚠️ 风险指标")

# 1. 症状轻微 (Coef: 1.3831)
# 修复：使用单引号包裹，避免内部双引号冲突
symptom_mild = st.sidebar.radio(
    '1. 发病时是否认为症状"很轻微"？',
    options=[0, 1],
    format_func=lambda x: "是 (Yes)" if x == 1 else "否 (No/严重)",
    index=0
)

# 2. 自救行为 (Coef: 0.9322)
self_relief = st.sidebar.radio(
    "2. 是否尝试过自救 (喝水/休息/按摩)？",
    options=[0, 1],
    format_func=lambda x: "是 (Yes)" if x == 1 else "否 (No)",
    index=0
)

# 3. 前驱症状 (Coef: 0.6475)
prodromal = st.sidebar.radio(
    "3. 发病前是否有前驱症状 (胸闷/乏力)？",
    options=[0, 1],
    format_func=lambda x: "是 (Yes)" if x == 1 else "否 (No)",
    index=0
)

# 4. 就医距离 (Coef: 0.3168)
distance = st.sidebar.selectbox(
    "4. 居住地距离医院的距离等级",
    options=[0, 1, 2, 3, 4],
    format_func=lambda x: f"Level {x} (距离等级 {x})",
    help="参考标准：Level 0 (<5km), Level 1 (5-10km)... 请按实际研究定义选择"
)

st.sidebar.markdown("---")

# --- 保护因素 (Coef < 0) ---
st.sidebar.subheader("🛡️ 保护指标")

# 5. 冠心病史 (Coef: -1.3021)
history_cad = st.sidebar.checkbox("5. 既往有冠心病史 (History of CAD)")
val_cad = 1 if history_cad else 0

# 6. PCI史 (Coef: -0.8703)
history_pci = st.sidebar.checkbox("6. 既往做过支架/PCI手术")
val_pci = 1 if history_pci else 0

# 7. 求助行为 (Coef: -0.4326)
ask_help = st.sidebar.checkbox("7. 发病时立即向他人求助 (Help-seeking)")
val_ask = 1 if ask_help else 0

# ==========================================
# 3. 核心计算 (严核对 Table 2 系数)
# ==========================================

# 截距 (const)
INTERCEPT = -1.3908

# 危险系数 (+)
COEF_MILD = 1.3831
COEF_SELF_RELIEF = 0.9322
COEF_PRODROMAL = 0.6475
COEF_DISTANCE = 0.3168

# 保护系数 (-)
COEF_CAD = -1.3021
COEF_PCI = -0.8703
COEF_ASK = -0.4326

# Logit 公式
logit = (INTERCEPT + 
         (COEF_MILD * symptom_mild) + 
         (COEF_SELF_RELIEF * self_relief) + 
         (COEF_PRODROMAL * prodromal) + 
         (COEF_DISTANCE * distance) + 
         (COEF_CAD * val_cad) + 
         (COEF_PCI * val_pci) + 
         (COEF_ASK * val_ask))

# 概率转换 (Sigmoid Function)
probability = 1 / (1 + np.exp(-logit))

# ==========================================
# 4. 结果展示
# ==========================================
st.markdown("---")
st.subheader("📊 预测结果")

col1, col2 = st.columns([1, 2])

with col1:
    st.metric("延迟概率", f"{probability:.1%}")

with col2:
    if probability < 0.5:
        st.success(f"✅ 低风险 (Low Risk)\n\n患者在6小时内到达医院的可能性较大。")
    else:
        st.error(f"🚨 高风险 (High Risk)\n\n患者极有可能发生院前延迟 (>6h)。\n\n建议：重点干预其自救观念。")

# 详细解释
with st.expander("查看详细风险评分详情"):
    st.write(f"基础分 (Intercept): {INTERCEPT}")
    st.write(f"症状认知影响: {COEF_MILD * symptom_mild:+.4f}")
    st.write(f"自救行为影响: {COEF_SELF_RELIEF * self_relief:+.4f}")
    st.write(f"既往病史保护: {(COEF_CAD * val_cad) + (COEF_PCI * val_pci):+.4f}")
