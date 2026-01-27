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
st.info("💡 说明：本工具基于临床数据训练，用于预测急性心肌梗死(AMI)患者是否能在 **发病 2 小时内** 到达医院。")

# ==========================================
# 2. 侧边栏：特征录入
# ==========================================
st.sidebar.header("📝 患者特征录入")
st.sidebar.subheader("⚠️ 风险指标")

# 1. 症状轻微
symptom_mild = st.sidebar.radio(
    '1. 发病时是否认为症状"很轻微"？',
    options=[0, 1],
    format_func=lambda x: "是 (Yes)" if x == 1 else "否 (No/严重)",
    index=0
)

# 2. 自救行为
self_relief = st.sidebar.radio(
    "2. 是否尝试过自救 (喝水/休息/按摩)？",
    options=[0, 1],
    format_func=lambda x: "是 (Yes)" if x == 1 else "否 (No)",
    index=0
)

# 3. 前驱症状
prodromal = st.sidebar.radio(
    "3. 发病前是否有前驱症状 (胸闷/乏力)？",
    options=[0, 1],
    format_func=lambda x: "是 (Yes)" if x == 1 else "否 (No)",
    index=0
)

# 4. 就医距离
distance = st.sidebar.selectbox(
    "4. 居住地距离医院的距离等级",
    options=[0, 1, 2, 3, 4], 
    format_func=lambda x: f"Level {x} (距离等级 {x})",
    help="参考标准：Level 0 (<5km), Level 1 (5-10km)..."
)

st.sidebar.markdown("---")
st.sidebar.subheader("🛡️ 保护指标")

# 5. 冠心病史
history_cad = st.sidebar.checkbox("5. 既往有冠心病史 (History of CAD)")
val_cad = 1 if history_cad else 0

# 6. PCI史
history_pci = st.sidebar.checkbox("6. 既往做过支架/PCI手术")
val_pci = 1 if history_pci else 0

# 7. 求助行为
ask_help = st.sidebar.checkbox("7. 发病时立即向他人求助 (Help-seeking)")
val_ask = 1 if ask_help else 0

# ==========================================
# 3. 核心计算
# ==========================================
INTERCEPT = -1.3908
COEF_MILD = 1.3831
COEF_SELF_RELIEF = 0.9322
COEF_PRODROMAL = 0.6475
COEF_DISTANCE = 0.3168
COEF_CAD = -1.3021
COEF_PCI = -0.8703
COEF_ASK = -0.4326

logit = (INTERCEPT + 
         (COEF_MILD * symptom_mild) + 
         (COEF_SELF_RELIEF * self_relief) + 
         (COEF_PRODROMAL * prodromal) + 
         (COEF_DISTANCE * distance) + 
         (COEF_CAD * val_cad) + 
         (COEF_PCI * val_pci) + 
         (COEF_ASK * val_ask))

probability = 1 / (1 + np.exp(-logit))

# ==========================================
# 4. 结果展示 (升级为三级风险)
# ==========================================
st.markdown("---")
st.subheader("📊 预测结果分析")

col1, col2 = st.columns([1, 2])

with col1:
    st.metric("延迟 (>2h) 概率", f"{probability:.1%}")
    
    # 简单的红绿灯视觉提示
    if probability < 0.35:
        st.write("🟢 风险较低")
    elif probability < 0.65:
        st.write("🟡 风险中等")
    else:
        st.write("🔴 风险极高")

with col2:
    # === 1. 低风险 (< 35%) ===
    if probability < 0.35:
        st.success(f"✅ **低风险 (Low Risk)**")
        st.markdown(f"""
        **预测**：患者能够及时到达医院的可能性较大。
        
        **💡 建议**：
        * 保持当前的警惕性。
        * **即使症状不重，也建议去社区医院做个心电图**，排除隐患。
        * 保持通讯畅通。
        """)

    # === 2. 中风险 (35% - 65%) [新增] ===
    elif probability < 0.65:
        st.warning(f"⚠️ **中风险 (Medium Risk)**")
        st.markdown(f"""
        **预测**：患者处于 **“犹豫期”**，非常有可能会拖延超过2小时。
        
        **💡 关键干预**：
        * 您的特征显示您可能正在犹豫（如症状不典型或想观察一下）。
        * **不要赌博！** 心梗的症状往往具有欺骗性。
        * **行动指令**：不要再等了，马上出发去医院。早去一小时，结果截然不同。
        """)

    # === 3. 高风险 (> 65%) ===
    else:
        st.error(f"🚨 **高风险 (High Risk)**")
        st.markdown(f"""
        **预测**：患者极大概率会发生严重延迟 (>2小时)。
        
        **🔥 红色警报**：
        * **高度危险！** 您具备多个容易导致拖延的特征（如忽视轻微症状、距离远或错误自救）。
        * **立即停止自救**：喝水、拍打、休息对心梗无效！
        * **唯一正确的做法**：立刻拨打 120，告知可能是心梗，要求救护车送至最近的胸痛中心。
        """)

# 详细解释
with st.expander("查看详细风险评分详情"):
    st.write("该评分基于 Logistic 回归系数计算：")
    st.write(f"基础分: {INTERCEPT}")
    st.write(f"症状认知(轻微)加分: {COEF_MILD * symptom_mild:+.4f}")
    st.write(f"自救行为加分: {COEF_SELF_RELIEF * self_relief:+.4f}")
    st.write(f"居住距离加分: {COEF_DISTANCE * distance:+.4f}")
    st.write(f"保护因素(病史/求助)减分: {(COEF_CAD * val_cad) + (COEF_PCI * val_pci) + (COEF_ASK * val_ask):+.4f}")
