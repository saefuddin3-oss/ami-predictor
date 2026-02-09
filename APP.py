import streamlit as st
import numpy as np

# ==========================================
# 1. 页面基本设置：温馨、简洁
# ==========================================
st.set_page_config(
    page_title="心肌梗死急救时间评估助手", 
    page_icon="❤️",
    layout="centered"
)

# 标题区：不使用"预测模型"这种高冷词汇，改用更亲民的表达
st.title("❤️ 胸痛急救时间评估助手")
st.markdown("### ⏱️ 帮您判断：我现在去医院会不会太晚？")
st.info("💡 **温馨提示**：本工具基于临床大数据分析，专门帮助犹豫不决的患者。如果您现在胸痛难忍，请直接拨打 **120**，不要测试了！")

# ==========================================
# 2. 侧边栏：像医生问诊一样的交互
# ==========================================
st.sidebar.header("📋 请告诉我们您的当前情况")

# --- 核心风险 1：自救行为 (权重最高 OR=3.01) ---
st.sidebar.subheader("1. 发病后的反应")
self_relief_input = st.sidebar.radio(
    "🔴 您（或患者）在感到不舒服后，尝试过自己处理吗？",
    options=["没有，立刻想去医院", "是的，尝试过自己缓解"],
    help="👉 **包括这些行为**：喝温水、躺下休息睡觉、用力咳嗽、找人按摩捶背、或者自己吃了点胃药/救心丸观察情况。",
    index=0
)
# 转换为模型数值
self_relief = 1 if "是的" in self_relief_input else 0


# --- 核心风险 2：症状认知 (权重次高 OR=2.58) ---
st.sidebar.subheader("2. 对疼痛的自我感觉")
symptom_mild_input = st.sidebar.radio(
    "🔴 您觉得现在的难受程度属于“轻微”吗？",
    options=["不，非常难受/痛得冒汗", "觉得还好，能忍受"],
    help="👉 **怎么算轻微？** 比如觉得像消化不良、胃疼、或者只是胸口有点闷，感觉“歇一歇就能好”，没有那种压榨性的剧痛。",
    index=0
)
symptom_mild = 1 if "觉得还好" in symptom_mild_input else 0


# --- 核心风险 3：距离 (权重 OR=1.82) ---
# 用户要求：1代表5km以下，其他合理分配
st.sidebar.subheader("3. 距离医院的路程")
distance_label = st.sidebar.select_slider(
    "🚗 从您现在的位置去最近的大医院，大概多远？",
    options=[
        "非常近 (< 5公里)", 
        "比较近 (5-10公里)", 
        "中等距离 (10-20公里)", 
        "比较远 (> 20公里)"
    ],
    value="非常近 (< 5公里)"
)

# 将文字转换为模型需要的 1, 2, 3, 4
if "非常近" in distance_label:
    distance_score = 1
elif "比较近" in distance_label:
    distance_score = 2
elif "中等距离" in distance_label:
    distance_score = 3
else:
    distance_score = 4


# --- 核心风险 4：前驱症状 (权重 OR=1.64) ---
st.sidebar.subheader("4. 发病前的预兆")
prodromal_input = st.sidebar.radio(
    "⚠️ 这两天发病前，身体有没有发出过“预警信号”？",
    options=["完全没有，突然发作", "有的，前几天就不舒服"],
    help="👉 **预警信号包括**：莫名的疲劳乏力、活动后胸闷气短、或者短暂的心慌心痛。",
    index=0
)
prodromal = 1 if "有的" in prodromal_input else 0


# --- 保护因素：病史 (权重 OR < 1) ---
st.sidebar.markdown("---")
st.sidebar.subheader("5. 既往病史")
st.sidebar.caption("如果您是心脏病老病号，通常警惕性会更高：")

history_cad = st.sidebar.checkbox("✅ 以前医生确诊过我有“冠心病”")
val_cad = 1 if history_cad else 0

history_pci = st.sidebar.checkbox("✅ 我以前做过心脏支架(PCI)手术")
val_pci = 1 if history_pci else 0


# ==========================================
# 3. 后台计算 (严格对应优化后的 OR 值)
# ==========================================
# 系数 = ln(OR)
COEF_SELF_RELIEF = 1.10  # ln(3.01) - 喝水休息导致延迟
COEF_MILD = 0.95         # ln(2.58) - 觉得不疼导致延迟
COEF_DISTANCE = 0.60     # ln(1.82) - 距离每增加一级
COEF_PRODROMAL = 0.49    # ln(1.64) - 有前驱症状
COEF_PCI = -0.76         # ln(0.47) - 做过支架(保护)
COEF_CAD = -1.31         # ln(0.27) - 有冠心病(保护)
INTERCEPT = -2.19        # 基础截距

# 计算 Logit
logit = (INTERCEPT + 
         (COEF_SELF_RELIEF * self_relief) + 
         (COEF_MILD * symptom_mild) + 
         (COEF_DISTANCE * distance_score) +  
         (COEF_PRODROMAL * prodromal) + 
         (COEF_PCI * val_pci) + 
         (COEF_CAD * val_cad))

# 转换为概率
probability = 1 / (1 + np.exp(-logit))

# ==========================================
# 4. 结果展示：情感化 + 警示化
# ==========================================
st.markdown("---")

# 根据概率划分三个等级的文案
if probability < 0.35:
    risk_level = "低风险"
    color = "green"
    icon = "🟢"
    msg_title = "很有可能及时赶到"
elif probability < 0.65:
    risk_level = "中风险"
    color = "orange"
    icon = "🟡"
    msg_title = "您可能正在犹豫，请立即行动！"
else:
    risk_level = "高风险"
    color = "red"
    icon = "🚨"
    msg_title = "极度危险！可能会严重延误！"

# 使用卡片式展示
with st.container():
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown(f"### {icon} {risk_level}")
        st.metric("延迟概率 (>2小时)", f"{probability:.1%}")
        st.caption("注：概率越低越好")
        
    with col2:
        st.markdown(f"#### {msg_title}")
        
        if risk_level == "高风险":
            st.error("""
            **为什么会这样？**
            数据分析发现，您可能犯了两个最大的错误：
            1. ❌ **试图自救**：喝水、休息、按摩对心梗**无效**，只会浪费救命时间！
            2. ❌ **低估病情**：心梗不一定都痛得死去活来，**不痛不代表没事**。
            
            **👉 马上行动：停止观察，立刻去医院！**
            """)
        elif risk_level == "中风险":
            st.warning("""
            **您现在的状态很典型：**
            您可能觉得“再观察一下”、“等天亮再说”或者“不想麻烦家人”。
            
            **👉 医生建议**：
            心肌细胞死亡是不可逆的。既然已经有点不舒服，**宁可去医院查出没事，也不要在家赌运气。**
            """)
        else:
            st.success("""
            **做得很好！**
            您的警惕性很高（或者距离医院很近）。
            
            **👉 保持现状**：
            请继续保持这种警觉性。如果症状持续不缓解，即便风险评分低，也建议去急诊科做个心电图求个心安。
            """)

# ==========================================
# 5. 底部详细解释 (可选)
# ==========================================
with st.expander("ℹ️ 为什么问这些问题？(点击查看科学依据)"):
    st.markdown("""
    本测试基于 294 例心梗患者的真实数据分析（AUC=0.72），各因素影响如下：
    
    * **喝水/休息/按摩 (自救行为)**：这是最危险的因素！会让您平均多耽误 **3倍** 的时间。
    * **觉得症状轻微**：许多患者因为觉得“还能忍”就不去医院，导致风险增加 **2.5倍**。
    * **距离**：距离越远，客观路程时间越长，每增加一级距离，风险增加约 **1.8倍**。
    * **既往病史**：如果您以前得过病（冠心病/支架），通常会更有经验，反而能更快去医院。
    """)
