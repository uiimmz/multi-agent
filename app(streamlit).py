"""
Streamlit 网页端界面
多Agent电商商品多模态智能解析系统 - 浅色暖调主题
"""
import sys
import os
import json
import time
import tempfile
from datetime import datetime

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
from config import THRESHOLD_CONFIG, ABLATION_CONFIG, ABLATION_PRESETS, DEEPSEEK_BASE_URL, DEEPSEEK_API_KEY, DEEPSEEK_MODEL
from state import create_initial_state
from agents.main_agent import planning_node, aggregation_node
from agents.vision_agent import vision_agent
from agents.copy_agent import copy_agent
from agents.verify_agent import verify_agent
from utils import image_to_base64, add_log

# ============================================================
# 页面配置
# ============================================================

st.set_page_config(
    page_title="多Agent智能解析系统",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============================================================
# 自定义CSS - Soft Editorial 暖调简约风格
# ============================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;600;700&family=Noto+Sans+SC:wght@300;400;500;700&display=swap');

:root {
    --bg-primary: #0f1117;
    --bg-secondary: #1a1d29;
    --bg-tertiary: #252836;
    --bg-card: #1e2130;
    --accent-cyan: #00d4aa;
    --accent-blue: #3b82f6;
    --accent-purple: #8b5cf6;
    --accent-orange: #f59e0b;
    --accent-red: #ef4444;
    --accent-green: #22c55e;
    --text-primary: #e8eaed;
    --text-secondary: #9ca3af;
    --text-muted: #6b7280;
    --border-color: #2d3148;
    --glow-cyan: 0 0 20px rgba(0, 212, 170, 0.3);
    --glow-blue: 0 0 20px rgba(59, 130, 246, 0.3);
}

.stApp {
    background: var(--bg-primary) !important;
    color: var(--text-primary) !important;
}

.stApp > header { background: transparent !important; }

.main .block-container {
    padding-top: 2rem !important;
    padding-bottom: 2rem !important;
    max-width: 100% !important;
}

h1, h2, h3, h4, h5, h6 {
    font-family: 'Noto Sans SC', sans-serif !important;
    color: var(--text-primary) !important;
}

h1 {
    background: linear-gradient(135deg, var(--accent-cyan), var(--accent-blue)) !important;
    -webkit-background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
    background-clip: text !important;
    font-weight: 700 !important;
}

h2 {
    font-size: 1.2rem !important;
    border-bottom: 1px solid var(--border-color) !important;
    padding-bottom: 0.5rem !important;
}

div[data-testid="stVerticalBlock"] > div[data-testid="stVerticalBlockBorderWrapper"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border-color) !important;
    border-radius: 16px !important;
    padding: 1.5rem !important;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3) !important;
}

div[data-testid="stVerticalBlock"] > div[data-testid="stVerticalBlockBorderWrapper"]:hover {
    border-color: var(--accent-cyan) !important;
    box-shadow: var(--glow-cyan) !important;
}

.stButton > button {
    background: linear-gradient(135deg, var(--accent-cyan), var(--accent-blue)) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.75rem 2rem !important;
    font-family: 'Noto Sans SC', sans-serif !important;
    font-weight: 600 !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 15px rgba(0, 212, 170, 0.3) !important;
    width: 100% !important;
}

.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 25px rgba(0, 212, 170, 0.4) !important;
}

.stTextArea > div > div > textarea,
.stTextInput > div > div > input {
    background: var(--bg-tertiary) !important;
    border: 1px solid var(--border-color) !important;
    border-radius: 12px !important;
    color: var(--text-primary) !important;
    font-family: 'Noto Sans SC', sans-serif !important;
}

.stTextArea > div > div > textarea:focus,
.stTextInput > div > div > input:focus {
    border-color: var(--accent-cyan) !important;
    box-shadow: 0 0 0 2px rgba(0, 212, 170, 0.2) !important;
}

[data-testid="stFileUploader"] {
    background: var(--bg-tertiary) !important;
    border: 2px dashed var(--border-color) !important;
    border-radius: 16px !important;
}

[data-testid="stFileUploader"]:hover {
    border-color: var(--accent-cyan) !important;
}

.stSelectbox > div > div {
    background: var(--bg-tertiary) !important;
    border: 1px solid var(--border-color) !important;
    border-radius: 12px !important;
    color: var(--text-primary) !important;
}

.stSlider > div > div > div > div {
    background: var(--accent-cyan) !important;
}

.stNumberInput > div > div > input {
    background: var(--bg-tertiary) !important;
    border: 1px solid var(--border-color) !important;
    border-radius: 12px !important;
    color: var(--text-primary) !important;
}

.streamlit-expanderHeader {
    background: var(--bg-tertiary) !important;
    border-radius: 12px !important;
    border: 1px solid var(--border-color) !important;
    color: var(--text-primary) !important;
}

.streamlit-expanderContent {
    background: var(--bg-secondary) !important;
    border: 1px solid var(--border-color) !important;
    border-top: none !important;
    border-radius: 0 0 12px 12px !important;
}

[data-testid="stMetric"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border-color) !important;
    border-radius: 12px !important;
}

[data-testid="stMetric"] label { color: var(--text-secondary) !important; }
[data-testid="stMetric"] [data-testid="stMetricValue"] {
    color: var(--accent-cyan) !important;
    font-family: 'JetBrains Mono', monospace !important;
}

hr { border-color: var(--border-color) !important; }
.stAlert { border-radius: 12px !important; }

#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* 流水线 */
.pipeline-container {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 0;
    padding: 1.5rem 1rem;
    background: var(--bg-card);
    border: 1px solid var(--border-color);
    border-radius: 16px;
    margin: 1rem 0;
    overflow-x: auto;
}

.pipeline-step {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 0.5rem;
    min-width: 70px;
}

.pipeline-node {
    width: 52px;
    height: 52px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.1rem;
    font-weight: 600;
    transition: all 0.4s ease;
}

.pipeline-node.pending {
    background: var(--bg-tertiary);
    border: 2px solid var(--border-color);
    color: var(--text-muted);
}

.pipeline-node.active {
    background: rgba(0, 212, 170, 0.15);
    border: 2px solid var(--accent-cyan);
    color: var(--accent-cyan);
    animation: node-pulse 2s infinite;
}

.pipeline-node.completed {
    background: rgba(34, 197, 94, 0.15);
    border: 2px solid var(--accent-green);
    color: var(--accent-green);
}

.pipeline-node-label {
    font-size: 0.72rem;
    color: var(--text-secondary);
    font-family: 'Noto Sans SC', sans-serif;
}

.pipeline-arrow {
    width: 36px;
    height: 2px;
    background: var(--border-color);
    margin: 0 0.2rem;
    position: relative;
    align-self: center;
    margin-bottom: 1.5rem;
}

.pipeline-arrow::after {
    content: '';
    position: absolute;
    right: -3px;
    top: -4px;
    border-left: 7px solid var(--border-color);
    border-top: 5px solid transparent;
    border-bottom: 5px solid transparent;
}

.pipeline-arrow.active {
    background: var(--accent-green);
}
.pipeline-arrow.active::after {
    border-left-color: var(--accent-green);
}

@keyframes node-pulse {
    0%, 100% { box-shadow: 0 0 5px rgba(0, 212, 170, 0.3); }
    50% { box-shadow: 0 0 25px rgba(0, 212, 170, 0.6); }
}

/* 仪表盘 */
.gauge-card {
    text-align: center;
    padding: 1rem 0.5rem;
    background: var(--bg-card);
    border: 1px solid var(--border-color);
    border-radius: 14px;
    transition: all 0.3s ease;
}
.gauge-card:hover {
    box-shadow: var(--glow-cyan);
    transform: translateY(-2px);
}
.gauge-value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.5rem;
    font-weight: 700;
    margin-top: -0.3rem;
}
.gauge-label {
    font-family: 'Noto Sans SC', sans-serif;
    color: var(--text-secondary);
    font-size: 0.78rem;
    margin-top: 0.2rem;
}

/* Agent卡片 */
.agent-detail-card {
    background: var(--bg-card);
    border: 1px solid var(--border-color);
    border-radius: 14px;
    padding: 1.3rem;
    margin-bottom: 1rem;
    transition: all 0.3s ease;
}
.agent-detail-card:hover {
    border-color: var(--accent-cyan);
    box-shadow: var(--glow-cyan);
}
.agent-detail-header {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    margin-bottom: 1rem;
    padding-bottom: 0.75rem;
    border-bottom: 1px solid var(--border-color);
}
.agent-icon {
    width: 38px;
    height: 38px;
    border-radius: 10px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.1rem;
}
.agent-icon.vision { background: rgba(139, 92, 246, 0.2); }
.agent-icon.copy { background: rgba(59, 130, 246, 0.2); }
.agent-icon.verify { background: rgba(245, 158, 11, 0.2); }
.agent-detail-title {
    font-family: 'Noto Sans SC', sans-serif;
    font-weight: 600;
    font-size: 0.95rem;
    color: var(--text-primary);
}
.agent-detail-badge {
    display: inline-block;
    padding: 0.2rem 0.6rem;
    border-radius: 20px;
    font-size: 0.7rem;
    font-weight: 600;
    font-family: 'JetBrains Mono', monospace;
    margin-left: auto;
}
.badge-success { background: rgba(34, 197, 94, 0.2); color: var(--accent-green); }
.badge-warning { background: rgba(245, 158, 11, 0.2); color: var(--accent-orange); }
.badge-info { background: rgba(59, 130, 246, 0.2); color: var(--accent-blue); }

/* 日志 */
.log-timeline {
    border-left: 3px solid var(--border-color);
    padding-left: 1.5rem;
    margin-left: 0.5rem;
}
.log-entry {
    position: relative;
    margin-bottom: 0.6rem;
    padding: 0.5rem 0.8rem;
    background: var(--bg-tertiary);
    border-radius: 8px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.75rem;
    color: var(--text-secondary);
}
.log-entry::before {
    content: '';
    position: absolute;
    left: -1.85rem;
    top: 50%;
    transform: translateY(-50%);
    width: 10px;
    height: 10px;
    border-radius: 50%;
    background: var(--accent-cyan);
    border: 2px solid var(--bg-primary);
}
.log-time { color: var(--accent-cyan); font-size: 0.7rem; }
.log-msg { margin-top: 0.15rem; line-height: 1.4; }

/* 最终输出 */
.final-output-card {
    background: linear-gradient(135deg, var(--bg-card), var(--bg-tertiary));
    border: 1px solid var(--accent-cyan);
    border-radius: 16px;
    padding: 1.5rem;
    line-height: 1.8;
    font-family: 'Noto Sans SC', sans-serif;
    font-size: 0.92rem;
    color: var(--text-primary);
    box-shadow: var(--glow-cyan);
}

/* 空状态 */
.empty-state {
    text-align: center;
    padding: 4rem 2rem;
    color: var(--text-muted);
}
.empty-state .empty-icon { font-size: 3.5rem; margin-bottom: 1rem; opacity: 0.6; }
.empty-state h3 { color: var(--text-secondary) !important; margin-bottom: 0.5rem !important; }
.empty-state p { font-size: 0.9rem; }

/* 标签 */
.tag {
    display: inline-block;
    padding: 0.15rem 0.5rem;
    border-radius: 16px;
    font-size: 0.68rem;
    font-weight: 600;
    font-family: 'JetBrains Mono', monospace;
}
.tag-vision { background: rgba(139, 92, 246, 0.2); color: var(--accent-purple); }
.tag-copy { background: rgba(59, 130, 246, 0.2); color: var(--accent-blue); }
.tag-verify { background: rgba(245, 158, 11, 0.2); color: var(--accent-orange); }
.tag-main { background: rgba(0, 212, 170, 0.2); color: var(--accent-cyan); }

@media (max-width: 768px) {
    .main .block-container { padding: 1rem !important; }
    .pipeline-container { flex-wrap: wrap; }
    .pipeline-arrow { width: 20px; }
}
</style>
""", unsafe_allow_html=True)


# ============================================================
# 辅助函数
# ============================================================

def get_confidence_color(value: float) -> str:
    if value >= 0.8: return "#6a9b6a"
    elif value >= 0.6: return "#d4a574"
    else: return "#dc5050"


def render_gauge(value: float, label: str, size: int = 110):
    color = get_confidence_color(value)
    circumference = 2 * 3.14159 * 42
    dashoffset = circumference * (1 - value)
    return f"""
    <div class="gauge-card">
        <svg width="{size}" height="{size}" viewBox="0 0 100 100">
            <circle cx="50" cy="50" r="42" fill="none" stroke="#e8e2dc" stroke-width="7"/>
            <circle cx="50" cy="50" r="42" fill="none" stroke="{color}" stroke-width="7"
                    stroke-dasharray="{circumference}" stroke-dashoffset="{dashoffset}"
                    transform="rotate(-90 50 50)" stroke-linecap="round"
                    style="transition: stroke-dashoffset 1s ease-in-out;"/>
        </svg>
        <div class="gauge-value" style="color: {color};">{value:.0%}</div>
        <div class="gauge-label">{label}</div>
    </div>
    """


def render_pipeline_status(current: str, completed: list):
    steps = [
        ("📋", "规划", "planning"),
        ("👁️", "视觉", "vision"),
        ("✍️", "文案", "copy"),
        ("✅", "校验", "verify"),
        ("📊", "聚合", "aggregate"),
    ]
    html = '<div class="pipeline-container">'
    for i, (icon, name, sid) in enumerate(steps):
        if sid in completed:
            cls = "completed"
        elif sid == current:
            cls = "active"
        else:
            cls = "pending"
        html += f'''
        <div class="pipeline-step">
            <div class="pipeline-node {cls}">{icon}</div>
            <div class="pipeline-node-label">{name}</div>
        </div>
        '''
        if i < len(steps) - 1:
            arrow_cls = "active" if sid in completed else ""
            html += f'<div class="pipeline-arrow {arrow_cls}"></div>'
    html += '</div>'
    return html


def parse_log_entry(log: str):
    if log.startswith("[") and "]" in log:
        parts = log.split("]", 1)
        time_str = parts[0].strip("[")
        message = parts[1].strip()
    else:
        time_str = ""
        message = log
    agent_type = "main"
    if "视觉" in message or "vision" in message.lower():
        agent_type = "vision"
    elif "文案" in message or "copy" in message.lower():
        agent_type = "copy"
    elif "校验" in message or "verify" in message.lower():
        agent_type = "verify"
    return time_str, message, agent_type


def render_log_timeline(logs: list):
    html = '<div class="log-timeline">'
    for log in logs:
        t, msg, atype = parse_log_entry(log)
        tag_cls = f"tag-{atype}"
        tag_name = {"main": "主Agent", "vision": "视觉", "copy": "文案", "verify": "校验"}[atype]
        html += f'''
        <div class="log-entry">
            <span class="log-time">{t}</span>
            <span class="tag {tag_cls}">{tag_name}</span>
            <div class="log-msg">{msg}</div>
        </div>
        '''
    html += '</div>'
    return html


def save_uploaded_file(uploaded_file):
    temp_dir = tempfile.mkdtemp()
    file_path = os.path.join(temp_dir, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path


def init_session_state():
    defaults = {
        "step": "input",           # input -> planned -> vision_done -> copy_done -> done
        "state": None,
        "image_path": None,
        "user_query": "",
        "processing": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def build_config(ablation_preset, final_threshold, max_retries, confidence_mode, w_vision=0.3, w_copy=0.3, w_verify=0.4):
    threshold_config = THRESHOLD_CONFIG.copy()
    threshold_config["final_threshold"] = final_threshold
    threshold_config["max_retries"] = max_retries

    ablation_config = ABLATION_PRESETS.get(ablation_preset, ABLATION_CONFIG.copy())
    ablation_config["confidence_mode"] = confidence_mode
    if confidence_mode == "weighted":
        ablation_config["weights"] = {"vision": w_vision, "copy": w_copy, "verify": w_verify}

    return threshold_config, ablation_config


# ============================================================
# 主界面
# ============================================================

def main():
    init_session_state()

    # 标题
    st.markdown("""
    <div style="text-align: center; padding: 0.5rem 0 1.5rem 0;">
        <h1 style="margin-bottom: 0.3rem;">多Agent智能解析系统</h1>
        <p style="color: var(--text-secondary); font-family: 'Noto Sans SC', sans-serif; font-size: 0.95rem; margin: 0;">
            基于LangGraph的电商商品多模态智能解析 · 视觉理解 · 文案生成 · 内容校验
        </p>
    </div>
    """, unsafe_allow_html=True)

    # 左右布局
    col_left, col_right = st.columns([2, 3], gap="large")

    # ============================================================
    # 左侧：输入区域
    # ============================================================
    with col_left:
        # 图片上传
        st.markdown("### 上传图像")
        uploaded_file = st.file_uploader(
            "拖拽或点击上传",
            type=["jpg", "jpeg", "png", "webp", "bmp"],
            help="支持 JPG、PNG、WebP、BMP 格式",
            label_visibility="collapsed"
        )

        if uploaded_file is not None:
            st.image(uploaded_file, caption=uploaded_file.name, use_container_width=True)
            file_size = len(uploaded_file.getvalue()) / 1024
            st.caption(f"文件大小: {file_size:.1f} KB")

        st.markdown("---")

        # 查询输入
        st.markdown("### 输入查询")
        user_query = st.text_area(
            "请输入您的问题或描述需求",
            placeholder="例如：\n• 请描述这张图片中的商品\n• 这个商品有什么特点？\n• 为这个商品写一段营销文案",
            height=100,
            label_visibility="collapsed",
            value=st.session_state.user_query
        )

        st.markdown("---")

        # 高级配置
        with st.expander("高级配置", expanded=False):
            ablation_preset = st.selectbox(
                "消融实验预设",
                options=list(ABLATION_PRESETS.keys()),
                format_func=lambda x: {"full": "完整流水线", "no_copy": "无文案Agent", "no_verify": "无校验Agent", "vision_only": "仅视觉Agent"}.get(x, x),
                index=0
            )

            col_t1, col_t2 = st.columns(2)
            with col_t1:
                final_threshold = st.slider("最终阈值", 0.0, 1.0, 0.7, 0.05, format="%.2f")
            with col_t2:
                max_retries = st.number_input("最大重试", 0, 5, 3, 1)

            confidence_mode = st.selectbox(
                "置信度模式",
                options=["weighted", "min", "verify_only", "model"],
                format_func=lambda x: {"weighted": "加权平均", "min": "最小值", "verify_only": "仅校验分数", "model": "模型判断"}.get(x, x),
                index=0
            )

            if confidence_mode == "weighted":
                w_col1, w_col2, w_col3 = st.columns(3)
                with w_col1:
                    w_vision = st.number_input("视觉权重", 0.0, 1.0, 0.3, 0.1)
                with w_col2:
                    w_copy = st.number_input("文案权重", 0.0, 1.0, 0.3, 0.1)
                with w_col3:
                    w_verify = st.number_input("校验权重", 0.0, 1.0, 0.4, 0.1)
            else:
                w_vision, w_copy, w_verify = 0.3, 0.3, 0.4

        # 操作按钮区
        st.markdown("<br>", unsafe_allow_html=True)

        step = st.session_state.step

        # 开始分析按钮
        if step == "input":
            if st.button("开始分析", use_container_width=True, type="primary"):
                if uploaded_file is None:
                    st.error("请先上传图片")
                elif not user_query.strip():
                    st.error("请输入查询内容")
                else:
                    # 保存文件
                    image_path = save_uploaded_file(uploaded_file)
                    st.session_state.image_path = image_path
                    st.session_state.user_query = user_query

                    # 构建配置
                    tc, ac = build_config(ablation_preset, final_threshold, max_retries, confidence_mode, w_vision, w_copy, w_verify)

                    # 创建初始状态
                    state = create_initial_state(image_path, user_query, tc, ac)

                    # 1. 运行规划
                    with st.spinner("主Agent规划中..."):
                        planning_result = planning_node(state)
                        state.update(planning_result)

                    # 读取图像base64
                    try:
                        b64 = image_to_base64(image_path)
                        state["image_base64"] = b64
                    except:
                        pass

                    # 2. 运行视觉Agent
                    with st.spinner("视觉Agent分析中..."):
                        vision_result = vision_agent(state)
                        state.update(vision_result)

                    # 3. 运行文案Agent
                    with st.spinner("文案Agent生成中..."):
                        copy_result = copy_agent(state)
                        state.update(copy_result)

                    # 4. 主Agent收集结果，传递给校验Agent
                    add_log(state, "主Agent收集视觉和文案结果，准备传递给校验Agent")
                    state["routing_decision"] = "verify"

                    # 5. 运行校验Agent
                    with st.spinner("校验Agent验证中..."):
                        verify_result = verify_agent(state)
                        state.update(verify_result)

                    # 6. 主Agent聚合结果
                    with st.spinner("主Agent聚合结果..."):
                        agg_result = aggregation_node(state)
                        state.update(agg_result)

                    state["logs"] = state.get("logs", [])
                    st.session_state.state = state
                    st.session_state.step = "done"
                    st.rerun()

        # 重新分析按钮
        elif step == "done":
            st.markdown("---")
            if st.button("重新分析", use_container_width=True):
                # 清理临时文件
                if st.session_state.image_path:
                    try:
                        os.remove(st.session_state.image_path)
                        os.rmdir(os.path.dirname(st.session_state.image_path))
                    except:
                        pass
                st.session_state.step = "input"
                st.session_state.state = None
                st.session_state.image_path = None
                st.rerun()

    # ============================================================
    # 右侧：结果展示
    # ============================================================
    with col_right:
        step = st.session_state.step
        state = st.session_state.state

        if step == "input":
            # 空状态
            st.markdown("""
            <div class="empty-state">
                <div class="empty-icon">🔬</div>
                <h3>等待分析</h3>
                <p>请上传图片并输入查询内容，然后点击"开始分析"</p>
            </div>
            """, unsafe_allow_html=True)

        elif state is not None:
            # 流水线状态
            st.markdown("### Agent 流水线")
            completed = ["planning"]
            if state.get("vision_result", {}).get("description"):
                completed.append("vision")
            if state.get("copy_result", {}).get("copywriting"):
                completed.append("copy")
            if state.get("verify_result", {}).get("score") is not None:
                completed.append("verify")
            if step == "done":
                completed.append("aggregate")

            current = {"vision_done": "vision", "copy_done": "copy", "done": "aggregate"}.get(step, "")
            st.markdown(render_pipeline_status(current, completed), unsafe_allow_html=True)

            # 置信度仪表盘
            st.markdown("### 置信度概览")
            gauge_cols = st.columns(4)
            with gauge_cols[0]:
                fc = state.get("final_confidence", 0) if step == "done" else 0
                st.markdown(render_gauge(fc, "综合置信度"), unsafe_allow_html=True)
            with gauge_cols[1]:
                vc = state.get("vision_result", {}).get("confidence", 0)
                st.markdown(render_gauge(vc, "视觉Agent"), unsafe_allow_html=True)
            with gauge_cols[2]:
                cc = state.get("copy_result", {}).get("confidence", 0) if step in ["copy_done", "done"] else 0
                st.markdown(render_gauge(cc, "文案Agent"), unsafe_allow_html=True)
            with gauge_cols[3]:
                vs = state.get("verify_result", {}).get("score", 0) if step == "done" else 0
                st.markdown(render_gauge(vs, "校验Agent"), unsafe_allow_html=True)

            # 视觉结果
            if state.get("vision_result", {}).get("description"):
                st.markdown("### 视觉分析结果")
                vr = state["vision_result"]
                st.markdown(f"""
                <div class="agent-detail-card">
                    <div class="agent-detail-header">
                        <div class="agent-icon vision">👁️</div>
                        <div class="agent-detail-title">视觉理解Agent</div>
                        <span class="agent-detail-badge badge-success">置信度 {vr.get("confidence", 0):.0%}</span>
                    </div>
                    <div style="color: var(--text-secondary); line-height: 1.8; font-size: 0.9rem;">
                        {vr.get("description", "").replace(chr(10), "<br>")}
                    </div>
                </div>
                """, unsafe_allow_html=True)

            # 文案结果
            if step in ["copy_done", "done"] and state.get("copy_result", {}).get("copywriting"):
                st.markdown("### 文案生成结果")
                cr = state["copy_result"]
                cw = cr.get("copywriting", "")
                if cw != "[用户跳过]":
                    st.markdown(f"""
                    <div class="agent-detail-card">
                        <div class="agent-detail-header">
                            <div class="agent-icon copy">✍️</div>
                            <div class="agent-detail-title">文案生成Agent</div>
                            <span class="agent-detail-badge badge-info">置信度 {cr.get("confidence", 0):.0%}</span>
                        </div>
                        <div style="color: var(--text-secondary); line-height: 1.8; font-size: 0.9rem;">
                            {cw.replace(chr(10), "<br>")}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.info("用户已跳过文案生成")

            # 校验结果
            if step == "done" and state.get("verify_result", {}).get("score") is not None:
                st.markdown("### 校验结论")
                vr = state["verify_result"]
                comment = vr.get("comment", "")
                # 过滤掉可能的错误信息
                if "原API错误" in comment:
                    comment = comment.split("(原API错误")[0].strip()
                score = vr.get("score", 0)
                passed = vr.get("passed", False)
                badge_cls = "badge-success" if passed else "badge-warning"
                badge_text = "通过" if passed else "未通过"

                st.markdown(f"""
                <div class="agent-detail-card">
                    <div class="agent-detail-header">
                        <div class="agent-icon verify">✅</div>
                        <div class="agent-detail-title">内容校验Agent</div>
                        <span class="agent-detail-badge {badge_cls}">{badge_text}</span>
                    </div>
                    <div style="display: flex; gap: 2rem; margin-bottom: 0.75rem;">
                        <div><span style="color: var(--text-muted);">校验分数:</span> <strong style="color: {get_confidence_color(score)};">{score:.2%}</strong></div>
                        <div><span style="color: var(--text-muted);">状态:</span> <strong>{"通过" if passed else "未通过"}</strong></div>
                    </div>
                    <div style="color: var(--text-secondary); font-size: 0.9rem;">{comment}</div>
                </div>
                """, unsafe_allow_html=True)

            # 最终输出
            if step == "done" and state.get("final_output"):
                st.markdown("### 最终输出")
                st.markdown(f"""
                <div class="final-output-card">
                    {state["final_output"].replace(chr(10), "<br>")}
                </div>
                """, unsafe_allow_html=True)

                # 重试信息
                retry_counts = state.get("retry_counts", {})
                if any(v > 0 for v in retry_counts.values()):
                    st.markdown("### 重试统计")
                    for agent, count in retry_counts.items():
                        if count > 0:
                            name = {"vision": "视觉", "copy": "文案", "verify": "校验"}.get(agent, agent)
                            st.markdown(f"- **{name}Agent:** 重试 {count} 次")

            # 执行日志
            logs = state.get("logs", [])
            if logs:
                with st.expander("执行日志", expanded=False):
                    st.markdown(render_log_timeline(logs), unsafe_allow_html=True)


if __name__ == "__main__":
    main()
