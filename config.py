"""
配置模块
包含API配置、阈值配置、消融配置等
"""
import os
from typing import Dict, Any

# ============================================================
# API配置
# ============================================================

# DeepSeek (主Agent)
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_MODEL = "deepseek-v4-pro"

# Qwen (视觉Agent、文案Agent)
QWEN_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
QWEN_VL_MODEL = "qwen3-vl-plus"
QWEN_TEXT_MODEL = "qwen3.6-plus"

# MiMo (校验Agent)
MIMO_BASE_URL = "https://api.xiaomimimo.com/v1"
MIMO_MODEL = "mimo-v2.5-pro"

# ============================================================
# API密钥 (从环境变量读取)
# ============================================================

def load_api_keys() -> Dict[str, str]:
    """从apikey文件加载API密钥"""
    keys = {}
    apikey_path = os.path.join(os.path.dirname(__file__), "apikey")

    if os.path.exists(apikey_path):
        with open(apikey_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
            for i in range(0, len(lines), 2):
                if i + 1 < len(lines):
                    provider = lines[i].strip().lower()
                    key = lines[i + 1].strip()
                    if "deepseek" in provider:
                        keys["DEEPSEEK_API_KEY"] = key
                    elif "qwen" in provider:
                        keys["QWEN_API_KEY"] = key
                    elif "mimo" in provider:
                        keys["MIMO_API_KEY"] = key
                    elif "langsmith" in provider:
                        keys["LANGCHAIN_API_KEY"] = key
    return keys

# 加载密钥
_api_keys = load_api_keys()

DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", _api_keys.get("DEEPSEEK_API_KEY", ""))
QWEN_API_KEY = os.environ.get("QWEN_API_KEY", _api_keys.get("QWEN_API_KEY", ""))
MIMO_API_KEY = os.environ.get("MIMO_API_KEY", _api_keys.get("MIMO_API_KEY", ""))

# ============================================================
# LangSmith配置
# ============================================================

LANGCHAIN_API_KEY = os.environ.get("LANGCHAIN_API_KEY", _api_keys.get("LANGCHAIN_API_KEY", ""))
LANGCHAIN_PROJECT = "multi-agent-ecommerce"
LANGCHAIN_TRACING_V2 = "true"

# 设置环境变量
os.environ["LANGCHAIN_API_KEY"] = LANGCHAIN_API_KEY
os.environ["LANGCHAIN_PROJECT"] = LANGCHAIN_PROJECT
os.environ["LANGCHAIN_TRACING_V2"] = LANGCHAIN_TRACING_V2

# ============================================================
# 阈值配置
# ============================================================

THRESHOLD_CONFIG = {
    "final_threshold": 0.8,           # 最终置信度阈值
    "max_retries": 3,                 # 最大重试次数
    "vision_threshold": 0.6,          # 视觉Agent阈值
    "copy_threshold": 0.6,            # 文案Agent阈值
    "verify_threshold": 0.6,          # 校验Agent阈值
}

# ============================================================
# 消融实验配置
# ============================================================

ABLATION_CONFIG = {
    # Agent开关
    "enable_vision": True,            # 是否启用视觉Agent
    "enable_copy": True,              # 是否启用文案Agent
    "enable_verify": True,            # 是否启用校验Agent

    # 置信度计算模式: model / min / weighted / verify_only
    "confidence_mode": "weighted",

    # 加权平均的权重 (仅在confidence_mode="weighted"时生效)
    "weights": {
        "vision": 0.3,
        "copy": 0.3,
        "verify": 0.4,
    },
}

# ============================================================
# 预定义消融配置
# ============================================================

ABLATION_PRESETS = {
    "full": {
        "enable_vision": True,
        "enable_copy": True,
        "enable_verify": True,
        "confidence_mode": "weighted",
        "weights": {"vision": 0.3, "copy": 0.3, "verify": 0.4},
    },
    "no_copy": {
        "enable_vision": True,
        "enable_copy": False,
        "enable_verify": True,
        "confidence_mode": "weighted",
        "weights": {"vision": 0.4, "verify": 0.6},
    },
    "no_verify": {
        "enable_vision": True,
        "enable_copy": True,
        "enable_verify": False,
        "confidence_mode": "min",
        "weights": {"vision": 0.5, "copy": 0.5},
    },
    "vision_only": {
        "enable_vision": True,
        "enable_copy": False,
        "enable_verify": False,
        "confidence_mode": "model",
        "weights": {"vision": 1.0},
    },
}

# ============================================================
# 默认执行计划
# ============================================================

DEFAULT_PLAN = {
    "task_type": "describe",
    "steps": ["vision", "copy", "verify"],
    "reasoning": "默认执行计划"
}
