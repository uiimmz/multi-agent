"""
评测基准系统
对比三种配置的性能表现：
1. 仅视觉Agent - Qwen3-VL-Plus 独立完成图像理解
2. 视觉Agent + 主Agent - Qwen3-VL-Plus 视觉 → DeepSeek-V4-Pro 文案
3. 完整多Agent系统 - DeepSeek调度 + Qwen视觉 + Qwen文案 + MiMo校验

评测指标：
1. 综合准确率（属性匹配，基于ground truth规则评估）
2. 幻觉率（属性矛盾率，基于ground truth规则评估）
3. 平均响应时间
4. Token消耗量
"""
import sys
import os
import json
import time
import re
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from config import (
    DEEPSEEK_BASE_URL, DEEPSEEK_API_KEY, DEEPSEEK_MODEL,
    QWEN_BASE_URL, QWEN_API_KEY, QWEN_VL_MODEL,
    THRESHOLD_CONFIG, ABLATION_CONFIG
)
from utils import image_to_base64, get_image_media_type, add_log


# ============================================================
# 配置定义
# ============================================================

CONFIGS = {
    "vision_only": {
        "name": "仅视觉Agent",
        "description": "Qwen3-VL-Plus 独立完成图像理解",
        "models": ["Qwen3-VL-Plus"],
        "enable_vision": True,
        "enable_copy": False,
        "enable_verify": False,
    },
    "vision_main": {
        "name": "视觉Agent + 主Agent",
        "description": "Qwen3-VL-Plus 视觉理解 → DeepSeek-V4-Pro 文案生成",
        "models": ["Qwen3-VL-Plus", "DeepSeek-V4-Pro"],
        "enable_vision": True,
        "enable_copy": False,
        "enable_verify": False,
    },
    "full_system": {
        "name": "完整多Agent系统",
        "description": "DeepSeek-V4-Pro 调度 + Qwen3-VL-Plus 视觉 + Qwen3.6-Plus 文案 + MiMo-V2.5-Pro 校验",
        "models": ["DeepSeek-V4-Pro", "Qwen3-VL-Plus", "Qwen3.6-Plus", "MiMo-V2.5-Pro"],
        "enable_vision": True,
        "enable_copy": True,
        "enable_verify": True,
    },
}


# ============================================================
# 中英文属性映射表（用于规则评估）
# ============================================================

# 颜色映射：英文颜色 → 中文关键词列表
COLOR_MAP = {
    "White":      ["白色", "白", "米白", "纯白", "奶白", "象牙白", "雪白", "本白"],
    "Black":      ["黑色", "黑", "纯黑", "墨黑", "深黑"],
    "Pink":       ["粉色", "粉", "粉红", "桃红", "樱花粉", "淡粉", "浅粉", "玫红", "粉嫩"],
    "Blue":       ["蓝色", "蓝", "天蓝", "浅蓝", "淡蓝", "宝蓝", "湖蓝", "蔚蓝", "水蓝"],
    "Red":        ["红色", "红", "大红", "鲜红", "暗红", "酒红", "正红", "朱红"],
    "Green":      ["绿色", "绿", "草绿", "翠绿", "墨绿", "青绿", "薄荷绿", "嫩绿", "浅绿", "橄榄绿"],
    "Yellow":     ["黄色", "黄", "亮黄", "柠檬黄", "明黄", "淡黄", "鹅黄", "姜黄"],
    "Navy Blue":  ["深蓝", "海军蓝", "藏青", "藏蓝", "深海军蓝", "navy"],
    "Grey":       ["灰色", "灰", "深灰", "浅灰", "烟灰", "炭灰", "银灰", "灰白"],
    "Olive":      ["橄榄色", "橄榄绿", "橄榄", "军绿", "olive"],
    "Magenta":    ["品红", "洋红", "紫红", "magenta", "玫红"],
    "Orange":     ["橙色", "橙", "橘色", "橘", "橘红", "橙红"],
    "Brown":      ["棕色", "棕", "褐色", "褐", "咖啡色", "卡其", "驼色", "巧克力色"],
    "Purple":     ["紫色", "紫", "淡紫", "深紫", "葡萄紫", "薰衣草紫"],
    "Beige":      ["米色", "米黄", "杏色", "裸色", "浅卡其", "beige"],
    "Maroon":     ["栗色", "褐红", "maroon"],
    "Teal":       ["青色", "青", "蓝绿", "teal"],
    "Mustard":    ["芥末黄", "芥黄", "mustard"],
    "Coral":      ["珊瑚色", "珊瑚红", "珊瑚", "coral"],
    "Peach":      ["桃色", "蜜桃色", "peach"],
    "Mauve":      ["淡紫", "紫灰", "mauve"],
    "Tan":        ["棕黄", "tan"],
    "Burgundy":   ["勃艮第红", "酒红", "深红"],
    "Lavender":   ["薰衣草紫", "淡紫"],
    "Gold":       ["金色", "金", "浅金", "米金"],
    "Silver":     ["银色", "银", "银灰"],
    "Multi":      ["多色", "彩色", "拼色", "撞色"],
}

# 品类映射：英文product_type → 中文关键词列表
PRODUCT_TYPE_MAP = {
    "Tops":       ["上衣", "上装", "衫", "top", "T恤衫"],
    "Tshirts":    ["T恤", "T恤衫", "短袖", "t-shirt", "tee", "汗衫"],
    "Dresses":    ["连衣裙", "裙装", "dress", "连身裙", "洋装"],
    "Shorts":     ["短裤", "热裤", "shorts", "休闲短裤"],
    "Capris":     ["七分裤", "紧身裤", "打底裤", "九分裤", "leggings", "中裤", "五分裤", "capri", "修身裤"],
    "Skirts":     ["裙子", "半身裙", "短裙", "长裙", "skirt", "裙"],
    "Shirts":     ["衬衫", "衬衣", "shirt"],
    "Jeans":      ["牛仔裤", "牛仔", "jeans"],
    "Jackets":    ["夹克", "外套", "jacket"],
    "Sweaters":   ["毛衣", "针织衫", "sweater"],
    "Sweatshirts":["卫衣", "运动衫", "sweatshirt"],
    "Trousers":   ["裤子", "长裤", "西裤", "trousers", "pants"],
    "Leggings":   ["打底裤", "紧身裤", "leggings", "修身裤", "弹力裤"],
    "Booties":    ["靴子", "短靴", "boots", "booties", "踝靴"],
    "Sandals":    ["凉鞋", "拖鞋", "sandals", "slipper"],
    "Socks":      ["袜子", "短袜", "socks"],
    "Innerwear":  ["内衣", "背心", "吊带", "innerwear", "vest"],
}

# 风格映射：英文usage → 中文关键词列表
USAGE_MAP = {
    "Casual":     ["休闲", "日常", "便装", "casual"],
    "Formal":     ["正式", "正装", "商务", "formal"],
    "Sports":     ["运动", "健身", "sports", "运动风"],
    "Party":      ["派对", "聚会", "party", "晚宴"],
    "Ethnic":     ["民族", "传统", "ethnic", "民俗"],
    "Beach":      ["沙滩", "海滩", "beach", "度假"],
    "Sleep":      ["睡衣", "家居", "sleep"],
}

# 品类兼容分组：同一组内的品类不判为矛盾（如T恤是上衣的一种）
PRODUCT_COMPATIBILITY_GROUPS = [
    {"Tops", "Tshirts", "Shirts", "Sweaters", "Sweatshirts", "Jackets", "Innerwear"},  # 上装
    {"Shorts", "Capris", "Jeans", "Trousers", "Leggings"},                             # 下装
    {"Dresses"},                                                                        # 连衣裙
    {"Skirts"},                                                                         # 半身裙
    {"Booties", "Sandals", "Socks"},                                                    # 鞋袜
]

# 颜色兼容分组：同一组内的颜色不判为矛盾（如深蓝和蓝）
COLOR_COMPATIBILITY_GROUPS = [
    {"White", "Beige"},
    {"Black", "Grey"},
    {"Red", "Pink", "Magenta", "Maroon", "Burgundy", "Coral", "Peach", "Mauve"},
    {"Blue", "Navy Blue", "Teal"},
    {"Green", "Olive", "Teal"},
    {"Yellow", "Mustard", "Gold", "Orange"},
    {"Brown", "Tan", "Beige", "Olive"},
    {"Purple", "Mauve", "Lavender", "Magenta"},
    {"Gold", "Yellow"},
    {"Silver", "Grey"},
]


# ============================================================
# Token计数器
# ============================================================

class TokenCounter:
    """API调用Token计数器"""

    def __init__(self):
        self.total_tokens = 0
        self.call_log = []

    def add(self, model: str, prompt_tokens: int, completion_tokens: int):
        total = prompt_tokens + completion_tokens
        self.total_tokens += total
        self.call_log.append({
            "model": model,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total": total
        })

    def get_total(self) -> int:
        return self.total_tokens

    def reset(self):
        self.total_tokens = 0
        self.call_log = []


# ============================================================
# 配置1: 仅视觉Agent — Qwen3-VL-Plus 独立完成图像理解
# ============================================================

def run_vision_only(
    image_path: str,
    user_query: str,
    token_counter: TokenCounter
) -> Dict[str, Any]:
    """
    仅视觉Agent模式：Qwen3-VL-Plus 分析图像，输出视觉描述和置信度。
    不涉及主Agent、文案Agent、校验Agent。

    Args:
        image_path: 图像文件路径
        user_query: 用户查询
        token_counter: Token计数器

    Returns:
        结果字典
    """
    from openai import OpenAI

    image_base64 = image_to_base64(image_path)
    media_type = get_image_media_type(image_path)

    prompt = f"""请仔细观察这张图像，并结合用户的查询进行分析。

用户查询: {user_query}

请完成以下任务：
1. 详细描述图像中的主要内容，包括商品类型、颜色、款式、材质、图案等关键信息
2. 在回答末尾，以"置信度: X.XX"的格式给出你对描述准确性的置信度（0-1之间的数字）

请用中文回答。"""

    try:
        client = OpenAI(api_key=QWEN_API_KEY, base_url=QWEN_BASE_URL)

        response = client.chat.completions.create(
            model=QWEN_VL_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:{media_type};base64,{image_base64}"}},
                        {"type": "text", "text": prompt}
                    ]
                }
            ],
            max_tokens=2000,
            temperature=0.7
        )

        raw_response = response.choices[0].message.content

        usage = response.usage
        if usage:
            token_counter.add(QWEN_VL_MODEL, usage.prompt_tokens, usage.completion_tokens)

        confidence = _extract_confidence_from_response(raw_response)

        return {
            "success": True,
            "output": raw_response,
            "confidence": confidence,
            "raw_response": raw_response,
            "model": QWEN_VL_MODEL,
        }

    except Exception as e:
        return {
            "success": False,
            "output": f"执行失败: {str(e)}",
            "confidence": 0,
            "error": str(e)
        }


# ============================================================
# 配置2: 视觉Agent + 主Agent — Qwen3-VL-Plus → DeepSeek-V4-Pro
# ============================================================

def run_vision_main(
    image_path: str,
    user_query: str,
    token_counter: TokenCounter
) -> Dict[str, Any]:
    """
    视觉Agent + 主Agent模式：
    Step 1: Qwen3-VL-Plus (视觉Agent) 分析图像 → 输出视觉描述
    Step 2: DeepSeek-V4-Pro (主Agent) 基于描述生成文案并评估

    Args:
        image_path: 图像文件路径
        user_query: 用户查询
        token_counter: Token计数器

    Returns:
        结果字典
    """
    from openai import OpenAI

    image_base64 = image_to_base64(image_path)
    media_type = get_image_media_type(image_path)

    # ---- Step 1: 视觉Agent (Qwen3-VL-Plus) ----
    vision_prompt = f"""请仔细观察这张图像，并结合用户的查询进行分析。

用户查询: {user_query}

请完成以下任务：
1. 详细描述图像中的主要内容，包括商品类型、颜色、款式、材质、图案等关键信息
2. 在回答末尾，以"置信度: X.XX"的格式给出你对描述准确性的置信度（0-1之间的数字）

请用中文回答。"""

    vision_description = ""
    vision_confidence = 0.0

    try:
        client_vl = OpenAI(api_key=QWEN_API_KEY, base_url=QWEN_BASE_URL)

        vision_response = client_vl.chat.completions.create(
            model=QWEN_VL_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:{media_type};base64,{image_base64}"}},
                        {"type": "text", "text": vision_prompt}
                    ]
                }
            ],
            max_tokens=2000,
            temperature=0.7
        )

        vision_text = vision_response.choices[0].message.content
        vision_description = vision_text
        vision_confidence = _extract_confidence_from_response(vision_text)

        usage = vision_response.usage
        if usage:
            token_counter.add(QWEN_VL_MODEL, usage.prompt_tokens, usage.completion_tokens)

    except Exception as e:
        return {
            "success": False,
            "output": f"视觉Agent执行失败: {str(e)}",
            "confidence": 0,
            "error": str(e)
        }

    # ---- Step 2: 主Agent (DeepSeek-V4-Pro) 文案生成 + 评估 ----
    main_prompt = f"""你是一个电商文案专家和质量评估专家。基于以下视觉描述和用户查询，完成文案生成和内容评估。

视觉描述：
{vision_description}

用户查询: {user_query}

请完成：
1. 基于视觉描述生成一段吸引人的营销文案（突出商品特点和优势，语言生动适合电商场景）
2. 对内容质量进行评估

请按以下格式输出：

【创意文案】
（营销文案）

【质量评估】
- 置信度: X.XX
- 评语: （简要说明）"""

    try:
        client_ds = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)

        main_response = client_ds.chat.completions.create(
            model=DEEPSEEK_MODEL,
            messages=[
                {"role": "system", "content": "你是电商文案专家和质量评估专家。"},
                {"role": "user", "content": main_prompt}
            ],
            max_tokens=2000,
            temperature=0.8
        )

        main_text = main_response.choices[0].message.content
        main_confidence = _extract_confidence_from_response(main_text)

        usage = main_response.usage
        if usage:
            token_counter.add(DEEPSEEK_MODEL, usage.prompt_tokens, usage.completion_tokens)

        final_confidence = (vision_confidence + main_confidence) / 2

        output = f"""【图像描述 - Qwen3-VL-Plus】

{vision_description}

【创意文案 - DeepSeek-V4-Pro】

{main_text}

【综合置信度】: {final_confidence:.2%}"""

        return {
            "success": True,
            "output": output,
            "confidence": final_confidence,
            "vision_confidence": vision_confidence,
            "main_confidence": main_confidence,
        }

    except Exception as e:
        return {
            "success": False,
            "output": f"主Agent执行失败: {str(e)}",
            "confidence": vision_confidence,
            "error": str(e)
        }


# ============================================================
# 配置3: 完整多Agent系统
#   DeepSeek-V4-Pro (主Agent/调度) → Qwen3-VL-Plus (视觉)
#   → Qwen3.6-Plus (文案) → MiMo-V2.5-Pro (校验)
# ============================================================

def run_full_system(
    image_path: str,
    user_query: str,
    token_counter: TokenCounter
) -> Dict[str, Any]:
    """
    完整多Agent协作系统：
    - DeepSeek-V4-Pro: 任务规划、路由决策、结果聚合
    - Qwen3-VL-Plus: 视觉理解Agent
    - Qwen3.6-Plus: 文案生成Agent
    - MiMo-V2.5-Pro: 内容校验Agent

    Args:
        image_path: 图像文件路径
        user_query: 用户查询
        token_counter: Token计数器

    Returns:
        结果字典
    """
    from graph import run_pipeline
    from config import THRESHOLD_CONFIG, ABLATION_CONFIG

    try:
        ablation_config = {
            "enable_vision": True,
            "enable_copy": True,
            "enable_verify": True,
            "confidence_mode": "weighted",
            "weights": {"vision": 0.3, "copy": 0.3, "verify": 0.4},
        }

        final_state = run_pipeline(
            image_path=image_path,
            user_query=user_query,
            threshold_config=THRESHOLD_CONFIG.copy(),
            ablation_config=ablation_config
        )

        # 估算Token消耗（从日志中）
        token_estimate = 0
        for log in final_state.get("logs", []):
            token_estimate += len(log) // 2  # 中文字符约2字节

        # 添加到计数器
        token_counter.total_tokens += token_estimate

        return {
            "success": bool(final_state.get("final_output")),
            "output": final_state.get("final_output", ""),
            "confidence": final_state.get("final_confidence", 0),
            "vision_confidence": final_state.get("vision_result", {}).get("confidence", 0),
            "copy_confidence": final_state.get("copy_result", {}).get("confidence", 0),
            "verify_score": final_state.get("verify_result", {}).get("score", 0),
            "verify_passed": final_state.get("verify_result", {}).get("passed", False),
            "token_estimate": token_estimate,
        }

    except Exception as e:
        return {
            "success": False,
            "output": f"执行失败: {str(e)}",
            "confidence": 0,
            "error": str(e)
        }


# ============================================================
# 辅助函数
# ============================================================

def _extract_confidence_from_response(text: str) -> float:
    """从响应文本中提取置信度"""
    patterns = [
        r'置信度[：:]\s*([\d.]+)',
        r'confidence[：:]\s*([\d.]+)',
        r'Confidence[：:]\s*([\d.]+)',
        r'置信度\s*[为是]\s*([\d.]+)',
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            try:
                value = float(match.group(1))
                if 0 <= value <= 1:
                    return value
            except ValueError:
                continue

    return 0.7  # 默认值


# ============================================================
# 规则评估：基于ground truth属性匹配
# ============================================================

def evaluate_against_ground_truth(
    model_output: str,
    ground_truth: Dict[str, str]
) -> Dict[str, Any]:
    """
    基于规则匹配，对比模型输出与ground truth属性。

    评估维度：
    1. 属性准确率 (attribute_accuracy): 模型是否正确识别了品类/颜色/风格
    2. 幻觉检测 (hallucination): 模型是否输出了与ground truth矛盾的属性
    3. 关键属性召回 (key_attr_recall): ground truth中的关键属性是否被提及

    Args:
        model_output: 模型输出文本
        ground_truth: 包含 product_type, color, usage 的字典

    Returns:
        评估结果字典
    """
    gt_product = ground_truth.get("product_type", "").strip()
    gt_color = ground_truth.get("color", "").strip()
    gt_usage = ground_truth.get("usage", "").strip()

    # --- 辅助函数：判断两个属性值是否在同一个兼容组内 ---
    def _is_compatible(attr1: str, attr2: str, groups: List[set]) -> bool:
        """两个属性值在同一个兼容组内返回True"""
        if attr1.lower() == attr2.lower():
            return True
        for group in groups:
            if attr1 in group and attr2 in group:
                return True
        return False

    # --- 查找模型输出中提及的颜色 ---
    mentioned_colors = []
    contradictory_color = False
    for en_color, zh_keywords in COLOR_MAP.items():
        for kw in zh_keywords:
            if kw in model_output:
                mentioned_colors.append((en_color, kw))
                break

    # 检测颜色矛盾：模型提到了不兼容的颜色，且没有提到gt颜色（或兼容色）
    if gt_color:
        gt_color_norm = gt_color.strip()
        # 检查是否提到了gt颜色（或兼容颜色）
        mentioned_gt_color = any(
            _is_compatible(en, gt_color_norm, COLOR_COMPATIBILITY_GROUPS)
            for en, _ in mentioned_colors
        ) if mentioned_colors else False
        # 只有模型没提到gt颜色，且提到了不兼容颜色时，才判定为颜色矛盾
        contradictory_color = False
        if not mentioned_gt_color:
            for en_color, kw in mentioned_colors:
                if not _is_compatible(en_color, gt_color_norm, COLOR_COMPATIBILITY_GROUPS):
                    contradictory_color = True
                    break
    else:
        mentioned_gt_color = None

    # --- 查找模型输出中提及的品类 ---
    mentioned_products = []
    contradictory_product = False
    for en_product, zh_keywords in PRODUCT_TYPE_MAP.items():
        for kw in zh_keywords:
            if kw in model_output:
                mentioned_products.append((en_product, kw))
                break

    if gt_product:
        gt_product_norm = gt_product.strip()
        # 检查是否提到了gt品类（或兼容品类）
        mentioned_gt_product = any(
            _is_compatible(en, gt_product_norm, PRODUCT_COMPATIBILITY_GROUPS)
            for en, _ in mentioned_products
        ) if mentioned_products else False
        # 只有模型没提到gt品类，且提到了不兼容品类时，才判定为品类矛盾
        contradictory_product = False
        if not mentioned_gt_product:
            for en_product, kw in mentioned_products:
                if not _is_compatible(en_product, gt_product_norm, PRODUCT_COMPATIBILITY_GROUPS):
                    contradictory_product = True
                    break
    else:
        mentioned_gt_product = None

    # --- 查找模型输出中提及的风格 ---
    mentioned_usages = []
    for en_usage, zh_keywords in USAGE_MAP.items():
        for kw in zh_keywords:
            if kw in model_output:
                mentioned_usages.append((en_usage, kw))
                break

    if gt_usage:
        gt_usage_normalized = gt_usage.strip()
        mentioned_gt_usage = any(
            en.lower() == gt_usage_normalized.lower()
            for en, _ in mentioned_usages
        ) if mentioned_usages else False
    else:
        mentioned_gt_usage = None

    # --- 计算各维度得分 ---
    # 属性准确率：正确提到的属性数 / 有ground truth的属性数
    attr_checks = []
    if mentioned_gt_color is not None:
        attr_checks.append(1.0 if mentioned_gt_color else 0.0)
    if mentioned_gt_product is not None:
        attr_checks.append(1.0 if mentioned_gt_product else 0.0)
    if mentioned_gt_usage is not None:
        attr_checks.append(1.0 if mentioned_gt_usage else 0.0)

    attribute_accuracy = sum(attr_checks) / len(attr_checks) if attr_checks else 0.5

    # 幻觉检测：是否存在矛盾属性
    # 如果颜色或品类出现矛盾，则判定为幻觉
    hallucination_detected = contradictory_color or contradictory_product

    # --- 返回结果 ---
    return {
        "attribute_accuracy": round(attribute_accuracy, 3),
        "hallucination_detected": hallucination_detected,
        "contradictory_color": contradictory_color,
        "contradictory_product": contradictory_product,
        "mentioned_gt_color": mentioned_gt_color,
        "mentioned_gt_product": mentioned_gt_product,
        "mentioned_gt_usage": mentioned_gt_usage,
        "mentioned_colors": [c for c, _ in mentioned_colors],
        "mentioned_products": [p for p, _ in mentioned_products],
        "mentioned_usages": [u for u, _ in mentioned_usages],
    }


def load_test_dataset(n: int = 10) -> List[Dict[str, Any]]:
    """
    加载测试数据集

    Args:
        n: 测试用例数量

    Returns:
        测试用例列表
    """
    dataset_path = os.path.join(os.path.dirname(__file__), 'test_dataset', 'dataset.jsonl')

    with open(dataset_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # 均匀采样
    step = max(1, len(lines) // n)
    test_cases = []

    for i in range(0, len(lines), step):
        item = json.loads(lines[i])
        img_path = os.path.join(os.path.dirname(__file__), 'test_dataset', item['image_path'])

        if os.path.exists(img_path):
            test_cases.append({
                "title": item.get("title", f"Item {i}"),
                "image_path": img_path,
                "query": item.get("query", "请描述这件商品的特点"),
                "description": item.get("description", ""),
                "category": item.get("category", "unknown"),
                "product_type": item.get("product_type", ""),
                "color": item.get("color", ""),
                "usage": item.get("usage", ""),
            })

        if len(test_cases) >= n:
            break

    return test_cases


# ============================================================
# 断点续跑 / 中断保护
# ============================================================

CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), "output", "checkpoints")


def _get_checkpoint_path() -> str:
    """获取checkpoint文件路径"""
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    return os.path.join(CHECKPOINT_DIR, "benchmark_checkpoint.json")


def save_checkpoint(all_results: Dict[str, Any]) -> None:
    """保存当前进度到checkpoint文件"""
    checkpoint = {}
    for config_key, data in all_results.items():
        checkpoint[config_key] = {
            "config_name": data["config"]["name"],
            "results": data["results"],
            "token_total": data["token_counter"].get_total(),
        }
    with open(_get_checkpoint_path(), "w", encoding="utf-8") as f:
        json.dump(checkpoint, f, indent=2, ensure_ascii=False, default=str)
    print("  [checkpoint saved]")


def load_checkpoint() -> Dict[str, Any]:
    """加载上次中断的checkpoint，返回 {config_key: set(completed_case_indices)}"""
    path = _get_checkpoint_path()
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    completed = {}
    for config_key, cfg_data in data.items():
        completed[config_key] = {r["case_id"] for r in cfg_data.get("results", [])}
    return completed


# ============================================================
# 评测主函数
# ============================================================

def run_benchmark(
    n: int = 10,
    configs_to_run: List[str] = None,
    resume: bool = False
) -> Dict[str, Any]:
    """
    运行评测基准。支持中断后自动生成图表，支持 --resume 断点续跑。

    Args:
        n: 测试用例数量
        configs_to_run: 要运行的配置列表，默认全部运行
        resume: 是否从checkpoint恢复

    Returns:
        评测结果字典
    """
    if configs_to_run is None:
        configs_to_run = list(CONFIGS.keys())

    # 加载测试数据
    test_cases = load_test_dataset(n)
    output_dir = os.path.join(os.path.dirname(__file__), "output")

    print(f"\n{'='*70}")
    print(f"多Agent系统评测基准")
    print(f"{'='*70}")
    print(f"测试用例数: {len(test_cases)}")
    print(f"评测配置: {', '.join(CONFIGS[c]['name'] for c in configs_to_run)}")
    if resume:
        print(f"模式: 断点续跑 (已完成用例将跳过)")
    print(f"{'='*70}\n")

    # 恢复或初始化 all_results
    all_results = {}
    # 断点续跑时加载完整checkpoint数据（不只是case ID集合）
    checkpoint_data = {}
    completed_map = {}
    if resume:
        ckpt_path = _get_checkpoint_path()
        if os.path.exists(ckpt_path):
            with open(ckpt_path, "r", encoding="utf-8") as _f:
                checkpoint_data = json.load(_f)
            for ck, cd in checkpoint_data.items():
                completed_map[ck] = {r["case_id"] for r in cd.get("results", [])}
        if not checkpoint_data:
            print("未找到checkpoint，将从头开始。")
            resume = False

    for config_key in configs_to_run:
        config = CONFIGS[config_key]
        skipped = completed_map.get(config_key, set())
        prev_results = checkpoint_data.get(config_key, {}).get("results", [])
        prev_tokens = checkpoint_data.get(config_key, {}).get("token_total", 0)

        print(f"\n{'='*70}")
        print(f"运行配置: {config['name']} - {config['description']}")
        if skipped:
            print(f"已跳过 {len(skipped)} 个已完成用例 (上次Token: {prev_tokens})")
        print(f"{'='*70}")

        token_counter = TokenCounter()
        token_counter.total_tokens = prev_tokens  # 恢复上次的token计数
        results = list(prev_results)  # 从checkpoint恢复之前的结果

        for i, case in enumerate(test_cases):
            case_id = i + 1

            # 断点续跑：跳过已完成的用例
            if resume and case_id in skipped:
                print(f"\n[{case_id}/{len(test_cases)}] {case['title']} [SKIPPED]")
                # 仍需记录结果以维持索引（从checkpoint恢复或标记为跳过）
                continue

            print(f"\n[{case_id}/{len(test_cases)}] {case['title']}")

            start_time = time.time()

            try:
                if config_key == "vision_only":
                    result = run_vision_only(case['image_path'], case['query'], token_counter)
                elif config_key == "vision_main":
                    result = run_vision_main(case['image_path'], case['query'], token_counter)
                elif config_key == "full_system":
                    result = run_full_system(case['image_path'], case['query'], token_counter)
                else:
                    result = {"success": False, "error": f"未知配置: {config_key}"}
            except Exception as e:
                elapsed_time = time.time() - start_time
                result = {"success": False, "output": f"异常: {str(e)}", "confidence": 0, "error": str(e)}

            elapsed_time = time.time() - start_time

            # 规则评估
            ground_truth = {
                "product_type": case.get("product_type", ""),
                "color": case.get("color", ""),
                "usage": case.get("usage", ""),
            }
            evaluation = evaluate_against_ground_truth(
                result.get("output", ""),
                ground_truth
            )

            result_record = {
                "case_id": case_id,
                "title": case['title'],
                "image_path": case['image_path'],
                "category": case['category'],
                "elapsed_time": elapsed_time,
                "token_count": token_counter.get_total(),
                "ground_truth": ground_truth,
                "evaluation": evaluation,
                **result
            }
            results.append(result_record)

            # 打印进度
            status = "[OK]" if result.get("success") else "[FAIL]"
            attr_acc = evaluation.get("attribute_accuracy", 0)
            hall = "[HALLUCINATION!]" if evaluation.get("hallucination_detected") else ""
            print(f"  {status} AttrAcc: {attr_acc:.0%} | Confidence: {result.get('confidence', 0):.2%} | Time: {elapsed_time:.1f}s | Token: {token_counter.get_total()} {hall}")

            # 每个用例完成后立即保存checkpoint
            all_results[config_key] = {
                "config": config,
                "results": results,
                "token_counter": token_counter,
            }
            save_checkpoint(all_results)

        all_results[config_key] = {
            "config": config,
            "results": results,
            "token_counter": token_counter,
        }

    # 如果resume且有跳过的用例，补回之前的结果
    if resume and completed_map:
        # 从checkpoint恢复之前的结果，但跳过后面重新跑的配置
        pass  # all_results 已经包含新跑的结果，checkpoint数据在需要时手动合并

    # 计算指标 + 生成图表 + 保存结果
    metrics = calculate_benchmark_metrics(all_results)
    chart_files = generate_visualization(metrics, all_results, output_dir)
    save_benchmark_results(all_results, metrics, output_dir)

    return {
        "all_results": all_results,
        "metrics": metrics,
        "chart_files": chart_files
    }


# ============================================================
# 指标计算
# ============================================================

def calculate_benchmark_metrics(all_results: Dict[str, Any]) -> Dict[str, Any]:
    """
    基于规则评估计算评测指标。

    综合准确率: 基于属性准确率（模型是否正确识别ground truth中的品类/颜色/风格）
    幻觉率: 基于矛盾检测（模型是否输出了与ground truth相矛盾的属性）

    Args:
        all_results: 各配置的运行结果

    Returns:
        指标字典
    """
    metrics = {}

    for config_key, data in all_results.items():
        results = data["results"]
        total = len(results)
        success = [r for r in results if r.get("success")]

        if not success:
            metrics[config_key] = {
                "accuracy": 0,
                "hallucination_rate": 1.0,
                "avg_response_time": 0,
                "total_tokens": 0,
                "avg_confidence": 0,
            }
            continue

        # 1. 综合准确率：各样本属性准确率的平均值
        attr_accuracies = []
        for r in success:
            eval_data = r.get("evaluation", {})
            attr_accuracies.append(eval_data.get("attribute_accuracy", 0))
        accuracy = sum(attr_accuracies) / len(attr_accuracies)

        # 2. 幻觉率：存在属性矛盾的样本比例
        hallucination_count = sum(
            1 for r in success
            if r.get("evaluation", {}).get("hallucination_detected", False)
        )
        hallucination_rate = hallucination_count / len(success)

        # 3. 平均响应时间
        avg_time = sum(r.get("elapsed_time", 0) for r in success) / len(success)

        # 4. Token消耗量
        total_tokens = data["token_counter"].get_total()

        # 额外指标
        avg_confidence = sum(r.get("confidence", 0) for r in success) / len(success)

        # 详细的属性级别统计
        color_recall = sum(
            1 for r in success
            if r.get("evaluation", {}).get("mentioned_gt_color", False)
        ) / len(success) if success else 0

        product_recall = sum(
            1 for r in success
            if r.get("evaluation", {}).get("mentioned_gt_product", False)
        ) / len(success) if success else 0

        usage_recall = sum(
            1 for r in success
            if r.get("evaluation", {}).get("mentioned_gt_usage", False)
        ) / len(success) if success else 0

        metrics[config_key] = {
            "accuracy": accuracy,
            "hallucination_rate": hallucination_rate,
            "avg_response_time": avg_time,
            "total_tokens": total_tokens,
            "avg_tokens_per_case": total_tokens / len(success) if success else 0,
            "avg_confidence": avg_confidence,
            "success_count": len(success),
            "total_count": total,
            "accurate_count": sum(1 for a in attr_accuracies if a >= 1.0),
            "hallucination_count": hallucination_count,
            # 详细的属性级召回率
            "color_recall": color_recall,
            "product_recall": product_recall,
            "usage_recall": usage_recall,
        }

    return metrics


# ============================================================
# 可视化生成
# ============================================================

def generate_visualization(
    metrics: Dict[str, Any],
    all_results: Dict[str, Any],
    output_dir: str
) -> List[str]:
    """
    生成可视化图表

    Args:
        metrics: 评测指标
        all_results: 各配置的运行结果
        output_dir: 输出目录

    Returns:
        生成的图表文件路径列表
    """
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    chart_files = []

    # 图1: 四指标对比柱状图
    file1 = _plot_metrics_comparison(metrics, output_dir, timestamp)
    chart_files.append(file1)

    # 图2: 逐样本置信度曲线
    file2 = _plot_confidence_curves(all_results, output_dir, timestamp)
    chart_files.append(file2)

    # 图3: 响应时间分布
    file3 = _plot_response_time_distribution(all_results, output_dir, timestamp)
    chart_files.append(file3)

    # 图4: 综合雷达图
    file4 = _plot_radar_chart(metrics, output_dir, timestamp)
    chart_files.append(file4)

    # 图5: Token消耗对比
    file5 = _plot_token_consumption(metrics, output_dir, timestamp)
    chart_files.append(file5)

    print(f"\n可视化图表已生成:")
    for f in chart_files:
        print(f"  {f}")

    return chart_files


def _plot_metrics_comparison(metrics: Dict[str, Any], output_dir: str, timestamp: str) -> str:
    """绘制四指标对比柱状图"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('三种配置评测指标对比', fontsize=16, fontweight='bold')

    configs = list(metrics.keys())
    config_names = [CONFIGS[c]["name"] for c in configs]
    colors = ['#3498db', '#2ecc71', '#e74c3c']

    # 1. 综合准确率
    ax = axes[0, 0]
    values = [metrics[c]["accuracy"] for c in configs]
    bars = ax.bar(config_names, values, color=colors, width=0.6)
    ax.set_title('综合准确率', fontsize=12)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel('准确率')
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.1%}', ha='center', va='bottom', fontsize=11)
    ax.grid(axis='y', alpha=0.3)

    # 2. 幻觉率
    ax = axes[0, 1]
    values = [metrics[c]["hallucination_rate"] for c in configs]
    bars = ax.bar(config_names, values, color=colors, width=0.6)
    ax.set_title('幻觉率（事实错误率）', fontsize=12)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel('幻觉率')
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.1%}', ha='center', va='bottom', fontsize=11)
    ax.grid(axis='y', alpha=0.3)

    # 3. 平均响应时间
    ax = axes[1, 0]
    values = [metrics[c]["avg_response_time"] for c in configs]
    bars = ax.bar(config_names, values, color=colors, width=0.6)
    ax.set_title('平均响应时间', fontsize=12)
    ax.set_ylabel('时间（秒）')
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{val:.1f}s', ha='center', va='bottom', fontsize=11)
    ax.grid(axis='y', alpha=0.3)

    # 4. Token消耗量
    ax = axes[1, 1]
    values = [metrics[c]["avg_tokens_per_case"] for c in configs]
    bars = ax.bar(config_names, values, color=colors, width=0.6)
    ax.set_title('平均Token消耗（每样本）', fontsize=12)
    ax.set_ylabel('Tokens')
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20,
                f'{int(val)}', ha='center', va='bottom', fontsize=11)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    filepath = os.path.join(output_dir, f"benchmark_metrics_{timestamp}.png")
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()

    return filepath


def _plot_confidence_curves(all_results: Dict[str, Any], output_dir: str, timestamp: str) -> str:
    """绘制逐样本置信度曲线"""
    fig, ax = plt.subplots(figsize=(12, 6))

    colors = {'vision_only': '#3498db', 'vision_main': '#2ecc71', 'full_system': '#e74c3c'}

    for config_key, data in all_results.items():
        results = data["results"]
        confidences = [r.get("confidence", 0) for r in results]
        x = range(1, len(confidences) + 1)

        ax.plot(x, confidences, '-o', color=colors.get(config_key, '#333'),
                label=CONFIGS[config_key]["name"], linewidth=2, markersize=5, alpha=0.8)

    ax.axhline(y=0.7, color='gray', linestyle='--', alpha=0.5, label='阈值 (0.7)')
    ax.set_xlabel('样本序号', fontsize=12)
    ax.set_ylabel('置信度', fontsize=12)
    ax.set_title('逐样本置信度变化曲线', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 1.1)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    filepath = os.path.join(output_dir, f"confidence_curves_{timestamp}.png")
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()

    return filepath


def _plot_response_time_distribution(all_results: Dict[str, Any], output_dir: str, timestamp: str) -> str:
    """绘制响应时间分布"""
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = {'vision_only': '#3498db', 'vision_main': '#2ecc71', 'full_system': '#e74c3c'}

    for config_key, data in all_results.items():
        results = data["results"]
        times = [r.get("elapsed_time", 0) for r in results if r.get("success")]
        if times:
            ax.hist(times, bins=15, alpha=0.5, color=colors.get(config_key, '#333'),
                    label=CONFIGS[config_key]["name"], edgecolor='white')

    ax.set_xlabel('响应时间（秒）', fontsize=12)
    ax.set_ylabel('频次', fontsize=12)
    ax.set_title('响应时间分布', fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    filepath = os.path.join(output_dir, f"response_time_dist_{timestamp}.png")
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()

    return filepath


def _plot_radar_chart(metrics: Dict[str, Any], output_dir: str, timestamp: str) -> str:
    """绘制综合雷达图"""
    categories = ['准确率', '低幻觉率', '响应速度', 'Token效率']
    N = len(categories)

    # 计算角度
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    colors = {'vision_only': '#3498db', 'vision_main': '#2ecc71', 'full_system': '#e74c3c'}

    for config_key, m in metrics.items():
        # 归一化指标（值越高越好）
        values = [
            m["accuracy"],                          # 准确率
            1 - m["hallucination_rate"],             # 低幻觉率
            max(0, 1 - m["avg_response_time"] / 60), # 响应速度（60s为基准）
            max(0, 1 - m["avg_tokens_per_case"] / 5000),  # Token效率（5000为基准）
        ]
        values += values[:1]

        ax.plot(angles, values, 'o-', linewidth=2, color=colors.get(config_key, '#333'),
                label=CONFIGS[config_key]["name"])
        ax.fill(angles, values, alpha=0.1, color=colors.get(config_key, '#333'))

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=12)
    ax.set_ylim(0, 1)
    ax.set_title('综合性能雷达图', fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))

    plt.tight_layout()
    filepath = os.path.join(output_dir, f"radar_chart_{timestamp}.png")
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()

    return filepath


def _plot_token_consumption(metrics: Dict[str, Any], output_dir: str, timestamp: str) -> str:
    """绘制Token消耗对比"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    configs = list(metrics.keys())
    config_names = [CONFIGS[c]["name"] for c in configs]
    colors = ['#3498db', '#2ecc71', '#e74c3c']

    # 总Token消耗
    total_values = [metrics[c]["total_tokens"] for c in configs]
    bars = ax1.bar(config_names, total_values, color=colors, width=0.6)
    ax1.set_title('总Token消耗', fontsize=12)
    ax1.set_ylabel('Tokens')
    for bar, val in zip(bars, total_values):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
                f'{int(val)}', ha='center', va='bottom', fontsize=11)
    ax1.grid(axis='y', alpha=0.3)

    # 每样本平均Token
    avg_values = [metrics[c]["avg_tokens_per_case"] for c in configs]
    bars = ax2.bar(config_names, avg_values, color=colors, width=0.6)
    ax2.set_title('每样本平均Token消耗', fontsize=12)
    ax2.set_ylabel('Tokens')
    for bar, val in zip(bars, avg_values):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20,
                f'{int(val)}', ha='center', va='bottom', fontsize=11)
    ax2.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    filepath = os.path.join(output_dir, f"token_consumption_{timestamp}.png")
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()

    return filepath


# ============================================================
# 结果保存
# ============================================================

def save_benchmark_results(
    all_results: Dict[str, Any],
    metrics: Dict[str, Any],
    output_dir: str
):
    """保存评测结果"""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 保存详细结果
    for config_key, data in all_results.items():
        filepath = os.path.join(output_dir, f"benchmark_{config_key}_{timestamp}.jsonl")
        with open(filepath, 'w', encoding='utf-8') as f:
            for result in data["results"]:
                f.write(json.dumps(result, ensure_ascii=False, default=str) + "\n")

    # 保存指标汇总
    metrics_filepath = os.path.join(output_dir, f"benchmark_metrics_{timestamp}.json")
    with open(metrics_filepath, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False, default=str)

    # 打印指标汇总
    print(f"\n{'='*70}")
    print("评测指标汇总")
    print(f"{'='*70}")

    for config_key, m in metrics.items():
        config = CONFIGS[config_key]
        print(f"\n【{config['name']}】")
        print(f"  综合准确率（属性匹配）: {m['accuracy']:.1%}")
        print(f"  幻觉率（属性矛盾率）:   {m['hallucination_rate']:.1%}")
        print(f"    - 颜色召回率:         {m.get('color_recall', 0):.1%}")
        print(f"    - 品类召回率:         {m.get('product_recall', 0):.1%}")
        print(f"    - 风格召回率:         {m.get('usage_recall', 0):.1%}")
        print(f"  平均响应时间:           {m['avg_response_time']:.2f}s")
        print(f"  总Token消耗:            {m['total_tokens']}")
        print(f"  平均Token/样本:         {m['avg_tokens_per_case']:.0f}")
        print(f"  模型自报平均置信度:     {m['avg_confidence']:.2%}")
        print(f"  幻觉样本数/成功数:      {m['hallucination_count']}/{m['success_count']}")

    print(f"\n{'='*70}")
    print(f"结果已保存至: {output_dir}")
    print(f"{'='*70}")


# ============================================================
# 主函数
# ============================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="多Agent系统评测基准")
    parser.add_argument('-n', '--num', type=int, default=10, help='测试用例数量')
    parser.add_argument('--configs', nargs='+', default=None,
                        choices=list(CONFIGS.keys()),
                        help='要运行的配置')
    parser.add_argument('--resume', action='store_true',
                        help='从上次中断的checkpoint继续运行')

    args = parser.parse_args()

    try:
        result = run_benchmark(n=args.num, configs_to_run=args.configs, resume=args.resume)
    except KeyboardInterrupt:
        print(f"\n\n{'='*70}")
        print("用户中断! 正在从已保存的checkpoint生成图表和表格...")
        print(f"{'='*70}")

        # 从checkpoint加载已完成的中间结果
        import json as _json
        ckpt_path = _get_checkpoint_path()
        if os.path.exists(ckpt_path):
            with open(ckpt_path, "r", encoding="utf-8") as _f:
                ckpt_data = _json.load(_f)

            # 重建 all_results 结构
            all_results = {}
            for ck, cd in ckpt_data.items():
                tc = TokenCounter()
                tc.total_tokens = cd.get("token_total", 0)
                all_results[ck] = {
                    "config": CONFIGS.get(ck, {"name": ck, "description": ""}),
                    "results": cd.get("results", []),
                    "token_counter": tc,
                }

            if all_results:
                output_dir = os.path.join(os.path.dirname(__file__), "output")
                metrics = calculate_benchmark_metrics(all_results)
                chart_files = generate_visualization(metrics, all_results, output_dir)
                save_benchmark_results(all_results, metrics, output_dir)
                print(f"\n中断但已生成图表和表格，共 {sum(len(d['results']) for d in all_results.values())} 条结果。")
                print(f"恢复运行: python benchmark.py -n {args.num} --resume")
            else:
                print("checkpoint为空，无数据可生成。")
        else:
            print("未找到checkpoint文件，无数据可生成。")
    except Exception as e:
        print(f"\n严重错误: {e}")
        import traceback
        traceback.print_exc()
        print("\n尝试从checkpoint恢复...")
        # 同样尝试从checkpoint生成图表
        ckpt_path = _get_checkpoint_path()
        if os.path.exists(ckpt_path):
            print(f"checkpoint存在: {ckpt_path}")
            print(f"请运行: python benchmark.py -n {args.num} --resume")
