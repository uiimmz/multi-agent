"""
工具函数模块
包含图像处理、日志、API调用、评测等功能
"""
import os
import re
import json
import time
import base64
import random
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 非交互式后端

# ============================================================
# 日志配置
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("MultiAgent")


def add_log(state: dict, message: str) -> None:
    """
    向状态中添加带时间戳的日志

    Args:
        state: 当前状态
        message: 日志消息
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    log_entry = f"[{timestamp}] {message}"

    if "logs" not in state:
        state["logs"] = []
    state["logs"].append(log_entry)
    logger.info(message)


# ============================================================
# 图像处理
# ============================================================

def image_to_base64(image_path: str) -> str:
    """
    将图像文件转换为base64编码

    Args:
        image_path: 图像文件路径

    Returns:
        base64编码字符串
    """
    with open(image_path, "rb") as f:
        image_data = f.read()
    return base64.b64encode(image_data).decode("utf-8")


def get_image_media_type(image_path: str) -> str:
    """
    根据文件扩展名获取图像MIME类型

    Args:
        image_path: 图像文件路径

    Returns:
        MIME类型字符串
    """
    ext = Path(image_path).suffix.lower()
    media_types = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".gif": "image/gif",
        ".webp": "image/webp",
        ".bmp": "image/bmp",
    }
    return media_types.get(ext, "image/jpeg")


# ============================================================
# 响应解析
# ============================================================

def extract_confidence_from_text(text: str, default: float = 0.7) -> float:
    """
    从文本中提取置信度值
    支持格式: "置信度: 0.85" 或 "confidence: 0.85" 或 "置信度：0.85"

    Args:
        text: 包含置信度的文本
        default: 默认置信度

    Returns:
        提取的置信度值
    """
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

    logger.warning(f"无法从文本中提取置信度，使用默认值: {default}")
    return default


def extract_json_from_text(text: str) -> Optional[Dict[str, Any]]:
    """
    从文本中提取JSON对象

    Args:
        text: 包含JSON的文本

    Returns:
        提取的字典，失败返回None
    """
    # 尝试直接解析
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 尝试提取```json ... ```格式
    json_match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(1))
        except json.JSONDecodeError:
            pass

    # 尝试提取{ ... }格式
    brace_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', text, re.DOTALL)
    if brace_match:
        try:
            return json.loads(brace_match.group(0))
        except json.JSONDecodeError:
            pass

    return None


def parse_verify_response(text: str) -> Dict[str, Any]:
    """
    解析校验Agent的响应

    Args:
        text: 校验Agent的响应文本

    Returns:
        包含score, comment, passed的字典
    """
    result = {
        "score": 0.7,
        "comment": "无法解析校验结果",
        "passed": True
    }

    # 尝试提取JSON
    json_data = extract_json_from_text(text)
    if json_data:
        if "score" in json_data:
            result["score"] = float(json_data["score"])
        if "comment" in json_data:
            result["comment"] = json_data["comment"]
        if "passed" in json_data:
            result["passed"] = bool(json_data["passed"])
        return result

    # 尝试提取分数
    score_match = re.search(r'(?:score|分数|得分)[：:]\s*([\d.]+)', text, re.IGNORECASE)
    if score_match:
        result["score"] = float(score_match.group(1))

    result["passed"] = result["score"] >= 0.7
    return result


# ============================================================
# 置信度计算
# ============================================================

def calculate_final_confidence(
    vision_confidence: Optional[float],
    copy_confidence: Optional[float],
    verify_score: Optional[float],
    ablation_config: Dict[str, Any]
) -> float:
    """
    根据消融配置计算最终置信度

    Args:
        vision_confidence: 视觉Agent置信度
        copy_confidence: 文案Agent置信度
        verify_score: 校验Agent分数
        ablation_config: 消融配置

    Returns:
        最终置信度
    """
    mode = ablation_config.get("confidence_mode", "weighted")
    weights = ablation_config.get("weights", {})

    # 收集启用的Agent的置信度
    confidences = {}
    if ablation_config.get("enable_vision", True) and vision_confidence is not None:
        confidences["vision"] = vision_confidence
    if ablation_config.get("enable_copy", True) and copy_confidence is not None:
        confidences["copy"] = copy_confidence
    if ablation_config.get("enable_verify", True) and verify_score is not None:
        confidences["verify"] = verify_score

    if not confidences:
        return 0.7  # 默认值

    if mode == "min":
        return min(confidences.values())

    elif mode == "weighted":
        total_weight = 0
        weighted_sum = 0
        for key, conf in confidences.items():
            weight = weights.get(key, 1.0)
            weighted_sum += conf * weight
            total_weight += weight
        return weighted_sum / total_weight if total_weight > 0 else 0.7

    elif mode == "verify_only":
        return verify_score if verify_score is not None else 0.7

    elif mode == "model":
        # 由模型自行判断，这里取平均值
        return sum(confidences.values()) / len(confidences)

    return 0.7


# ============================================================
# 评测指标
# ============================================================

def calculate_metrics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    计算评测指标

    Args:
        results: 批量运行结果列表

    Returns:
        包含各项指标的字典
    """
    if not results:
        return {}

    total = len(results)
    success_count = sum(1 for r in results if r.get("success", False))
    total_time = sum(r.get("response_time", 0) for r in results)
    total_tokens = sum(r.get("token_count", 0) for r in results)

    # 计算置信度统计
    confidences = [r.get("final_confidence", 0) for r in results if r.get("success")]
    avg_confidence = sum(confidences) / len(confidences) if confidences else 0

    metrics = {
        "total_count": total,
        "success_count": success_count,
        "success_rate": success_count / total if total > 0 else 0,
        "average_response_time": total_time / total if total > 0 else 0,
        "total_response_time": total_time,
        "total_token_count": total_tokens,
        "average_token_count": total_tokens / total if total > 0 else 0,
        "average_confidence": avg_confidence,
    }

    return metrics


# ============================================================
# 可视化
# ============================================================

def plot_metrics(metrics: Dict[str, Any], output_dir: str = "output") -> List[str]:
    """
    生成评测指标可视化图表

    Args:
        metrics: 评测指标字典
        output_dir: 输出目录

    Returns:
        生成的图片文件路径列表
    """
    os.makedirs(output_dir, exist_ok=True)
    generated_files = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    # 图1: 综合指标柱状图
    fig, ax = plt.subplots(figsize=(10, 6))
    labels = ['成功率', '平均置信度']
    values = [metrics.get('success_rate', 0), metrics.get('average_confidence', 0)]
    colors = ['#2ecc71', '#3498db']
    bars = ax.bar(labels, values, color=colors, width=0.5)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel('数值')
    ax.set_title('综合评测指标')
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.2%}', ha='center', va='bottom', fontsize=12)
    plt.tight_layout()
    filepath1 = os.path.join(output_dir, f"metrics_summary_{timestamp}.png")
    plt.savefig(filepath1, dpi=150)
    plt.close()
    generated_files.append(filepath1)

    # 图2: 响应时间和Token消耗
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # 响应时间
    time_data = metrics.get('average_response_time', 0)
    ax1.bar(['平均响应时间'], [time_data], color='#e74c3c', width=0.3)
    ax1.set_ylabel('秒')
    ax1.set_title('平均响应时间')
    ax1.text(0, time_data + 0.1, f'{time_data:.2f}s', ha='center', fontsize=12)

    # Token消耗
    token_data = metrics.get('average_token_count', 0)
    ax2.bar(['平均Token消耗'], [token_data], color='#9b59b6', width=0.3)
    ax2.set_ylabel('Tokens')
    ax2.set_title('平均Token消耗')
    ax2.text(0, token_data + 50, f'{int(token_data)}', ha='center', fontsize=12)

    plt.tight_layout()
    filepath2 = os.path.join(output_dir, f"metrics_performance_{timestamp}.png")
    plt.savefig(filepath2, dpi=150)
    plt.close()
    generated_files.append(filepath2)

    logger.info(f"可视化图表已保存: {generated_files}")
    return generated_files


def plot_confidence_curve(confidences: List[float], output_dir: str = "output") -> str:
    """
    绘制置信度变化曲线

    Args:
        confidences: 置信度列表
        output_dir: 输出目录

    Returns:
        生成的图片文件路径
    """
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(range(1, len(confidences) + 1), confidences, 'b-o', linewidth=2, markersize=8)
    ax.axhline(y=0.7, color='r', linestyle='--', label='阈值 (0.7)')
    ax.set_xlabel('样本序号')
    ax.set_ylabel('置信度')
    ax.set_title('置信度变化曲线')
    ax.set_ylim(0, 1.1)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    filepath = os.path.join(output_dir, f"confidence_curve_{timestamp}.png")
    plt.savefig(filepath, dpi=150)
    plt.close()

    logger.info(f"置信度曲线已保存: {filepath}")
    return filepath


# ============================================================
# 批量结果保存
# ============================================================

def save_results_jsonl(results: List[Dict[str, Any]], output_path: str) -> None:
    """
    将结果保存为JSON Lines格式

    Args:
        results: 结果列表
        output_path: 输出文件路径
    """
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

    logger.info(f"结果已保存至: {output_path}")


# ============================================================
# 模拟校验 (当API不可用时)
# ============================================================

def simulate_verify_score() -> Tuple[float, str]:
    """
    模拟校验分数（当真实API不可用时）

    Returns:
        (分数, 评语) 元组
    """
    score = round(random.uniform(0.6, 0.9), 2)
    comments = [
        "内容质量良好，描述准确",
        "文案创意不错，但可进一步优化",
        "视觉描述基本准确，置信度适中",
        "整体质量可接受，建议微调",
        "内容一致性较好，通过校验",
    ]
    comment = random.choice(comments)
    return score, comment
