"""
主Agent模块
包含规划节点、路由决策节点、聚合节点
使用DeepSeek-V4-Pro模型进行任务规划和结果聚合
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
from typing import Dict, Any, Literal
from openai import OpenAI

from config import DEEPSEEK_BASE_URL, DEEPSEEK_API_KEY, DEEPSEEK_MODEL, DEFAULT_PLAN
from utils import extract_json_from_text, calculate_final_confidence, add_log


# ============================================================
# 规划节点
# ============================================================

def planning_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    规划节点
    调用DeepSeek-V4-Pro分析用户输入，生成执行计划

    Args:
        state: 当前状态字典

    Returns:
        更新后的状态字典（task_type, execution_plan字段）
    """
    add_log(state, ">>> 主Agent规划节点开始执行")

    user_query = state.get("user_query", "")
    image_path = state.get("image_path", "")

    # 构造提示词
    prompt = f"""请分析以下用户输入，判断任务类型并生成执行计划。

用户查询: {user_query}
图像路径: {image_path}

请返回JSON格式的计划，包含以下字段：
- task_type: 任务类型，"qa"(问答) 或 "describe"(描述)
- steps: 执行步骤列表，可选值为 "vision", "copy", "verify"
- reasoning: 规划理由

执行步骤说明：
- vision: 视觉理解，分析图像内容
- copy: 文案生成，基于视觉描述生成创意文案
- verify: 内容校验，验证生成内容的质量

默认执行顺序为: vision -> copy -> verify

返回格式示例：
```json
{{
    "task_type": "describe",
    "steps": ["vision", "copy", "verify"],
    "reasoning": "用户需要描述图像内容，需要完整的流水线处理"
}}
```"""

    # 调用API
    try:
        client = OpenAI(
            api_key=DEEPSEEK_API_KEY,
            base_url=DEEPSEEK_BASE_URL
        )

        add_log(state, f"调用规划模型: {DEEPSEEK_MODEL}")

        response = client.chat.completions.create(
            model=DEEPSEEK_MODEL,
            messages=[
                {"role": "system", "content": "你是一个任务规划专家，负责分析用户需求并制定执行计划。"},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1000,
            temperature=0.3
        )

        raw_response = response.choices[0].message.content
        add_log(state, f"规划模型响应成功")

        # 解析响应
        plan_data = extract_json_from_text(raw_response)

        if plan_data:
            task_type = plan_data.get("task_type", "describe")
            steps = plan_data.get("steps", ["vision", "copy", "verify"])
            reasoning = plan_data.get("reasoning", "")
        else:
            add_log(state, "警告: 无法解析规划响应，使用默认计划")
            task_type = DEFAULT_PLAN["task_type"]
            steps = DEFAULT_PLAN["steps"]
            reasoning = DEFAULT_PLAN["reasoning"]

        # 验证步骤有效性
        valid_steps = ["vision", "copy", "verify"]
        steps = [s for s in steps if s in valid_steps]
        if not steps:
            steps = ["vision", "copy", "verify"]

        result = {
            "task_type": task_type,
            "execution_plan": {
                "task_type": task_type,
                "steps": steps,
                "reasoning": reasoning
            },
            "routing_decision": "vision"  # 初始路由到视觉Agent
        }

        add_log(state, f"规划完成: 任务类型={task_type}, 步骤={steps}")
        return result

    except Exception as e:
        add_log(state, f"规划模型调用失败: {str(e)}，使用默认计划")
        return {
            "task_type": DEFAULT_PLAN["task_type"],
            "execution_plan": DEFAULT_PLAN,
            "routing_decision": "vision"
        }


# ============================================================
# 路由决策节点
# ============================================================

def routing_decision(state: Dict[str, Any]) -> Literal["vision", "copy", "verify", "aggregate"]:
    """
    路由决策函数
    根据当前状态决定下一步路由

    Args:
        state: 当前状态字典

    Returns:
        路由目标节点名称
    """
    add_log(state, ">>> 路由决策节点")

    execution_plan = state.get("execution_plan", {})
    steps = execution_plan.get("steps", ["vision", "copy", "verify"])
    ablation_config = state.get("ablation_config", {})

    # 检查各Agent的完成状态和消融状态
    vision_done = bool(state.get("vision_result", {}).get("description"))
    copy_done = bool(state.get("copy_result", {}).get("copywriting"))
    verify_done = bool(state.get("verify_result", {}).get("score") is not None)

    vision_enabled = ablation_config.get("enable_vision", True)
    copy_enabled = ablation_config.get("enable_copy", True)
    verify_enabled = ablation_config.get("enable_verify", True)

    # 按照执行计划顺序路由
    for step in steps:
        if step == "vision" and vision_enabled and not vision_done:
            add_log(state, "路由决策: -> vision")
            return "vision"
        elif step == "copy" and copy_enabled and not copy_done:
            # 文案Agent依赖视觉Agent的结果
            if vision_done or not vision_enabled:
                add_log(state, "路由决策: -> copy")
                return "copy"
        elif step == "verify" and verify_enabled and not verify_done:
            # 校验Agent依赖视觉和文案的结果
            if (vision_done or not vision_enabled) and (copy_done or not copy_enabled):
                add_log(state, "路由决策: -> verify")
                return "verify"

    # 所有步骤完成，进入聚合
    add_log(state, "路由决策: -> aggregate")
    return "aggregate"


# ============================================================
# 聚合节点
# ============================================================

def aggregation_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    聚合节点
    收集所有子Agent输出，计算最终置信度，决定是否重试或输出结果

    Args:
        state: 当前状态字典

    Returns:
        更新后的状态字典（final_output, final_confidence字段）
    """
    add_log(state, ">>> 主Agent聚合节点开始执行")

    # 获取各Agent结果
    vision_result = state.get("vision_result", {})
    copy_result = state.get("copy_result", {})
    verify_result = state.get("verify_result", {})

    vision_confidence = vision_result.get("confidence")
    copy_confidence = copy_result.get("confidence")
    verify_score = verify_result.get("score")

    ablation_config = state.get("ablation_config", {})
    threshold_config = state.get("threshold_config", {})
    retry_counts = state.get("retry_counts", {"vision": 0, "copy": 0, "verify": 0})

    # 计算最终置信度 - 由主Agent模型判断
    final_confidence = _ask_model_for_confidence(
        state,
        vision_confidence,
        copy_confidence,
        verify_score,
        vision_result,
        copy_result,
        verify_result
    )

    add_log(state, f"主Agent判断置信度: 视觉={vision_confidence}, 文案={copy_confidence}, 校验={verify_score}")
    add_log(state, f"最终置信度: {final_confidence:.2f}")

    # 检查是否需要重试
    final_threshold = threshold_config.get("final_threshold", 0.7)
    max_retries = threshold_config.get("max_retries", 3)

    if final_confidence < final_threshold:
        # 找出置信度最低的环节
        lowest_agent = _find_lowest_confidence_agent(
            vision_confidence, copy_confidence, verify_score,
            ablation_config, retry_counts, max_retries
        )

        if lowest_agent and retry_counts.get(lowest_agent, 0) < max_retries:
            add_log(state, f"置信度低于阈值，重试{lowest_agent}Agent")
            return {
                "routing_decision": lowest_agent,
                "retry_counts": {
                    **retry_counts,
                    lowest_agent: retry_counts.get(lowest_agent, 0) + 1
                }
            }

    # 生成最终输出
    task_type = state.get("task_type", "describe")
    user_query = state.get("user_query", "")

    final_output = _generate_final_output(
        task_type, user_query,
        vision_result, copy_result, verify_result,
        final_confidence
    )

    add_log(state, f"聚合完成，最终置信度: {final_confidence:.2f}")

    return {
        "final_output": final_output,
        "final_confidence": final_confidence,
        "routing_decision": "end"
    }


def _ask_model_for_confidence(
    state: Dict[str, Any],
    vision_conf: float,
    copy_conf: float,
    verify_score: float,
    vision_result: Dict,
    copy_result: Dict,
    verify_result: Dict
) -> float:
    """
    调用主Agent模型判断最终置信度

    Args:
        state: 当前状态
        vision_conf: 视觉置信度
        copy_conf: 文案置信度
        verify_score: 校验分数
        vision_result: 视觉结果
        copy_result: 文案结果
        verify_result: 校验结果

    Returns:
        模型判断的最终置信度
    """
    # 构造提示词
    vision_desc = vision_result.get("description", "")[:200]
    copywriting = copy_result.get("copywriting", "")[:200]
    verify_comment = verify_result.get("comment", "")

    prompt = f"""请根据以下各Agent的输出结果，判断最终的综合置信度（0-1之间的数字）。

视觉Agent:
- 置信度: {vision_conf}
- 描述摘要: {vision_desc}

文案Agent:
- 置信度: {copy_conf}
- 文案摘要: {copywriting}

校验Agent:
- 分数: {verify_score}
- 评语: {verify_comment}

请综合考虑：
1. 各Agent自身的置信度/分数
2. 描述和文案的质量、一致性
3. 校验结果的可靠性

只返回一个0-1之间的数字，不要返回其他内容。例如: 0.82"""

    try:
        client = OpenAI(
            api_key=DEEPSEEK_API_KEY,
            base_url=DEEPSEEK_BASE_URL
        )

        response = client.chat.completions.create(
            model=DEEPSEEK_MODEL,
            messages=[
                {"role": "system", "content": "你是一个置信度评估专家，负责综合判断多Agent系统的输出质量。只返回数字。"},
                {"role": "user", "content": prompt}
            ],
            max_tokens=500,
            temperature=0.1
        )

        raw = response.choices[0].message.content.strip()
        # 提取数字
        import re
        match = re.search(r'(\d+\.?\d*)', raw)
        if match:
            value = float(match.group(1))
            if 0 <= value <= 1:
                add_log(state, f"主Agent判断置信度: {value:.2f}")
                return value

        # 解析失败，使用默认计算
        add_log(state, "主Agent置信度解析失败，使用默认计算")

    except Exception as e:
        add_log(state, f"主Agent置信度判断失败: {str(e)}，使用默认计算")

    # 回退到默认计算
    return calculate_final_confidence(vision_conf, copy_conf, verify_score, state.get("ablation_config", {}))


def _find_lowest_confidence_agent(
    vision_conf: float,
    copy_conf: float,
    verify_score: float,
    ablation_config: Dict[str, Any],
    retry_counts: Dict[str, int],
    max_retries: int
) -> str:
    """
    找出置信度最低且可重试的Agent

    Returns:
        Agent名称，或空字符串表示无需重试
    """
    candidates = {}

    if ablation_config.get("enable_vision", True) and vision_conf is not None:
        if retry_counts.get("vision", 0) < max_retries:
            candidates["vision"] = vision_conf

    if ablation_config.get("enable_copy", True) and copy_conf is not None:
        if retry_counts.get("copy", 0) < max_retries:
            candidates["copy"] = copy_conf

    if ablation_config.get("enable_verify", True) and verify_score is not None:
        if retry_counts.get("verify", 0) < max_retries:
            candidates["verify"] = verify_score

    if not candidates:
        return ""

    return min(candidates, key=candidates.get)


def _generate_final_output(
    task_type: str,
    user_query: str,
    vision_result: Dict[str, Any],
    copy_result: Dict[str, Any],
    verify_result: Dict[str, Any],
    final_confidence: float
) -> str:
    """
    生成最终输出文本

    Args:
        task_type: 任务类型
        user_query: 用户查询
        vision_result: 视觉Agent结果
        copy_result: 文案Agent结果
        verify_result: 校验Agent结果
        final_confidence: 最终置信度

    Returns:
        最终输出文本
    """
    vision_desc = vision_result.get("description", "")
    copywriting = copy_result.get("copywriting", "")
    verify_comment = verify_result.get("comment", "")
    verify_score = verify_result.get("score", 0)
    verify_passed = verify_result.get("passed", False)

    if task_type == "qa":
        # 问答模式：直接回答用户问题
        output = f"""【问答结果】

{copywriting if copywriting and '[消融]' not in copywriting else vision_desc}

---
置信度: {final_confidence:.2%}
校验状态: {'通过' if verify_passed else '未通过'} (分数: {verify_score:.2f})
校验评语: {verify_comment}"""
    else:
        # 描述模式：输出视觉描述+生成文案+校验结论
        output = f"""【图像描述】

{vision_desc}

【创意文案】

{copywriting}

【校验结论】
- 校验分数: {verify_score:.2f}
- 校验状态: {'通过' if verify_passed else '未通过'}
- 评语: {verify_comment}

【综合置信度】: {final_confidence:.2%}"""

    return output


# ============================================================
# 重试时清空对应Agent的输出
# ============================================================

def reset_agent_output(state: Dict[str, Any], agent_name: str) -> Dict[str, Any]:
    """
    重试前清空指定Agent的输出

    Args:
        state: 当前状态
        agent_name: Agent名称 ("vision", "copy", "verify")

    Returns:
        更新后的状态
    """
    add_log(state, f"清空{agent_name}Agent的输出，准备重试")

    updates = {}
    if agent_name == "vision":
        updates["vision_result"] = {}
    elif agent_name == "copy":
        updates["copy_result"] = {}
    elif agent_name == "verify":
        updates["verify_result"] = {}

    updates["routing_decision"] = agent_name
    return updates
