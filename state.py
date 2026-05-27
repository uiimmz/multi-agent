"""
状态定义模块
定义LangGraph的状态结构
"""
from typing import TypedDict, Dict, List, Any, Optional
from typing_extensions import Annotated


class VisionResult(TypedDict, total=False):
    """视觉Agent输出结果"""
    description: str           # 视觉描述
    confidence: float          # 置信度
    raw_response: str          # 原始响应


class CopyResult(TypedDict, total=False):
    """文案Agent输出结果"""
    copywriting: str           # 生成的文案
    confidence: float          # 置信度
    raw_response: str          # 原始响应


class VerifyResult(TypedDict, total=False):
    """校验Agent输出结果"""
    score: float               # 校验分数 (0-1)
    comment: str               # 评语
    passed: bool               # 是否通过
    raw_response: str          # 原始响应


class RetryCounts(TypedDict):
    """重试次数记录"""
    vision: int
    copy: int
    verify: int


class AgentState(TypedDict, total=False):
    """
    多Agent系统的完整状态
    """
    # ============ 输入 ============
    image_path: str                    # 图像文件路径
    image_base64: str                  # 图像base64编码
    user_query: str                    # 用户查询文本

    # ============ 任务规划 ============
    task_type: str                     # 任务类型: "qa"(问答) 或 "describe"(描述)
    execution_plan: Dict[str, Any]     # 执行计划 (JSON结构)
    routing_decision: str              # 路由决策: "vision" / "copy" / "verify" / "aggregate" / "end"

    # ============ 视觉Agent输出 ============
    vision_result: VisionResult        # 视觉描述结果

    # ============ 文案Agent输出 ============
    copy_result: CopyResult            # 文案生成结果

    # ============ 校验Agent输出 ============
    verify_result: VerifyResult        # 校验结果

    # ============ 最终输出 ============
    final_output: str                  # 最终输出文本
    final_confidence: float            # 最终置信度

    # ============ 控制配置 ============
    retry_counts: RetryCounts          # 各Agent重试次数
    threshold_config: Dict[str, Any]   # 阈值配置字典
    ablation_config: Dict[str, Any]    # 消融配置字典
    ablation_tags: List[str]           # 消融标签列表

    # ============ 日志 ============
    logs: List[str]                    # 日志列表


def create_initial_state(
    image_path: str,
    user_query: str,
    threshold_config: Dict[str, Any],
    ablation_config: Dict[str, Any]
) -> AgentState:
    """
    创建初始状态

    Args:
        image_path: 图像文件路径
        user_query: 用户查询文本
        threshold_config: 阈值配置
        ablation_config: 消融配置

    Returns:
        初始状态字典
    """
    return AgentState(
        # 输入
        image_path=image_path,
        image_base64="",
        user_query=user_query,

        # 任务规划
        task_type="",
        execution_plan={},
        routing_decision="",

        # 视觉Agent输出
        vision_result=VisionResult(),

        # 文案Agent输出
        copy_result=CopyResult(),

        # 校验Agent输出
        verify_result=VerifyResult(),

        # 最终输出
        final_output="",
        final_confidence=0.0,

        # 控制配置
        retry_counts=RetryCounts(vision=0, copy=0, verify=0),
        threshold_config=threshold_config,
        ablation_config=ablation_config,
        ablation_tags=[],

        # 日志
        logs=[]
    )
