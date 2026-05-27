"""
LangGraph图定义模块
定义多Agent系统的执行流程
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from typing import Dict, Any, TypedDict, Annotated
from langgraph.graph import StateGraph, END

from state import AgentState, create_initial_state
from agents.main_agent import planning_node, routing_decision, aggregation_node, reset_agent_output
from agents.vision_agent import vision_agent
from agents.copy_agent import copy_agent
from agents.verify_agent import verify_agent


# ============================================================
# 辅助函数：合并状态更新
# ============================================================

def merge_state_updates(current: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    """
    合并状态更新，用于reducer

    Args:
        current: 当前状态
        updates: 更新内容

    Returns:
        合并后的状态
    """
    result = current.copy()
    for key, value in updates.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = {**result[key], **value}
        elif key == "logs" and isinstance(result.get("logs"), list) and isinstance(value, list):
            result["logs"] = result["logs"] + value
        elif key == "ablation_tags" and isinstance(result.get("ablation_tags"), list) and isinstance(value, list):
            result["ablation_tags"] = result["ablation_tags"] + value
        else:
            result[key] = value
    return result


# ============================================================
# 包装Agent节点函数
# ============================================================

def vision_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    视觉Agent节点包装
    执行后无条件返回路由节点
    """
    result = vision_agent(state)
    # 设置路由决策为返回路由节点
    result["routing_decision"] = "router"
    return result


def copy_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    文案Agent节点包装
    执行后无条件返回路由节点
    """
    result = copy_agent(state)
    result["routing_decision"] = "router"
    return result


def verify_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    校验Agent节点包装
    执行后无条件返回路由节点
    """
    result = verify_agent(state)
    result["routing_decision"] = "router"
    return result


def router_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    路由节点
    调用路由决策函数，返回路由决策结果
    """
    decision = routing_decision(state)
    return {"routing_decision": decision}


def aggregation_wrapper(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    聚合节点包装
    处理重试逻辑
    """
    result = aggregation_node(state)

    # 如果需要重试，清空对应Agent的输出
    routing = result.get("routing_decision", "end")
    if routing in ["vision", "copy", "verify"]:
        reset_updates = reset_agent_output(state, routing)
        result = {**result, **reset_updates}

    return result


# ============================================================
# 条件边函数
# ============================================================

def route_after_planning(state: Dict[str, Any]) -> str:
    """
    规划后的路由
    根据执行计划决定进入哪个Agent
    """
    execution_plan = state.get("execution_plan", {})
    steps = execution_plan.get("steps", ["vision", "copy", "verify"])
    ablation_config = state.get("ablation_config", {})

    # 找到第一个需要执行的步骤
    for step in steps:
        if step == "vision" and ablation_config.get("enable_vision", True):
            return "vision"
        elif step == "copy" and ablation_config.get("enable_copy", True):
            return "copy"
        elif step == "verify" and ablation_config.get("enable_verify", True):
            return "verify"

    # 如果所有步骤都被消融，直接进入聚合
    return "aggregate"


def route_after_router(state: Dict[str, Any]) -> str:
    """
    路由节点后的分发
    根据routing_decision决定下一步
    """
    decision = state.get("routing_decision", "end")

    if decision == "vision":
        return "vision"
    elif decision == "copy":
        return "copy"
    elif decision == "verify":
        return "verify"
    elif decision == "aggregate":
        return "aggregate"
    else:
        return "end"


def route_after_aggregation(state: Dict[str, Any]) -> str:
    """
    聚合后的路由
    判断是结束还是需要重试
    """
    decision = state.get("routing_decision", "end")

    if decision in ["vision", "copy", "verify"]:
        return decision  # 需要重试
    else:
        return "end"  # 结束


# ============================================================
# 构建LangGraph图
# ============================================================

def build_graph() -> StateGraph:
    """
    构建多Agent系统的LangGraph图

    Returns:
        编译后的StateGraph
    """
    # 创建状态图
    workflow = StateGraph(AgentState)

    # 添加节点
    workflow.add_node("planning", planning_node)       # 规划节点
    workflow.add_node("router", router_node)           # 路由节点
    workflow.add_node("vision", vision_node)           # 视觉Agent
    workflow.add_node("copy", copy_node)               # 文案Agent
    workflow.add_node("verify", verify_node)           # 校验Agent
    workflow.add_node("aggregate", aggregation_wrapper) # 聚合节点

    # 设置入口点
    workflow.set_entry_point("planning")

    # 规划节点 -> 路由节点（根据执行计划决定）
    workflow.add_conditional_edges(
        "planning",
        route_after_planning,
        {
            "vision": "vision",
            "copy": "copy",
            "verify": "verify",
            "aggregate": "aggregate"
        }
    )

    # 子Agent执行后 -> 路由节点
    workflow.add_edge("vision", "router")
    workflow.add_edge("copy", "router")
    workflow.add_edge("verify", "router")

    # 路由节点 -> 分发到子Agent或聚合
    workflow.add_conditional_edges(
        "router",
        route_after_router,
        {
            "vision": "vision",
            "copy": "copy",
            "verify": "verify",
            "aggregate": "aggregate",
            "end": END
        }
    )

    # 聚合节点 -> 结束或重试
    workflow.add_conditional_edges(
        "aggregate",
        route_after_aggregation,
        {
            "vision": "vision",
            "copy": "copy",
            "verify": "verify",
            "end": END
        }
    )

    # 编译图
    app = workflow.compile()

    return app


# ============================================================
# 单次运行函数
# ============================================================

def run_pipeline(
    image_path: str,
    user_query: str,
    threshold_config: Dict[str, Any] = None,
    ablation_config: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    运行单次流水线

    Args:
        image_path: 图像文件路径
        user_query: 用户查询
        threshold_config: 阈值配置覆盖
        ablation_config: 消融配置覆盖

    Returns:
        最终状态字典
    """
    from config import THRESHOLD_CONFIG, ABLATION_CONFIG

    # 使用提供的配置或默认配置
    t_config = threshold_config or THRESHOLD_CONFIG.copy()
    a_config = ablation_config or ABLATION_CONFIG.copy()

    # 创建初始状态
    initial_state = create_initial_state(
        image_path=image_path,
        user_query=user_query,
        threshold_config=t_config,
        ablation_config=a_config
    )

    # 构建并运行图
    app = build_graph()
    final_state = app.invoke(initial_state)

    return final_state


# ============================================================
# 批量运行函数
# ============================================================

def batch_run(
    test_cases: list,
    threshold_config: Dict[str, Any] = None,
    ablation_config: Dict[str, Any] = None
) -> list:
    """
    批量运行测试用例

    Args:
        test_cases: 测试用例列表，每个元素为 {"image_path": str, "query": str}
        threshold_config: 阈值配置
        ablation_config: 消融配置

    Returns:
        结果列表
    """
    import time
    from config import THRESHOLD_CONFIG, ABLATION_CONFIG

    t_config = threshold_config or THRESHOLD_CONFIG.copy()
    a_config = ablation_config or ABLATION_CONFIG.copy()

    results = []

    for i, case in enumerate(test_cases):
        print(f"\n{'='*50}")
        print(f"处理测试用例 {i+1}/{len(test_cases)}")
        print(f"图像: {case['image_path']}")
        print(f"查询: {case['query']}")
        print(f"{'='*50}")

        start_time = time.time()

        try:
            final_state = run_pipeline(
                image_path=case["image_path"],
                user_query=case["query"],
                threshold_config=t_config,
                ablation_config=a_config
            )

            response_time = time.time() - start_time

            # 估算token消耗（简单估算）
            token_count = 0
            for log in final_state.get("logs", []):
                token_count += len(log) // 4  # 粗略估算

            result = {
                "case_id": i + 1,
                "image_path": case["image_path"],
                "query": case["query"],
                "success": bool(final_state.get("final_output")),
                "final_output": final_state.get("final_output", ""),
                "final_confidence": final_state.get("final_confidence", 0),
                "response_time": response_time,
                "token_count": token_count,
                "task_type": final_state.get("task_type", ""),
                "vision_confidence": final_state.get("vision_result", {}).get("confidence", 0),
                "copy_confidence": final_state.get("copy_result", {}).get("confidence", 0),
                "verify_score": final_state.get("verify_result", {}).get("score", 0),
                "retry_counts": final_state.get("retry_counts", {}),
                "ablation_tags": final_state.get("ablation_tags", []),
                "logs": final_state.get("logs", [])
            }

        except Exception as e:
            response_time = time.time() - start_time
            result = {
                "case_id": i + 1,
                "image_path": case["image_path"],
                "query": case["query"],
                "success": False,
                "final_output": f"执行失败: {str(e)}",
                "final_confidence": 0,
                "response_time": response_time,
                "token_count": 0,
                "error": str(e)
            }

        results.append(result)

    return results
