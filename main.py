"""
主程序入口
支持命令行单次调用和批量测试
"""
import sys
import os
import json
import argparse
from datetime import datetime

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import THRESHOLD_CONFIG, ABLATION_CONFIG, ABLATION_PRESETS
from graph import run_pipeline, batch_run
from utils import calculate_metrics, plot_metrics, plot_confidence_curve, save_results_jsonl


def single_run_demo():
    """
    单次调用演示
    """
    print("\n" + "="*60)
    print("多Agent电商商品多模态智能解析系统 - 单次调用演示")
    print("="*60)

    # 示例输入（实际使用时可通过命令行参数传入）
    image_path = input("\n请输入图像文件路径: ").strip()
    user_query = input("请输入用户查询: ").strip()

    if not os.path.exists(image_path):
        print(f"错误: 图像文件不存在 - {image_path}")
        return

    print(f"\n开始处理...")
    print(f"图像: {image_path}")
    print(f"查询: {user_query}")
    print("-"*60)

    # 运行流水线
    final_state = run_pipeline(image_path, user_query)

    # 输出结果
    print("\n" + "="*60)
    print("处理结果")
    print("="*60)

    print(f"\n任务类型: {final_state.get('task_type', 'unknown')}")
    print(f"\n最终置信度: {final_state.get('final_confidence', 0):.2%}")

    print(f"\n{'='*60}")
    print("最终输出")
    print(f"{'='*60}")
    print(final_state.get('final_output', '无输出'))

    # 输出各Agent结果
    print(f"\n{'='*60}")
    print("各Agent详情")
    print(f"{'='*60}")

    vision_result = final_state.get('vision_result', {})
    if vision_result:
        print(f"\n[视觉Agent]")
        print(f"  置信度: {vision_result.get('confidence', 0):.2%}")
        print(f"  描述: {vision_result.get('description', '无')[:100]}...")

    copy_result = final_state.get('copy_result', {})
    if copy_result:
        print(f"\n[文案Agent]")
        print(f"  置信度: {copy_result.get('confidence', 0):.2%}")
        print(f"  文案: {copy_result.get('copywriting', '无')[:100]}...")

    verify_result = final_state.get('verify_result', {})
    if verify_result:
        print(f"\n[校验Agent]")
        print(f"  分数: {verify_result.get('score', 0):.2%}")
        print(f"  通过: {verify_result.get('passed', False)}")
        print(f"  评语: {verify_result.get('comment', '无')}")

    # 输出重试信息
    retry_counts = final_state.get('retry_counts', {})
    if any(v > 0 for v in retry_counts.values()):
        print(f"\n[重试统计]")
        for agent, count in retry_counts.items():
            if count > 0:
                print(f"  {agent}: {count}次")

    # 输出消融标签
    ablation_tags = final_state.get('ablation_tags', [])
    if ablation_tags:
        print(f"\n[消融标签] {', '.join(ablation_tags)}")

    # 输出日志
    print(f"\n{'='*60}")
    print("执行日志")
    print(f"{'='*60}")
    for log in final_state.get('logs', []):
        print(f"  {log}")

    return final_state


def batch_run_demo(test_cases_file: str = None, ablation_preset: str = "full"):
    """
    批量运行演示

    Args:
        test_cases_file: 测试用例JSON文件路径
        ablation_preset: 消融预设名称
    """
    print("\n" + "="*60)
    print("多Agent电商商品多模态智能解析系统 - 批量测试")
    print("="*60)

    # 加载测试用例
    if test_cases_file and os.path.exists(test_cases_file):
        with open(test_cases_file, "r", encoding="utf-8") as f:
            test_cases = json.load(f)
    else:
        # 默认测试用例
        test_cases = [
            {"image_path": "test_images/test1.jpg", "query": "描述这张图片"},
            {"image_path": "test_images/test2.jpg", "query": "这是什么商品？"},
        ]
        print(f"警告: 未找到测试用例文件，使用默认测试用例")

    # 获取消融配置
    ablation_config = ABLATION_PRESETS.get(ablation_preset, ABLATION_CONFIG.copy())
    print(f"\n消融预设: {ablation_preset}")
    print(f"消融配置: {json.dumps(ablation_config, indent=2, ensure_ascii=False)}")

    # 批量运行
    results = batch_run(test_cases, ablation_config=ablation_config)

    # 计算指标
    metrics = calculate_metrics(results)

    # 输出结果
    print(f"\n{'='*60}")
    print("批量测试结果")
    print(f"{'='*60}")
    print(f"\n总用例数: {metrics.get('total_count', 0)}")
    print(f"成功数: {metrics.get('success_count', 0)}")
    print(f"成功率: {metrics.get('success_rate', 0):.2%}")
    print(f"平均响应时间: {metrics.get('average_response_time', 0):.2f}s")
    print(f"总Token消耗: {metrics.get('total_token_count', 0)}")
    print(f"平均Token消耗: {metrics.get('average_token_count', 0):.0f}")
    print(f"平均置信度: {metrics.get('average_confidence', 0):.2%}")

    # 保存结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    # 保存JSON Lines
    jsonl_path = os.path.join(output_dir, f"batch_results_{timestamp}.jsonl")
    save_results_jsonl(results, jsonl_path)
    print(f"\n结果已保存: {jsonl_path}")

    # 保存指标
    metrics_path = os.path.join(output_dir, f"metrics_{timestamp}.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"指标已保存: {metrics_path}")

    # 生成可视化
    print(f"\n生成可视化图表...")
    chart_files = plot_metrics(metrics, output_dir)
    for cf in chart_files:
        print(f"  图表: {cf}")

    # 生成置信度曲线
    confidences = [r.get("final_confidence", 0) for r in results if r.get("success")]
    if confidences:
        curve_file = plot_confidence_curve(confidences, output_dir)
        print(f"  置信度曲线: {curve_file}")

    return results, metrics


def main():
    """
    主函数
    """
    parser = argparse.ArgumentParser(description="多Agent电商商品多模态智能解析系统")
    parser.add_argument("--mode", choices=["single", "batch"], default="single",
                       help="运行模式: single(单次调用) 或 batch(批量测试)")
    parser.add_argument("--image", type=str, help="图像文件路径 (单次调用模式)")
    parser.add_argument("--query", type=str, help="用户查询 (单次调用模式)")
    parser.add_argument("--test-cases", type=str, help="测试用例JSON文件路径 (批量测试模式)")
    parser.add_argument("--ablation", type=str, default="full",
                       choices=list(ABLATION_PRESETS.keys()),
                       help="消融预设名称")

    args = parser.parse_args()

    if args.mode == "single":
        if args.image and args.query:
            # 命令行参数模式
            final_state = run_pipeline(args.image, args.query)
            print(final_state.get("final_output", ""))
        else:
            # 交互式模式
            single_run_demo()

    elif args.mode == "batch":
        batch_run_demo(args.test_cases, args.ablation)


if __name__ == "__main__":
    main()
