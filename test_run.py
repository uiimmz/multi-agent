"""
批量测试脚本
运行多个测试用例，收集指标，生成可视化图表
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from graph import run_pipeline
from utils import plot_metrics, plot_confidence_curve


def load_test_dataset(n=5):
    """加载测试数据集"""
    with open('test_dataset/dataset.jsonl', 'r', encoding='utf-8') as f:
        lines = f.readlines()

    step = max(1, len(lines) // n)
    test_cases = []
    for i in range(0, len(lines), step):
        item = json.loads(lines[i])
        img_path = os.path.join('test_dataset', item['image_path'])
        if os.path.exists(img_path):
            test_cases.append({
                "title": item.get("title", f"Item {i}"),
                "image_path": img_path,
                "query": "Describe this fashion item and write creative marketing copy for it",
                "category": item.get("category", "unknown")
            })
        if len(test_cases) >= n:
            break

    return test_cases


def run_batch_test(n=5):
    """运行批量测试"""
    test_cases = load_test_dataset(n)
    print(f"Loaded {len(test_cases)} test cases")

    results = []
    for i, case in enumerate(test_cases):
        print(f"\n{'='*60}")
        print(f"[{i+1}/{len(test_cases)}] {case['title']}")
        print(f"Image: {case['image_path']}")
        print(f"{'='*60}")

        start_time = time.time()
        try:
            state = run_pipeline(case['image_path'], case['query'])
            elapsed = time.time() - start_time

            result = {
                "test_title": case['title'],
                "test_image": case['image_path'],
                "category": case['category'],
                "elapsed_time": elapsed,
                "final_confidence": state.get("final_confidence", 0),
                "vision_confidence": state.get("vision_result", {}).get("confidence", 0),
                "copy_confidence": state.get("copy_result", {}).get("confidence", 0),
                "verify_score": state.get("verify_result", {}).get("score", 0),
                "verify_passed": state.get("verify_result", {}).get("passed", False),
                "vision_desc": state.get("vision_result", {}).get("description", "")[:200],
                "copywriting": state.get("copy_result", {}).get("copywriting", "")[:200],
                "verify_comment": state.get("verify_result", {}).get("comment", ""),
                "success": True
            }
            results.append(result)

            print(f"  Vision confidence: {result['vision_confidence']:.2f}")
            print(f"  Copy confidence: {result['copy_confidence']:.2f}")
            print(f"  Verify score: {result['verify_score']:.2f}")
            print(f"  Final confidence: {result['final_confidence']:.2%}")
            print(f"  Time: {elapsed:.1f}s")

        except Exception as e:
            elapsed = time.time() - start_time
            print(f"  ERROR: {e}")
            results.append({
                "test_title": case['title'],
                "test_image": case['image_path'],
                "category": case['category'],
                "elapsed_time": elapsed,
                "final_confidence": 0,
                "success": False,
                "error": str(e)
            })

    # 保存原始结果
    with open('test_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)

    print(f"\n{'='*60}")
    print(f"All {len(results)} tests completed")
    print(f"Results saved to test_results.json")

    # 计算指标
    metrics = compute_metrics(results)
    print(f"\n{'='*60}")
    print("Evaluation Metrics:")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")

    # 生成图表
    try:
        charts = plot_metrics(metrics, output_dir="output")
        print(f"\nCharts saved: {charts}")
    except Exception as e:
        print(f"Chart generation failed: {e}")

    try:
        confidences = [r['final_confidence'] for r in results if r.get('success')]
        if confidences:
            curve = plot_confidence_curve(confidences, output_dir="output")
            print(f"Confidence curve: {curve}")
    except Exception as e:
        print(f"Confidence curve failed: {e}")

    return results, metrics


def compute_metrics(results):
    """计算评测指标"""
    total = len(results)
    success = [r for r in results if r.get('success')]
    success_count = len(success)

    if not success:
        return {"total": total, "success_count": 0, "success_rate": 0}

    confidences = [r['final_confidence'] for r in success]
    times = [r['elapsed_time'] for r in success]
    vision_confs = [r.get('vision_confidence', 0) for r in success]
    copy_confs = [r.get('copy_confidence', 0) for r in success]
    verify_scores = [r.get('verify_score', 0) for r in success]

    return {
        "total_count": total,
        "success_count": success_count,
        "success_rate": success_count / total,
        "average_confidence": sum(confidences) / len(confidences),
        "min_confidence": min(confidences),
        "max_confidence": max(confidences),
        "average_response_time": sum(times) / len(times),
        "total_response_time": sum(times),
        "average_vision_confidence": sum(vision_confs) / len(vision_confs),
        "average_copy_confidence": sum(copy_confs) / len(copy_confs),
        "average_verify_score": sum(verify_scores) / len(verify_scores),
        "high_confidence_count": sum(1 for c in confidences if c >= 0.8),
        "high_confidence_rate": sum(1 for c in confidences if c >= 0.8) / len(confidences),
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--num', type=int, default=5, help='Number of test cases')
    args = parser.parse_args()

    run_batch_test(args.num)
