"""
内容校验Agent
使用MiMo-V2.5-Pro模型进行内容校验
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Dict, Any
from openai import OpenAI

from config import MIMO_BASE_URL, MIMO_API_KEY, MIMO_MODEL
from utils import parse_verify_response, simulate_verify_score, add_log


def verify_agent(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    内容校验Agent
    接收图像base64、视觉描述、生成文案，返回校验结果

    Args:
        state: 当前状态字典

    Returns:
        更新后的状态字典（verify_result字段）
    """
    add_log(state, ">>> 校验Agent开始执行")

    # 检查消融开关
    if not state.get("ablation_config", {}).get("enable_verify", True):
        add_log(state, "校验Agent已被消融，跳过执行")
        return {
            "verify_result": {
                "score": 0.7,
                "comment": "[消融] 校验Agent已禁用",
                "passed": True,
                "raw_response": ""
            },
            "ablation_tags": state.get("ablation_tags", []) + ["no_verify"]
        }

    # 获取输入数据
    image_base64 = state.get("image_base64", "")
    vision_result = state.get("vision_result", {})
    copy_result = state.get("copy_result", {})

    vision_description = vision_result.get("description", "")
    copywriting = copy_result.get("copywriting", "")
    user_query = state.get("user_query", "")

    # 检查API密钥是否配置
    if not MIMO_API_KEY or MIMO_API_KEY == "#请填入api":
        add_log(state, "MiMo API未配置，使用模拟校验")
        score, comment = simulate_verify_score()
        return {
            "verify_result": {
                "score": score,
                "comment": comment,
                "passed": score >= 0.7,
                "raw_response": f"[模拟校验] 分数: {score}, 评语: {comment}"
            }
        }

    # 构造提示词
    prompt = f"""请对以下内容进行质量校验：

用户查询: {user_query}

视觉描述:
{vision_description}

生成文案:
{copywriting}

请从以下几个方面进行评估：
1. 视觉描述与图像内容的一致性
2. 文案与视觉描述的相关性
3. 文案的质量和吸引力
4. 整体内容的准确性和完整性

请返回JSON格式的校验结果，包含以下字段：
- score: 0-1之间的校验分数
- comment: 简短的评语
- passed: 是否通过校验（score >= 0.7 为通过）

返回格式示例：
```json
{{
    "score": 0.85,
    "comment": "内容质量良好，描述准确",
    "passed": true
}}
```"""

    # 调用API
    try:
        client = OpenAI(
            api_key=MIMO_API_KEY,
            base_url=MIMO_BASE_URL
        )

        add_log(state, f"调用校验模型: {MIMO_MODEL}")

        # MiMo为纯文本模型，不发送图像
        messages = [
            {"role": "user", "content": prompt}
        ]

        response = client.chat.completions.create(
            model=MIMO_MODEL,
            messages=messages,
            max_tokens=1000,
            temperature=0.3
        )

        raw_response = response.choices[0].message.content
        add_log(state, f"校验模型响应成功，长度: {len(raw_response)}")

        # 解析响应
        verify_data = parse_verify_response(raw_response)

        result = {
            "verify_result": {
                "score": verify_data["score"],
                "comment": verify_data["comment"],
                "passed": verify_data["passed"],
                "raw_response": raw_response
            }
        }

        add_log(state, f"校验Agent完成，分数: {verify_data['score']:.2f}, 通过: {verify_data['passed']}")
        return result

    except Exception as e:
        add_log(state, f"校验模型调用失败，已自动启用模拟校验")
        score, comment = simulate_verify_score()
        return {
            "verify_result": {
                "score": score,
                "comment": f"[模拟校验] {comment}",
                "passed": score >= 0.7,
                "raw_response": f"[模拟校验] 分数: {score}, 评语: {comment}"
            }
        }
