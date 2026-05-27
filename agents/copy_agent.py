"""
文案生成Agent
使用qwen3.6-plus模型生成创意文案
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Dict, Any
from openai import OpenAI

from config import QWEN_BASE_URL, QWEN_API_KEY, QWEN_TEXT_MODEL
from utils import extract_confidence_from_text, add_log


def copy_agent(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    文案生成Agent
    基于视觉描述和用户查询生成创意文案

    Args:
        state: 当前状态字典

    Returns:
        更新后的状态字典（copy_result字段）
    """
    add_log(state, ">>> 文案Agent开始执行")

    # 检查消融开关
    if not state.get("ablation_config", {}).get("enable_copy", True):
        add_log(state, "文案Agent已被消融，跳过执行")
        return {
            "copy_result": {
                "copywriting": "[消融] 文案Agent已禁用",
                "confidence": 0.7,
                "raw_response": ""
            },
            "ablation_tags": state.get("ablation_tags", []) + ["no_copy"]
        }

    # 获取视觉描述
    vision_result = state.get("vision_result", {})
    vision_description = vision_result.get("description", "")
    user_query = state.get("user_query", "")
    task_type = state.get("task_type", "describe")

    if not vision_description:
        add_log(state, "警告: 视觉描述为空，使用默认描述")
        vision_description = "无法获取视觉描述"

    # 构造提示词
    if task_type == "qa":
        prompt = f"""基于以下视觉描述和用户问题，请生成一个准确、详细的回答。

视觉描述:
{vision_description}

用户问题: {user_query}

要求:
1. 回答要准确、有条理
2. 基于视觉描述中的信息进行回答
3. 如果信息不足，请说明推测依据
4. 在回答末尾，以"置信度: X.XX"的格式给出你对回答准确性的置信度（0-1之间的数字）

请用中文回答。"""
    else:  # describe
        prompt = f"""基于以下视觉描述，请生成一段吸引人的商品文案。

视觉描述:
{vision_description}

用户需求: {user_query}

要求:
1. 文案要生动、有吸引力
2. 突出商品的特点和优势
3. 语言流畅，适合电商场景
4. 在文案末尾，以"置信度: X.XX"的格式给出你对文案质量的置信度（0-1之间的数字）

请用中文回答。"""

    # 调用API
    try:
        client = OpenAI(
            api_key=QWEN_API_KEY,
            base_url=QWEN_BASE_URL
        )

        add_log(state, f"调用文案模型: {QWEN_TEXT_MODEL}")

        response = client.chat.completions.create(
            model=QWEN_TEXT_MODEL,
            messages=[
                {"role": "system", "content": "你是一位专业的电商文案撰写专家，擅长撰写吸引人的商品描述和创意文案。"},
                {"role": "user", "content": prompt}
            ],
            max_tokens=2000,
            temperature=0.8
        )

        raw_response = response.choices[0].message.content
        add_log(state, f"文案模型响应成功，长度: {len(raw_response)}")

        # 解析响应，提取文案和置信度
        confidence = extract_confidence_from_text(raw_response, default=0.7)

        # 移除末尾的置信度行，得到纯文案
        copywriting = raw_response
        lines = copywriting.strip().split('\n')
        if lines and '置信度' in lines[-1]:
            copywriting = '\n'.join(lines[:-1]).strip()

        result = {
            "copy_result": {
                "copywriting": copywriting,
                "confidence": confidence,
                "raw_response": raw_response
            }
        }

        add_log(state, f"文案Agent完成，置信度: {confidence:.2f}")
        return result

    except Exception as e:
        add_log(state, f"文案模型调用失败: {str(e)}")
        return {
            "copy_result": {
                "copywriting": f"文案模型调用失败: {str(e)}",
                "confidence": 0.0,
                "raw_response": ""
            }
        }
