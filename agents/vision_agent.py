"""
视觉理解Agent
使用qwen3-vl-plus模型进行图像理解和描述
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Dict, Any
from openai import OpenAI

from config import QWEN_BASE_URL, QWEN_API_KEY, QWEN_VL_MODEL
from utils import image_to_base64, get_image_media_type, extract_confidence_from_text, add_log


def vision_agent(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    视觉理解Agent
    接收图像路径和用户查询，返回视觉描述和置信度

    Args:
        state: 当前状态字典

    Returns:
        更新后的状态字典（vision_result字段）
    """
    add_log(state, ">>> 视觉Agent开始执行")

    # 检查消融开关
    if not state.get("ablation_config", {}).get("enable_vision", True):
        add_log(state, "视觉Agent已被消融，跳过执行")
        return {
            "vision_result": {
                "description": "[消融] 视觉Agent已禁用",
                "confidence": 0.7,
                "raw_response": ""
            },
            "ablation_tags": state.get("ablation_tags", []) + ["no_vision"]
        }

    image_path = state.get("image_path", "")
    user_query = state.get("user_query", "")

    # 读取图像并转换为base64
    try:
        image_base64 = image_to_base64(image_path)
        media_type = get_image_media_type(image_path)
        add_log(state, f"图像读取成功: {image_path}")
    except Exception as e:
        add_log(state, f"图像读取失败: {str(e)}")
        return {
            "vision_result": {
                "description": f"图像读取失败: {str(e)}",
                "confidence": 0.0,
                "raw_response": ""
            }
        }

    # 构造提示词
    prompt = f"""请仔细观察这张图像，并结合用户的查询进行分析。

用户查询: {user_query}

请完成以下任务：
1. 详细描述图像中的主要内容，包括物体、场景、颜色、文字等关键信息
2. 如果用户查询是关于图像中特定内容的问题，请针对性回答
3. 在回答末尾，以"置信度: X.XX"的格式给出你对描述准确性的置信度（0-1之间的数字）

请用中文回答。"""

    # 构造OpenAI Vision API格式的消息
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{media_type};base64,{image_base64}"
                    }
                },
                {
                    "type": "text",
                    "text": prompt
                }
            ]
        }
    ]

    # 调用API
    try:
        client = OpenAI(
            api_key=QWEN_API_KEY,
            base_url=QWEN_BASE_URL
        )

        add_log(state, f"调用视觉模型: {QWEN_VL_MODEL}")

        response = client.chat.completions.create(
            model=QWEN_VL_MODEL,
            messages=messages,
            max_tokens=2000,
            temperature=0.7
        )

        raw_response = response.choices[0].message.content
        add_log(state, f"视觉模型响应成功，长度: {len(raw_response)}")

        # 解析响应，提取描述和置信度
        confidence = extract_confidence_from_text(raw_response, default=0.7)

        # 移除末尾的置信度行，得到纯描述
        description = raw_response
        lines = description.strip().split('\n')
        if lines and '置信度' in lines[-1]:
            description = '\n'.join(lines[:-1]).strip()

        result = {
            "vision_result": {
                "description": description,
                "confidence": confidence,
                "raw_response": raw_response
            },
            "image_base64": image_base64  # 保存base64供后续Agent使用
        }

        add_log(state, f"视觉Agent完成，置信度: {confidence:.2f}")
        return result

    except Exception as e:
        add_log(state, f"视觉模型调用失败: {str(e)}")
        return {
            "vision_result": {
                "description": f"视觉模型调用失败: {str(e)}",
                "confidence": 0.0,
                "raw_response": ""
            },
            "image_base64": image_base64
        }
