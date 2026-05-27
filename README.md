# 多Agent电商商品多模态智能解析系统

基于 LangGraph 的多智能体协作系统，实现电商商品图像的智能理解、创意文案生成与内容质量校验。

## 架构总览
<img width="1334" height="938" alt="微信图片_20260528004437_65_360" src="https://github.com/user-attachments/assets/e58ae0d2-6c4b-4730-8f1e-c3e147b08d43" />

```
用户输入（图像 + 文本）
        │
        ▼
┌───────────────────┐
│   主 Agent         │  DeepSeek-V4-Pro
│   任务规划 & 路由   │
└───────┬───────────┘
        │ 依次调用（子Agent间严格隔离）
  ┌─────┼─────┐
  ▼     ▼     ▼
┌────┐┌────┐┌────┐
│视觉││文案││校验│
│Agent││Agent││Agent│
├────┤├────┤├────┤
│Qwen││Qwen││MiMo│
│VL+ ││3.6-││V2.5│
│    ││Plus││Pro │
└──┬─┘└──┬─┘└──┬─┘
   │     │     │
   └─────┼─────┘
         ▼
┌───────────────────┐
│   主 Agent         │
│   结果聚合 & 重试   │
└───────┬───────────┘
        ▼
      最终输出
```

## 功能特性

- **多Agent协作**：主Agent负责任务规划与分发，三个子Agent各司其职、严格隔离
- **置信度机制**：支持模型提供、最小值、加权平均、仅校验四种置信度计算模式
- **智能重试**：综合置信度低于阈值时自动重试失败的Agent（最多3次）
- **消融实验**：可独立开关任意子Agent，提供多组预定义消融配置
- **LangSmith追踪**：全链路可观测，便于调试与实验分析
- **评测系统**：内置基准测试框架，支持准确率、幻觉率、响应时间、Token消耗等多维指标

## 快速开始

### 环境要求

- Python 3.9+
- API密钥（DeepSeek、Qwen/DashScope、MiMo）

### 安装

```bash
git clone https://github.com/uiimmz/multi-agent.git
cd multi-agent
pip install -r requirements.txt
```

### 配置API密钥

在项目根目录创建 `apikey` 文件，按以下格式填入密钥：

```
deepseek apikey
<你的DeepSeek密钥>
qwen apikey
<你的Qwen/DashScope密钥>
MiMo apikey
<你的MiMo密钥>
langsmith apikey
<你的LangSmith密钥>
```

### 运行

**命令行单次调用：**

```bash
python main.py --image path/to/image.jpg --query "这件衣服是什么颜色？"
```

**Streamlit交互界面：**

```bash
streamlit run "app(streamlit).py"
```

**基准测试：**

```bash
# 测试50条（完整系统）
python benchmark.py -n 50

# 测试20条（仅视觉Agent）
python benchmark.py -n 20 -c vision_only

# 中断后恢复
python benchmark.py -n 50 --resume
```

## 项目结构

```
├── agents/                  # Agent实现
│   ├── main_agent.py        # 主Agent（规划/路由/聚合）
│   ├── vision_agent.py      # 视觉理解Agent
│   ├── copy_agent.py        # 文案生成Agent
│   └── verify_agent.py      # 内容校验Agent
├── config.py                # API配置、阈值配置、消融配置
├── graph.py                 # LangGraph图构建
├── state.py                 # 状态定义（TypedDict）
├── utils.py                 # 工具函数
├── main.py                  # 命令行入口
├── app(streamlit).py        # Streamlit Web界面
├── benchmark.py             # 基准测试与评测
├── test_dataset/            # 测试数据集（1000条）
│   ├── dataset.jsonl
│   └── images/
├── requirements.txt
└── apikey                   # API密钥（不纳入版本控制）
```

## 消融实验

通过修改 `ABLATION_CONFIG` 可独立开关各子Agent：

| 预设 | 视觉 | 文案 | 校验 | 置信度模式 |
|------|:----:|:----:|:----:|-----------|
| `full` | ✓ | ✓ | ✓ | weighted |
| `no_copy` | ✓ | ✗ | ✓ | weighted |
| `no_verify` | ✓ | ✓ | ✗ | min |
| `vision_only` | ✓ | ✗ | ✗ | model |

## 评测指标

- **综合准确率**：基于规则的真值比对（品类/颜色/风格）
- **幻觉率**：模型输出与真值矛盾的比例
- **平均响应时间**：端到端推理耗时
- **Token消耗量**：所有Agent的Token使用汇总

## 许可证

仅供学习与研究使用。
