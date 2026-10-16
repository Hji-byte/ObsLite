# ObsLite: Efficient Agent Training with Selective Observation Masking

[English](README.md) | 简体中文

**ObsLite**通过选择性遮蔽旧的环境反馈，减少智能体训练中的重复输入。

项目使用WebShop环境，通过Qwen3.5-27B生成的1,418条教师轨迹对Qwen3.5-0.8B进行SFT，而后将ObsLite引入GRPO训练和评测：保留初始、当前及其之前最近四条observation，以及包含`Invalid action!`或`Invalid format.`的错误反馈，将其余旧observation替换成短占位符，同时保留全部历史Thought和Action。该策略缩短了模型读取的上下文，相比保留完整历史的GRPO，ObsLite正确率从41.2%提升至41.6%，同时平均总token减少约36.8%，训练耗时缩短约20.6%。



## 实验结果

实验使用单张NVIDIA RTX 6000D 84GB GPU，分别验证ObsLite直接用于SFT模型评测，以及用于GRPO训练与评测时的效果。

### SFT模型上的对比

使用Qwen3.5-27B生成的1,418条教师轨迹，对Qwen3.5-0.8B进行QLoRA监督微调。以下三个方案共用该SFT模型，仅改变评测上下文。其中两步O/A窗口保留最近两步的Observation和Action及当前Observation。

ObsLite的成功率和平均奖励最高，成功率由完整历史的35.8%提升至38.2%，同时平均总token减少约41.1%。两步O/A窗口虽然token更少，但成功率降至29.8%。

| 方案 | 成功率 | 平均奖励 | 平均交互步数 | 平均总token |
| --- | ---: | ---: | ---: | ---: |
| Qwen3.5-0.8B | 7.8% | 0.13218 | 16.578 | 139,606 |
| SFT完整历史 | 35.8% | 0.58315 | 8.172 | 47,791 |
| SFT＋ObsLite | **38.2%** | **0.60425** | **7.830** | 28,155 |
| SFT两步O/A窗口 | 29.8% | 0.48698 | 10.210 | **17,443** |

### GRPO训练中使用ObsLite

两组实验均以同一SFT模型为起点，采用LoRA进行20轮GRPO训练。训练时使用vLLM生成交互轨迹，每题采样8条，最多同时生成32条。

GRPO基线使用完整历史；ObsLite在训练过程中只保留部分历史环境反馈，让模型读取更短的上下文。ObsLite的成功率为41.6%，略高于完整历史的41.2%，平均奖励也略高；平均总token减少约36.8%，训练耗时缩短约20.6%。

| 方案 | 成功率 | 平均奖励 | 平均交互步数 | 平均总token | RL训练时长（小时） |
| --- | ---: | ---: | ---: | ---: | ---: |
| GRPO | 41.2% | 0.62943 | 7.700 | 40,485 | 14.78 |
| ObsLite＋GRPO | **41.6%** | **0.63695** | **7.542** | **25,571** | **11.74** |

逐轮耗时见[20轮训练时间表](results/TRAINING_TIME.md)，逐题与逐步记录在[results/episodes](results/episodes)。



## 工程优化

为降低训练和运行开销，采用了以下工程优化：

| 优化 | 作用 |
|---|---|
| 输入padding裁剪 | 计算前去掉输入左侧的无效填充，保留全部有效内容，减少空白计算。 |
| 按需计算logits | 只计算模型生成的Thought和Action所需的词表分数，减少计算和显存占用。 |
| FLA内核加速 | 使用FLA加速Qwen3.5的Gated DeltaNet计算，actor/reference采用FP32。 |
| 商品数据缓存 | 缓存WebShop商品数据的清洗结果，减少重复预处理开销。 |

在相同FP32精度下，使用FLA后，模型前向和反向计算的耗时减少约71%～80%。



## 目录结构

```text
configs/grpo.yaml       GRPO共用默认配置
scripts/                SFT、WebShop接入、上下文策略与运行支持
tools/                  训练准备、续训、评测和结果统计
requirements/           模型与WebShop依赖
data/
  teacher_sft.jsonl     SFT使用的1,418条教师轨迹
  train_tasks.jsonl     固定的1,024道RL训练题
  test_tasks.jsonl      500道评测题
results/                逐题结果和训练耗时表
```



## 安装

在已安装NVIDIA驱动、Conda和Git的Linux服务器上运行：

```bash
bash setup.sh
```

脚本安装依赖、下载verl-agent和ETO封装的WebShop环境，并自动下载、解压商品数据和搜索索引。



## 训练

需自行下载Qwen3.5-0.8B模型。SFT教师轨迹及训练题目已包含在仓库中，WebShop数据由安装脚本准备。

以下命令以基础模型保存在`models/Qwen3.5-0.8B`为例，请替换为实际路径。

**第一步：SFT。** 用教师轨迹训练基础模型，并保存到`models/sft-export`：

```bash
bash run.sh sft models/Qwen3.5-0.8B
```

**第二步：ObsLite GRPO。** 从上一步训练得到的SFT模型出发，在GRPO训练过程中使用选择性观察遮蔽，共训练20轮，每2轮保存一次：

```bash
bash run.sh train models/Qwen3.5-0.8B
```

程序会自动使用教师数据进行FP32/FLA计算检查，通过后开始训练，checkpoint保存在`runs/mask4-grpo20/checkpoints/`。



## 评测

下面以第20轮模型为例进行评测：

```bash
# 第20轮训练得到的模型
MODEL=runs/mask4-grpo20/checkpoints/global_step_20/actor/lora_adapter

# 评测500题，结果保存到 runs/eval-obslite
bash run.sh evaluate "$MODEL" runs/eval-obslite
```




## 致谢

本项目基于以下开源项目构建：

- [verl-agent](https://github.com/langfengQ/verl-agent)：提供智能体GRPO训练框架。
- [ETO](https://github.com/Yifan-Song793/ETO)：提供本项目使用的WebShop环境封装。
- [WebShop](https://github.com/princeton-nlp/WebShop)：提供模拟购物环境、商品数据和任务奖励。

感谢相关项目的开发者与研究者。
