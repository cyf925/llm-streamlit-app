# LLMTruthProject：多模型可信聚合与零知识证明系统

## 1. 项目简介

本项目是一个基于多个本地大语言模型的可信聚合系统。系统的核心目标是：当多个大模型对同一个任务给出不同回答时，不宜采用某一个模型的结果，而是先将不同模型的回答转换为统一的结构化事实，然后使用 TruthFinder 风格的真值发现算法进行可信聚合，最终得到更稳定、更可信、更容易解释的综合结果。

在此基础上，项目进一步引入 Groth16 零知识证明，用于证明系统在给定输入后，确实按照预先定义好的聚合规则完成了 TruthFinder 计算过程。

需要注意的是，零知识证明并不证明模型回答本身一定正确，也不证明归一化过程一定完全正确。它证明的是：给定已经数值化的输入之后，系统确实按照预设电路执行了聚合计算。

---

## 2. 项目背景与设计动机

大语言模型在翻译、医疗风险提示、法律条文解答、金融风险评估等任务中都可以给出较有价值的判断。但是，不同模型之间经常会出现以下问题：

1. 多个模型回答不完全一致；
2. 有些模型回答较保守，有些模型回答较激进；
3. 有些模型会遗漏关键信息；
4. 有些模型输出格式不稳定；
5. 单个模型的输出不一定足够可靠；
6. 用户很难判断应该相信哪个模型。

因此，本项目不把大语言模型当成唯一权威，而是把多个模型视为多个信息源。系统通过统一归一化、候选事实抽取、真值发现聚合、模型可信度更新等步骤，综合多个模型的判断结果。

项目的核心思想可以概括为：

```text
多个本地 LLM 输出
→ 场景归一化
→ 标准 object / fact 表示
→ TruthFinder 可信聚合
→ 前端解释展示
→ Groth16 证明聚合过程
```

---

## 3. 当前已实现的场景

当前项目主要包含两个场景：

1. 英文翻译推荐场景；
2. 医疗就医前风险提示场景。

后续可以继续扩展到更多结构化判断场景，例如：

- 法律问题解答；
- 金融风险评估；

---

## 4. 总体系统架构

项目整体采用“场景适配层 + TruthFinder 核心聚合层 + 前端展示层 + ZK 证明层”的结构。

### 4.1 场景适配层

场景适配层负责把不同模型的原始输出转换为统一格式。

不同场景下，用户输入和模型输出的内容可能完全不同。例如：

- 翻译场景中，用户输入英文句子；
- 医疗场景中，用户输入健康状况描述；

但是进入 TruthFinder 之前，所有场景都需要统一抽象为：

```text
object：当前任务中需要判断的问题；
fact：某个模型对该 object 给出的候选答案。
```

### 4.2 TruthFinder 核心聚合层

TruthFinder 层不直接处理自然语言长文本，而是处理结构化 facts。

它会根据多个模型的支持情况、模型之间的依赖关系、fact 之间的支持或冲突关系，迭代计算：

- 每个模型的可信度；
- 每个候选 fact 的置信度。

### 4.3 前端展示层

前端用于展示：

1. 用户输入区域；
2. 四个模型的原始或用户可读回答；
3. 不同 object 下的结构化判断；
4. 归一化结果；
5. TruthFinder 聚合结果；
6. 模型可信度；
7. 最终综合建议；
8. ZK 证明相关信息。

### 4.4 ZK 证明层

ZK 证明层用于证明：

```text
给定数值化输入后，系统确实按照预设 Q16 定点 TruthFinder 电路执行了固定轮数的聚合计算。
```

ZK 证明层不证明：

- 模型回答一定正确；
- 归一化结果一定正确；
- 医疗建议一定真实可靠；
- 翻译结果一定符合人工标准。

ZK 证明层只证明：

- 聚合计算过程没有被随意篡改；
- 公开输出确实由给定输入和固定电路计算得到。

---

## 5. 项目目录结构

当前项目目录结构如下：

```text
LLMTruthProject/
│
├── main_app.py
│
├── medical_app/
│   ├── __init__.py
│   ├── app_medical.py
│   ├── normalize_medical.py
│   ├── medical_truthfinder.py
│   ├── test_normalize_medical.py
│   └── test_medical_truthfinder.py
│
├── translation_app/
│   ├── app.py
│   ├── normalize.py
│   ├── TruthFinder.py
│   ├── circuit_spec.json
│   └── zk/
│       ├── TruthFinder_circuit_ref.py
│       ├── circuits/
│       │   └── truthfinder.circom
│       ├── expander.py
│       ├── prepare_circom_input.py
│       ├── runtime_input_builder.py
│       ├── truthfinder_runtime_input_schema.json
│       └── zk_input_builder.py
│
├── package.json
├── package-lock.json
├── .gitignore
└── README.md
```

---

## 6. 主要文件说明

### 6.1 `main_app.py`

项目主入口文件，用于统一进入不同场景。后续如果要集成多个系统，可以在这里做统一导航。

### 6.2 `medical_app/app_medical.py`

医疗场景前端页面。主要负责：

- 接收用户健康描述；
- 调用模型或 mock 数据；
- 展示模型用户可读回答；
- 展示结构化六 object 结果；
- 调用归一化模块；
- 调用医疗场景 TruthFinder；
- 展示可信聚合结果；
- 展示最终建议和免责声明。

### 6.3 `medical_app/normalize_medical.py`

医疗场景归一化代码。它是医疗场景中非常关键的一层，负责把模型输出和用户文本补丁转换为标准 facts。

主要功能包括：

1. 定义医疗场景的六个 object；
2. 定义每个 object 的标准候选 facts；
3. 处理模型结构化字段；
4. 处理用户原文补丁；
5. 处理 fallback；
6. 处理条件句；
7. 处理否定表达；
8. 处理高热数值识别；
9. 处理安全兜底规则；
10. 构造 TruthFinder 输入需要的事实表。

### 6.4 `medical_app/medical_truthfinder.py`

医疗场景 TruthFinder 核心代码。负责对归一化后的医疗 facts 进行可信聚合。

主要功能包括：

1. 构造候选 fact 集合；
2. 构造模型支持矩阵；
3. 支持单选 object 和多选 object；
4. 支持 `multi` 模式；
5. 支持 `zk_top1` 模式；
6. 计算模型之间的依赖关系；
7. 计算模型覆盖率；
8. 计算 effective trust；
9. 计算 fact 置信度；
10. 计算模型可信度；
11. 输出前端可解释结果；
12. 构造 ZK-friendly payload。

### 6.5 `medical_app/test_normalize_medical.py`

医疗归一化测试文件。用于验证：

- 条件句处理；
- 高热数值识别；
- 泌尿、皮肤、眼科、妇产、儿科等场景归一化；
- 安全兜底规则；
- fallback 逻辑；
- 轻症不误判急诊。

### 6.6 `medical_app/test_medical_truthfinder.py`

医疗 TruthFinder 测试文件。用于验证：

- `multi` 模式支持多选 facts；
- `zk_top1` 模式只选择 top1；
- 模型依赖关系；
- model coverage；
- effective trust；
- watch facts；
- 信息不足弱冲突；
- ZK-friendly payload 结构。

### 6.7 `translation_app/`

翻译场景代码目录。包含翻译场景前端、归一化、TruthFinder 和 ZK 输入构造代码。

### 6.8 `translation_app/zk/`

翻译场景零知识证明相关代码目录。包含 circom 电路、输入构造、Q16 参考实现等文件。

---

## 7. TruthFinder 抽象说明

本项目中的 TruthFinder 抽象来源于真值发现思想。

系统把每个大模型视为一个 source，把每个需要判断的问题视为 object，把模型对该问题给出的候选答案视为 fact。

| 概念 | 含义 |
|---|---|
| source | 一个本地大模型 |
| object | 当前任务中需要判断的问题 |
| fact | 某个模型给出的候选判断 |
| trust | 模型可信度 |
| confidence | fact 置信度 |

系统迭代更新：

1. 如果一个 fact 被多个高可信模型支持，则该 fact 置信度更高；
2. 如果一个模型经常支持高置信度 fact，则该模型可信度更高；
3. 如果多个模型过于相似，则它们的支持不能被完全当作独立证据；
4. 如果 facts 之间存在冲突关系，则冲突 fact 会互相抑制；
5. 如果 facts 之间存在支持关系，则支持 fact 会互相增强。

---

## 8. 医疗场景设计说明

医疗场景当前采用六个 object：

1. 是否存在危险信号；
2. 整体紧急程度；
3. 可能原因方向；
4. 关键风险信号；
5. 支持低风险或其他解释的因素；
6. 建议咨询科室。

这六个 object 的设计目标不是完成医学诊断，而是完成“就医前风险提示”。

### 8.1 医疗场景输入

用户输入一段自然语言健康描述，例如：

```text
最近两天胸闷，爬楼梯时明显，休息后缓解，偶尔心慌，没有发烧咳嗽。
```

系统调用四个本地大模型，让每个模型输出两层内容：

1. 用户可读分析；
2. 结构化六 object 判断。

### 8.2 医疗场景模型输出

每个模型应尽量输出类似结构：

```json
{
  "user_explanation": "从描述看，胸闷与活动相关，需要关注心血管方向风险。",
  "structured_analysis": {
    "danger_signal": "可能存在危险信号",
    "urgency_level": "尽快线下就医",
    "possible_cause": ["心血管相关"],
    "risk_signal": ["胸闷", "活动后加重", "心跳快或心慌"],
    "low_risk_factor": ["无发热"],
    "consult_department": ["心内科"]
  }
}
```

### 8.3 医疗场景归一化

模型的自然语言表达可能多种多样。例如：

- “喘不上气”；
- “呼吸困难”；
- “气不够用”；
- “胸口发紧”。

归一化层会尽量把这些不同表达映射到标准 fact，例如：

```text
气短
```

归一化层还会处理：

- 条件句；
- 否定表达；
- 高热数值；
- 儿童场景；
- 孕期风险；
- 严重过敏；
- 视力突然下降；
- 中毒风险；
- 泌尿红旗。

### 8.4 医疗场景 TruthFinder

TruthFinder 不直接聚合用户原文，而是聚合模型结构化字段中的标准 facts。

默认情况下，TruthFinder 主输入使用：

```text
source = from_model_fields
exclude_fallbacks = True
```

这表示：

- 只把模型结构化字段作为模型证据；
- 不把 fallback 当成模型真实判断；
- 不把 safety_overrides 当成模型支持；
- 不把 user_text_patch_facts 当成第五个 source。

这样可以保证 TruthFinder 的 source 语义干净。

### 8.5 医疗场景安全规则

医疗场景中，安全规则非常重要。

但是安全规则不进入 TruthFinder support。它们只用于最终解释和前端展示。

例如：

- 严重过敏反应；
- 药物过量或中毒；
- 突然一只眼看不见；
- 孕期出血或腹痛；
- 宝宝高烧且精神差；
- 严重出血。

这些情况即使模型之间没有完全一致，也应该在最终提示中被保守处理。

---

## 9. 翻译场景设计说明

翻译场景中：

- object 是用户选择或系统抽取出的英文关键词；
- fact 是某个模型给出的中文候选释义；
- source 是四个本地大模型。

翻译场景流程：

```text
英文句子输入
→ 关键词抽取
→ 四个模型分别翻译
→ 提取每个关键词的中文释义
→ 归一化候选释义
→ TruthFinder 聚合
→ 输出可信关键词释义
→ 下游整句翻译排序
```

翻译场景下的 ZK 证明主要证明 TruthFinder 聚合计算过程，而不是证明翻译语义一定正确。

---

## 10. 零知识证明设计说明

项目中使用 Groth16 / Circom 风格的零知识证明方案。

当前电路核心思想是：

1. 所有输入先在电路外归一化；
2. 候选 facts、imp/conf、dep_avg 等在电路外构造；
3. 电路内部只执行固定参数、固定轮数的 TruthFinder Q16 定点计算；
4. 电路输出公开结果；
5. verifier 验证 proof 是否匹配 public output。

### 10.1 ZK 证明能够证明什么

ZK 证明可以证明：

```text
给定输入后，系统确实按照预设电路完成了 TruthFinder 聚合计算。
```

### 10.2 ZK 证明不能证明什么

ZK 证明不能证明：

1. 模型回答一定正确；
2. 模型一定没有幻觉；
3. 医疗建议一定正确；
4. 翻译一定符合人工标准；
5. 归一化一定完全准确；
6. safety_overrides 一定完全合理。

### 10.3 为什么仍然需要 ZK

ZK 的价值在于证明系统没有随意修改聚合结果。

也就是说，用户或评审可以验证：

```text
这个结果确实是由公开承诺的算法和输入计算得到的。
```

---

## 11. 环境部署说明

### 11.1 推荐环境

建议使用：

- Windows 10 / Windows 11；
- Python 3.11 或更高；
- Git；
- VS Code 或 PyCharm；
- Streamlit；
- pytest；
- Ollama；
- Node.js；
- snarkjs；
- circom。

普通前端开发和 TruthFinder 算法开发不一定需要安装 ZK 工具链。

### 11.2 Python 依赖

常用依赖包括：

```text
streamlit
requests
pandas
numpy
pytest
openpyxl
```

安装命令：

```bash
python -m pip install streamlit requests pandas numpy pytest openpyxl
```

### 11.3 Node.js 依赖

如果需要使用 snarkjs 或相关 JS 工具，可以运行：

```bash
npm install
```

项目中的 `package.json` 和 `package-lock.json` 用于记录 Node.js 依赖信息。

注意：

```text
node_modules/ 不应上传到 GitHub。
```

---

## 12. Ollama 模型说明

项目设计上支持调用本地 Ollama 模型。

常用模型示例：

```text
qwen2.5:7b-instruct-q4_K_M
mistral:7b-instruct-v0.3-q5_0
gemma2:9b-instruct-q4_K_M
koesn/mistral-7b-instruct:Q4_0
```

Ollama 默认服务地址：

```text
http://localhost:11434
```

模型调用通常使用：

```text
/api/chat
/api/generate
```

普通组员开发前端或 TruthFinder 算法时，可以先使用 mock 数据，不需要每个人都下载大模型。

---

## 13. 运行方式

### 13.1 运行主入口

```bash
streamlit run main_app.py
```

### 13.2 运行医疗场景

```bash
streamlit run medical_app/app_medical.py
```

### 13.3 运行翻译场景

```bash
streamlit run translation_app/app.py
```

---

## 14. 测试方式

### 14.1 运行医疗归一化测试

```bash
python -m pytest medical_app/test_normalize_medical.py
```

### 14.2 运行医疗 TruthFinder 测试

```bash
python -m pytest medical_app/test_medical_truthfinder.py
```

### 14.3 运行全部测试

```bash
python -m pytest
```

如果修改了归一化或 TruthFinder 代码，必须优先运行对应测试。

---

## 15. 组员开发说明

### 15.1 前端组员

前端组员主要关注：

1. 页面布局；
2. 输入框；
3. 模型输出展示；
4. 归一化结果展示；
5. TruthFinder 结果展示；
6. 最终建议展示；
7. 医疗免责声明；
8. 多场景导航风格统一。

前端组员可以先使用 mock 数据，不需要一开始就配置 Ollama 或 ZK。

### 15.2 新场景 TruthFinder 组员

如果要新增其他场景，建议参考医疗场景结构。

一个新场景通常需要：

```text
scene_app/
├── app_scene.py
├── normalize_scene.py
├── scene_truthfinder.py
└── test_scene_truthfinder.py
```

新场景可以修改：

- object 定义；
- fact 候选集合；
- 归一化函数；
- 同义词词典；
- 冲突关系词典；
- 前端展示文案。

不建议随意修改：

- TruthFinder 核心迭代公式；
- tau 计算；
- sigmoid 计算；
- support 语义；
- dep_avg 语义；
- trust prior 语义；
- ZK 电路参数。

### 15.3 ZK 组员

ZK 组员主要关注：

1. circom 电路；
2. Q16 定点输入；
3. witness 生成；
4. Groth16 setup；
5. proof 生成；
6. public.json 解析；
7. verification key；
8. 前端证明状态展示。

ZK 文件通常较大，很多生成物不应上传 GitHub。

---

## 16. Git 与协作规范

### 16.1 不应上传的内容

以下内容不应上传 GitHub：

```text
__pycache__/
.pytest_cache/
node_modules/
models/
*.ptau
*.zkey
*.wtns
*.r1cs
*.sym
proof.json
public.json
verification_key.json
```

这些内容已经在 `.gitignore` 中排除。

### 16.2 推荐提交前检查

提交前建议运行：

```bash
git status
```

确认没有上传缓存、大模型文件、ZK 大型生成文件。

### 16.3 提交命令示例

```bash
git add .
git commit -m "说明本次修改内容"
git push
```

---

## 17. 医疗免责声明

医疗场景仅用于多模型意见聚合和就医前风险提示，不构成医学诊断，也不能替代医生、医院检查或专业医疗建议。

如果用户出现以下情况，应优先线下就医或急诊评估：

- 胸痛；
- 气短；
- 意识改变；
- 严重出血；
- 抽搐；
- 突发无力或言语不清；
- 高热伴精神差；
- 严重过敏反应；
- 喉头紧缩或面唇肿胀；
- 孕期出血或腹痛；
- 药物过量或中毒风险；
- 视力突然下降；
- 儿童精神差或反应差；
- 其他持续加重或明显异常情况。

---

## 18. 当前项目定位

本项目不是一个简单的 Chatbot，也不是一个单模型问答系统。

它的定位是：

```text
多模型输出
→ 统一归一化
→ TruthFinder 可信聚合
→ 前端解释
→ ZK 证明计算过程
```

项目重点不在于让某一个模型回答得最好，而在于构建一个可以对多个模型的结构化判断进行统一比较、聚合、解释和证明的系统框架。

---

## 19. 后续开发方向

后续可以继续完善：

1. 医疗场景前端；
2. 法律、金融等场景；
3. 其他安全场景；
4. 多场景统一入口；
5. 医疗场景 ZK 输入构造；
6. 医疗场景 Q16 reference；
7. 医疗场景 circom 电路；
8. 前端 ZK 证明展示；
9. 更完整的测试集；
10. 更规范的项目文档。

---

## 20. 简要总结

本项目当前已经具备以下能力：

1. 支持多个本地大模型；
2. 支持翻译场景可信聚合；
3. 支持医疗就医前风险提示场景；
4. 支持医疗场景归一化；
5. 支持医疗场景 TruthFinder 聚合；
6. 支持模型可信度计算；
7. 支持高风险候选提示；
8. 支持 Groth16 / Circom 证明思路；
9. 支持前端展示；
10. 支持继续扩展新场景。

项目后续的核心任务是继续完善前端、多场景集成和 ZK 证明链路。
