---
prompt_id: MVCNET_MI_EEG_PROJECT
version: '3.4'
level: 3
parent: {prompt_id: MIEEG_AI4S_EXECUTION, version: '3.4'}
status: ACTIVE
---
# MVCNet 项目适配 v3.4

## 阅读与研究认识入口

研究认识与唯一来源索引见 [研究总览](research/CURRENT_UNDERSTANDING.md)。该总览指向基模阅读材料与本地实现依据；本 L3 不重复列举论文路径。

首次采用基模理解时，重点核对四视图与 CNN／Transformer 两分支、CE／CVC／CMC 的对应及 raw-only 推理仍保留两分支；VCN 的每视图 BN 状态隔离与训练／推理统计一致性分别判断。 已有同版本核对可复用，缺口只限制依赖它的判断。

新判定映射要区分完整方法相对 baseline、候选相对 VCN 支架的新增价值及机制控制；不得用前者收益覆盖后者失败，也不自动套用旧 VBR 的优效门。 实际判定路径的对应、接线缺口与最小验证按 [L2 第2节](../../prompts/LEVEL2_EXECUTION_v3.4.md#2-用现有-brief-完成一页决策)执行，共享细则不在本 L3 重复。

仅在唯一 Prompt Index 为本 Program 选定兼容的 v3.4 链后生效；已有 campaign、Run、operation 及冻结续接按原绑定处理。本文件仅保留模型适配，不保存运行状态、候选排名、最新效果或动态资源阈值。

## 1. 研究主线与宿主身份

研究优先级和从候选到首项结构验证的决策按 [L1 第1节](../../prompts/LEVEL1_RESEARCH_v3.4.md#1-选择能改变研究判断的行动) 与 [L2 第2节](../../prompts/LEVEL2_EXECUTION_v3.4.md#2-用现有-brief-完成一页决策) 执行。VCN-P1 支架、增量解释和必要比较仍按本 L3 下文的精确边界判断。

VCN-P1 可作为有范围限制的研究支架。其已有 utility／attribution 证据不自动证明新颖性、跨数据集有效性或足以成为论文主方法；判断回到 anchor、Evidence Index 及对应 Claim 的精确范围。后续候选分别说明自身增量、相对支架的增量和必要的容量／机制解释，不预设与 VCN-P1 互补。

VCN-P1 历史身份以 `VCN_P1_ANCHOR.md` 和源码为准：stem 的 BatchNorm1d 按 raw／flip／freq／cr 路由；P1 分离 running state 和 affine，N1 共享 running state、保留 view-specific affine；训练使用实际视图，验证／测试固定 raw 路径。原实现与原证据作为只读参照。新候选使用独立 AI4S 模块身份；若研究改变这些机制，须明确干预、比较对象和授权范围，不把变化后的实现沿称历史 P1，也不继承其结果。

## 2. 实际接口与必要对照

候选经 `MVCNetAdapter` 的 `model.args.ai4s_mvcnet_module` 接入 `src.models.ai4s.mvcnet` gateway，复用统一 runner。不要绕过 adapter 另建训练主体，也不要同时启用旧 `vcn_p1_variant` 与 AI4S 模块路径。仅沿本次改动涉及的路径核对以下事项，无关项不增加每轮前置门：

- **作用路径：**模块是否实际进入 forward、loss、gradient 或目标训练阶段？涉及 view、训练目标或读出时，核对实际视图路由、状态更新及验证／测试 raw 路径；配置声明或类注册不替代实际生效证据。
- **宿主与增量：**写清规范 baseline、VCN-P1 支架及候选的关系。增量关闭时应恢复所声明的宿主，并按需要验证 parity；依托支架的增量主比较应对 S0，控制臂上的优势不能代替该比较。整体方法对 B0 与增量对 S0 回答不同问题。
- **对照：**可用 B0（规范 baseline）、S0（支架）、N0（机制控制）、P0（完整候选）、C0（额外控制）；每轮按问题选用并写明未启用项，不强制五臂齐全。区分 `stage_id` 与 `arm_id`；历史 P1、N0、N1 等名称按原合同解释，不覆盖原语义。
- **归因：**声称排除容量解释时，选择与该解释相关的匹配控制，说明匹配参数量还是计算量及其局限。参数数目相同不等于表示能力相同；缺少相应证据时收窄主张。与容量无关的问题可不启用 C0。
- **可比性：**数据集、预处理、划分、训练 profile、选模和度量单位从精确配置读取；例如 `accuracy_pp` 不与比例值混用。不同训练阶段、目标、数据范围或适配版本的结果不因同名模块或同 seed 自动成为配对证据。

## 3. 研究认识与来源


历史证据是否覆盖新候选的预测，按 L2 第2、6节核对。相似路线须说明与已关闭干预的实质差异；VCN-P1 的有限支持、新模块的未证实假设、精确负结果和工程失败分别保留。技术失败不作为科学否定，局部否定不扩大到整个机制。

| 用途 | 精确入口 |
|---|---|
| 执行器与规范 baseline 配置 | `ai4s/programs/mvcnet/PROGRAM.yaml` |
| 活动身份与运行状态 | `ai4s/programs/mvcnet/research/STATE.yaml` |
| 证据与精确关闭边界 | `ai4s/programs/mvcnet/research/EVIDENCE_INDEX.yaml` |
| 统一运行入口／adapter | `scripts/run_mvcnet.py`、`src/baselines/mvcnet.py` |
| AI4S 模块与 gateway | `src/models/ai4s/mvcnet/` |
| VCN-P1 范围与原实现 | `ai4s/archive/programs/mvcnet/research/VCN_P1_ANCHOR.md`、`src/models/proposed/vcn_p1_mvcnet.py` |

以上是定位入口，不是每轮全部必读清单或有效性证明。先按证据用途和 exposure 选材料，再读取精确合同与必要直接引用；正式评估／Claim 按 `ai4s/prompts/EVALUATION_EVIDENCE.md`，普通探索和筛查不自动升级为正式 Claim。

## 4. 执行边界

字段接口按 runtime“字段接口与产物闭环”核验；VBR 曾因真实 capacity 的 `attempt` 与手工样例的 `attempts` 不同在 TECH 后中断，新路径使用明确调度来源并验证补写后的真实材料，不以手写汇总替代生产输出。旧 Run 保留原失败与复用权限。

新 Transaction 合同默认启用 `analysis_v1` 精简回传；冻结前按共享 runtime“新合同冻结前检查”运行 `validate-new`，明确实际分析依赖、包上限和核验超时，缺字段即停止准备并补齐。已有冻结 Run／Bundle 继续原规则，不补写字段或自动迁移。

部署、排队、资源策略、验收与工程恢复统一继承 L2 及 `ai4s/prompts/TRANSACTION_RUNTIME.md`，不在本 L3 复制调度参数或追加本模型专属门。资源画像对应实际模型、生产输入和完整训练进程；小 batch smoke 的通过不证明正式 batch 可运行。新包读取实际生成的资源策略摘要，异常再核对冻结配置和源码。

普通工程修复、累计成本、有效进展和资格通过后的授权衔接统一按 [L2 工程恢复](../../prompts/LEVEL2_EXECUTION_v3.4.md#5-工程恢复先保留成果再关闭实际缺口)。本 Program 区分源码／验收修复与 VCN 支架、新增量的科学失败；不因后置字段错误重训合格学生，也不以修复为名重开旧 NO_SIGNAL 实例。

本版激活只选择新工作的提示词链，不启动或补跑实验。旧 Run 的科学配置、门、预算、停止与恢复规则保持原冻结绑定；新工作按 Prompt Index 选定的兼容链处理，冻结工作沿精确历史绑定继续。工程恢复复用已完成且合格的训练产物，按原合同验收，不用新规则重释旧失败或重开关闭路线。



