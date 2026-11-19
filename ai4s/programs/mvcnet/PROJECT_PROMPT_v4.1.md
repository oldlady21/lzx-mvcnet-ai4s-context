---
prompt_id: MVCNET_MI_EEG_PROJECT
version: '4.1'
level: 3
parent: {prompt_id: MIEEG_AI4S_EXECUTION, version: '4.0'}
status: ACTIVE
---
# MVCNet 项目适配 v4.1

## 阅读与研究认识入口

本适配承接 MVCNet L3 v4.0 的章节与职责。共享科研与执行规则见 L1/L2 v4.0。旧工作按原绑定处理。

研究认识与唯一来源索引见 [基模设计先验](research/CURRENT_UNDERSTANDING.md#基模设计先验)。该总览指向基模阅读材料与本地实现依据；本 L3 不重复列举论文路径。

首次采用基模理解时，区分论文 v4 的分类结构与本地实现。论文将 CNN 与 Transformer 特征拼接分类，测试时输入未增强试次。本地规范 MVCNet（`MVCNetAdapter`）、已核对的 VCN-P1 和 VBR-R1 分类走 raw → CNN/IFNet → 分类头。Transformer／投影器经 CVC、CMC 影响训练梯度，分类推理不直接调用。其他变体按对应版本的实际 `forward` 核对。VCN 的视图间 BN 状态隔离与训练／推理统计一致性分别判断。已有同版本核对可复用。

新判定映射要区分完整方法相对 baseline、候选相对 VCN 支架的新增价值及机制控制；不得用前者收益覆盖后者失败，也不自动套用旧 VBR 的优效门。 实际判定路径的对应、接线缺口与最小验证按 [L2 第2节](../../prompts/LEVEL2_EXECUTION_v4.0.md#2-用现有-brief-完成一页决策)执行，共享细则不在本 L3 重复。

本适配在唯一 Prompt Index 为 MVCNet 选定包含本版的兼容链后生效。已有 campaign、Run、operation 及冻结续接按原绑定处理。本文件维护长期研究主线和模型适配。运行状态以 State 为准，具体轮次的假设、任务、判定门和结果处置见当轮 research brief、冻结记录和收尾。

## 1. 研究主线与宿主身份

**在面向未见被试的运动想象 EEG 解码中，哪些类别判别线索能够跨人复用，模型在什么条件下能利用这些线索获得稳定收益？** MVCNet 提供了一个具体研究场景。训练可利用 raw、flip、freq、cr 多个视图与 CNN/IFNet、Transformer 异构分支，已核对的本地冻结宿主在推理时经 raw IFNet 分类。

候选可以改变视图构造、训练表示、分支交互或 raw 推理路径。完全作用于 raw 推理路径的候选也可作为主创新方向。优先考虑能针对具体信息路径提出有区分力预测的候选。说明拟议改变如何影响最终分类决策、在什么条件下应出现收益，以及哪项观测能推翻这一解释。

研究优先级和从候选到首项结构验证的决策按 [L1 第1节](../../prompts/LEVEL1_RESEARCH_v4.0.md#1-选择能改变研究判断的行动) 与 [L2 第2节](../../prompts/LEVEL2_EXECUTION_v4.0.md#2-用现有-brief-完成一页决策) 执行。VCN-P1 支架、增量解释和必要比较仍按本 L3 下文的精确边界判断。

以下身份锚定于 [MVCNet-S8B-r8 三臂冻结记录](research/identity_freeze_20261118/README.md)。源码、配置、数据角色、选模、结果和权重按原记录与各自合同核对。

| 身份 | 冻结定义 |
| --- | --- |
| B0 · 原生 MVCNet | 原 S8B r8 adapter 与 runner。保留原生 BN 路径和对应合同的增强与训练目标。 |
| S0 · VCN-P1 | 同一 MVCNet/IFNet stem 按视图分离 BN running state 与 affine。训练按实际视图路由，验证与测试固定 raw。 |
| P0 · VCN-P1＋ACM-R1 | 保留 S0 的前向、BN 路由和 CE/CVC 路径，在 CMC 的 IFNet 特征输入端使用 `detach()`，辅助端保留 CMC 梯度。 |

VCN-P1 可作为有范围限制的研究支架。其已有 utility／attribution 证据不自动证明新颖性、跨数据集有效性或足以成为论文主方法；判断回到 anchor、Evidence Index 及对应 Claim 的精确范围。后续候选分别说明自身增量、相对支架的增量和必要的容量／机制解释，不预设与 VCN-P1 互补。

VCN-P1 历史身份以 `VCN_P1_ANCHOR.md` 和源码为准：stem 的 BatchNorm1d 按 raw／flip／freq／cr 路由；P1 分离 running state 和 affine，N1 共享 running state、保留 view-specific affine；训练使用实际视图，验证／测试固定 raw 路径。原实现与原证据作为只读参照。新候选使用独立 AI4S 模块身份；若研究改变这些机制，须明确干预、比较对象和授权范围，不把变化后的实现沿称历史 P1，也不继承其结果。

P0 保留为冻结组合参照和可选训练策略。独立 ACM-R1 的冻结判决为 `NO_SIGNAL`。若未来已有稳定候选 M，且 ACM 的取舍会改变最终模型，可用同协议的 S0＋M 与 S0＋ACM-R1＋M 估计 ACM 的边际价值。组合整体效用、ACM 边际效用与交互分别按对应证据判断。[ACM-R1 原判决](research/mvcnet-acm-r1-screen-lab10-checkrepair2_decision.yaml)

## 2. 实际接口与必要对照

候选经 `MVCNetAdapter` 的 `model.args.ai4s_mvcnet_module` 接入 `src.models.ai4s.mvcnet` gateway，复用统一 runner。不要绕过 adapter 另建训练主体，也不要同时启用旧 `vcn_p1_variant` 与 AI4S 模块路径。仅沿本次改动涉及的路径核对以下事项，无关项不增加每轮前置门：

- **作用路径：**模块是否实际进入 forward、loss、gradient 或目标训练阶段？涉及 view、训练目标或读出时，核对实际视图路由、状态更新及验证／测试 raw 路径；配置声明或类注册不替代实际生效证据。
- **宿主与增量：**写清规范 baseline、VCN-P1 支架及候选的关系。增量关闭时应恢复所声明的宿主，并按需要验证 parity；依托支架的增量主比较应对 S0，控制臂上的优势不能代替该比较。独立的 raw 表示或替换机制可选用适用的原生或等配方宿主，并说明与 S0 公平比较的条件。整体方法对 B0 与增量对 S0 回答不同问题。
- **对照：**可按问题选用规范 baseline、支架、完整候选、机制控制与额外控制，并在简报写明采用的臂及理由。区分 `stage_id` 与 `arm_id`。三臂冻结语境中的 P0 指 VCN-P1＋ACM-R1。其他合同中的 P0、P1、N0、N1、C0 等臂名按对应合同解释。
- **归因：**声称排除容量解释时，选择与该解释相关的匹配控制，说明匹配参数量还是计算量及其局限。参数数目相同不等于表示能力相同；缺少相应证据时收窄主张。与容量无关的问题可不启用 C0。
- **可比性：**数据集、预处理、划分、训练 profile、选模和度量单位从精确配置读取；例如 `accuracy_pp` 不与比例值混用。不同训练阶段、目标、数据范围或适配版本的结果不因同名模块或同 seed 自动成为配对证据。

## 3. 研究认识与来源

按研究总览与直接证据识别当前仍值得检验的问题，依 L1/L2 形成当轮研究简报和可区分的下一项行动。后续任务随证据、收尾判断与授权更新。

历史证据是否覆盖新候选的预测，按 L2 第2、6节核对。相似路线须说明与已关闭干预的实质差异；VCN-P1 的有限支持、新模块的未证实假设、精确负结果和工程失败分别保留。技术失败不作为科学否定，局部否定不扩大到整个机制。

| 用途 | 精确入口 |
|---|---|
| 执行器与规范 baseline 配置 | `ai4s/programs/mvcnet/PROGRAM.yaml` |
| 活动身份与运行状态 | `ai4s/programs/mvcnet/research/STATE.yaml` |
| 证据与精确关闭边界 | `ai4s/programs/mvcnet/research/EVIDENCE_INDEX.yaml` |
| 统一运行入口／adapter | `scripts/run_mvcnet.py`、`src/baselines/mvcnet.py` |
| AI4S 模块与 gateway | `src/models/ai4s/mvcnet/` |
| B0／S0／P0 冻结身份、合同与关键资产 | `ai4s/programs/mvcnet/research/identity_freeze_20261118/README.md` |
| VCN-P1 范围与原实现 | `ai4s/archive/programs/mvcnet/research/VCN_P1_ANCHOR.md`、`src/models/proposed/vcn_p1_mvcnet.py` |

以上是定位入口，不是每轮全部必读清单或有效性证明。先按证据用途和 exposure 选材料，再读取精确合同与必要直接引用；正式评估／Claim 按 `ai4s/prompts/EVALUATION_EVIDENCE.md`，普通探索和筛查不自动升级为正式 Claim。

## 4. 执行边界

字段接口按 runtime“字段接口与产物闭环”核验；VBR 曾因真实 capacity 的 `attempt` 与手工样例的 `attempts` 不同在 TECH 后中断，新路径使用明确调度来源并验证补写后的真实材料，不以手写汇总替代生产输出。旧 Run 保留原失败与复用权限。

新 Transaction 合同默认启用 `analysis_v1` 精简回传；冻结前按共享 runtime“新合同冻结前检查”运行 `validate-new`，明确实际分析依赖、包上限和核验超时，缺字段即停止准备并补齐。已有冻结 Run／Bundle 继续原规则，不补写字段或自动迁移。

部署、排队、资源策略、验收与工程恢复统一继承 L2 及 `ai4s/prompts/TRANSACTION_RUNTIME.md`，不在本 L3 复制调度参数或追加本模型专属门。资源画像对应实际模型、生产输入和完整训练进程；小 batch smoke 的通过不证明正式 batch 可运行。新包读取实际生成的资源策略摘要，异常再核对冻结配置和源码。

普通工程修复、累计成本、有效进展和资格通过后的授权衔接统一按 [L2 工程恢复](../../prompts/LEVEL2_EXECUTION_v4.0.md#5-工程恢复先保留成果再关闭实际缺口)。本 Program 区分源码／验收修复与 VCN 支架、新增量的科学失败；不因后置字段错误重训合格学生，也不以修复为名重开旧 NO_SIGNAL 实例。

本版激活只选择新工作的提示词链，不启动或补跑实验。旧 Run 的科学配置、门、预算、停止与恢复规则保持原冻结绑定；新工作按 Prompt Index 选定的兼容链处理，冻结工作沿精确历史绑定继续。工程恢复复用已完成且合格的训练产物，按原合同验收，不用新规则重释旧失败或重开关闭路线。



