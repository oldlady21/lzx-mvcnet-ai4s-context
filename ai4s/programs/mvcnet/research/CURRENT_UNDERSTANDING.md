# MVCNet 当前研究认识

## 基模来源与认识边界

同一论文的加工材料不作独立证据计数。路径从本文件位置解析：

| 材料 | 用途 |
|---|---|
| [MVCNet-paper.md](../../../../refs/paper-cards/mvcnet/MVCNet-paper.md) | 原文读取材料 |
| [MVCNet-paper-card.md](../../../../refs/paper-cards/mvcnet/MVCNet-paper-card.md) | 贡献与证据边界 |
| [MVCNet-paper-xray.md](../../../../refs/paper-cards/mvcnet/MVCNet-paper-xray.md) | 设计推演 |
| [MVCNet-note.md](../../../../refs/paper-cards/mvcnet/MVCNet-note.md) | 综合理解 |

### 基模设计理解

作者报告：每条 trial 生成 raw 与三种增强视图，经 CNN、Transformer 两分支处理；分类 CE、CVC 跨视图约束和 CMC 跨编码器约束共同训练（Paper X-Ray §2、§3；Note「方法主线」）。阅读解释：CVC 想稳定视图内表示，CMC 想协调两种编码器；CMC 可能也压缩分支独有信息，论文没有直接验证互补性。当前实现须以 `src/baselines/mvcnet.py` 及具体 proposed 文件为准；例如 P1 将四种视图的 BN 状态分开，但仍使用普通 BatchNorm，不能将“视图隔离”推成训练/推理一致性。原文报告仅适用于作者所列数据与协议；本地支持范围见下方 U 证据及其 Evidence ID。关键未决项是完整方法相对强基线的增量、组件归因与适用范围；新证据若无稳定宿主增量，不应将其命名为主创新。

下文是当前可修正综合。作者报告、阅读推断、代码行为、本地观测应按所引来源理解；MVCNet的视图隔离和配套监督不能单凭结构推断互补机制成立。重要结论应能回到原文/源码及Evidence ID，并说明适用范围和何种新证据会改变取舍。历史轮次保留在原报告；其中运行状态只代表记录时点，当前状态以 STATE.yaml 为准。

2026-10-29补充核对：论文阅读材料中“raw-only推理仍保留两分支”的解释属于论文v4。**当前本地VCN-P1分类实现不同**：`src/baselines/mvcnet.py:212` 和原生参考 `_work/augmentation_readout_source_r1/ai4s/programs/mvcnet/research/augmentation_readout/train.py:28` 的推理均仅用CNN/IFNet→分类头；Transformer/投影器只参与训练对比目标。原生CVC沿batch轴拼接两种分支表示，仍含跨分支负项；因此不能假设去掉CMC某侧梯度就隔绝所有分支影响。本轮只是静态事实核对，不改基模、L3、历史合同或结果。后续依赖此事实的方向必须绑定实际native实现，不能套用论文融合分类图。直接核对及源码哈希见[下一辅助方向上下文](next_auxiliary_20261029/CONTEXT_MANIFEST.json)。

## 当前综合与关键缺口

VCN-P1 仍是有范围限制的支架，主创新资格尚未建立。最新固定 VBR 校正未形成稳定支架增量，后续选择应返回主方法证据、强对照与写作缺口；不能默认继续换归一化配方。反例与范围见下方 U2–U5，U5 的原报告保留冻结门与结论。改变此判断需要有实质新依据且符合关闭边界的新设计及合格证据，不能改 seed／日程救援已关闭实例。

以下保留设计时的认识与直接依据；其中“本轮”“选择 VBR”是历史，不是新行动安排。当前运行状态及新工作链只看 State 与 Prompt Index。

## 拟议主线与缺口

拟议主线是：增强视图采用与其统计分布相符的训练归一化，同时保护 raw-only 部署路径。VCN-P1 是有范围限制的支架，不能据此称论文主创新已经成立。主方法的新颖性、规范完整方法比较与适用范围仍有缺口。本轮按用户要求设计辅助方向，其投入理由必须是解决支架的具体局限，而不是再增加一个模块。

## U1：视图间的状态隔离不等于视图内部的训练／推理一致

源码 `src/models/proposed/vcn_p1_mvcnet.py` 中 P1 为 raw/flip/freq/cr 分离 running state 和 affine，但各 bank 仍调用普通 BatchNorm1d。训练使用当前 batch 统计、eval 使用 running 统计是实际实现行为；它在本任务上是否构成有害差异尚未证实。`VCN-P1-S8-SCOPED-2026-08-28` 的角色仅用于继承支架身份和既有支持范围，不用其历史 LOSO 效果选择新配置。`PROGRAM.yaml` 已登记 P1 稳定性 seeds 7..20 为 SATURATED，不再增加同一稳定性重复。

设计后果：允许研究视图内的训练归一化校正，但候选须独立命名；校正关闭恢复 P1，不能把新方法沿称原 P1。相关论文缺口是支架的部署一致性，而非另一个分类器。

## U2：分离增强读出尚未提供稳定的支架增量

`AI4S-MVCNET-mvcnet-augmentation-readout-screen-lab10-r1-resume69-repair2`，直接来源 `augmentation_readout/repair_resume69_oom_20261018/analysis_20261024/REPORT.md`。七源开发主比较 P0−S0 +0.675154pp，但两 seed 为 −0.057870/+1.408179pp，仅4/9被试正向，冻结判定 NO_SIGNAL；不能以均值或 CE 改善补偿失败门。控制比较也未建立按增强类型分头的必要性。

该轮当时的运输与标准 intake 差异见 [原分析](augmentation_readout/repair_resume69_oom_20261018/analysis_20261024/REPORT.md)，不由版本激活抹掉；不从该历史快照判断当前运行。

设计后果：关闭 augmentation-readout-separation-v01 精确实例，不改头数、分组、权重或 seed 救援；新方向不以“共享头冲突已证明”为前提。相关缺口是稳定增量与归因。

## U3：源信息可读、CE改善与可转移效用是不同问题

直接 evidence IDs：`AI4S-MVCNET-mvcnet-srtd-d0-diagnostic-lab10-r1`、`AI4S-MVCNET-mvcnet-srtd-t1-diagnostic-lab10-r2`、`AI4S-MVCNET-mvcnet-srtd-c1-diagnostic-lab10-r1`。各自同名 `_decision.yaml` 显示 D0 为 DIAGNOSTIC_SUPPORT，T1/C1 为 INSUFFICIENT_DIAGNOSTIC_SUPPORT；源内读出可拟合不能推出跨源可转移。`AI4S-MVCNET-mvcnet-source-excluded-distillation-screen-lab10-r4` 的 `source_excluded_distillation/analysis_r4/FOLLOW_UP.md` 保留效用门通过、对机制控制的覆盖门失败；`AI4S-MVCNET-mvcnet-source-class-relation-screen-ccmu-r6` 的 `source_class_relation/analysis_r6/REPORT.md` 保留宿主增量不足，弱控制优势不能替代它。

设计后果：本轮不选择 teacher、源路由、类关系或新头；新归一化设计不借这些结果声称机制缺陷已证实。必须同时比较宿主增量和通用归一化改动，防止把一般训练改善称为 VCN 特有互补。

## U4：负结果关闭精确干预，工程失败不否定科学机制

`AI4S-MVCNET-mvcnet-source-head-meta-screen-ccmu-r4` → `source_head_meta/analysis_r4/PRECISE_CLOSURE.json`；`AI4S-MVCNET-mvcnet-source-class-mean-alignment-screen-ccmu-r1` → `source_class_mean_alignment/analysis_r1/REPORT.md`；均维持精确 NO_SIGNAL，不做参数救援。Evidence Index 中 conv-SAM、SED、增强读出的 INVALID 工程条目不并入零效果统计，后续有效轮也不删除原成本与失败。

`VCN-V0-1-PARENT-BOUNDARY-2026-08-21` 的 old_moment_only_route_reentry 禁令仍有效。新候选不是重新用 moments 可分性释放旧路线：保留 P1 已有视图状态/affine布局，只前瞻改变每个 bank 的训练归一化算子。若后续直接合同核对显示与禁闭干预同一，撤销候选，不以更名规避。

## 历史设计取舍（已由 U5 收尾）

选择 [VBR-R1](vbr-r1/research_brief.md)：视图内训练归一化校正的单配置研究设计。它是已知 Batch Renormalization 思路的宿主适配，原创性未建立。若仅得到一般 renorm 收益，不包装成第二创新；若无稳定支架增量，关闭固定实例并回到主方法证据／写作缺口，不默认再换模块。

曝光说明：本次读取了 anchor 中既有 scoped LOSO 摘要及父级 discovery 边界，其已见事实不能擦除。上述数字不用于选择 VBR 超参/阈值；设计主要依据源码行为和 developmental 关闭边界。未来相同被试/数据仍属于已暴露开发范围，新 seed 不提供独立最终确认。正式评价必须另行审计选择历史。

## U5：固定 VBR-R1 校正未建立超越 VCN-P1 的稳定增量

`AI4S-MVCNET-mvcnet-vbr-r1-screen-lab10-r1-repair3`，直接记录 [最终报告](vbr-r1/execution/repair3/analysis_20261029/REPORT.md)及同目录 `PRECISE_CLOSURE.json`。repair3-packfix2 完成72 fresh fits及72 checks；完整远端逐SHA核验、冻结分析、独立复算和核心包本地41,472预测重算一致。按本轮用户新规则，完整约46.83GB保留Lab10，只回拉约27.22MB正式核心产物，明确 `full_payload_local=false`，不是不完整结果被当作全量本地验收。

P0−S0 +0.019290pp；seed69 −0.250772、seed70 +0.289352pp；正向4/9。均值、双seed同向与覆盖三门失败，冻结判定NO_SIGNAL。CE gain +0.012653与0名严重受损通过其各自门，不能补偿主门。P0−B0 +2.015818pp不说明VBR新增价值，因为S0−B0已有约+1.996528pp；本轮也不把N0或交互描述升级为新候选。

判断变化：视图内训练／推理统计差异仍是算子事实，但该固定有界校正未表现出稳定的支架效用；不再以本轮结果支持“这是VCN下一辅助创新”。这不否定所有归一化设计，也不否定既有VCN scoped证据。设计后果：关闭 `vcn-view-batch-renorm-v01`，不通过调日程/限幅/seed或更名救援，返回主方法证据、对照与写作缺口；没有自动新训练授权。State已完成本轮、暂停且无活动Run，原275h累计预算和2026-10-24T04:35:51Z起点、全部工程失败历史保留。

## U6：v3.3下的下一项设计取舍——先区分来源条件与通用标签保护

2026-10-29，仅设计。v3.3已是MVCNet新工作活动链，原VBR工作仍按v3.2关闭；版本升级不更改其门、预算或结果。主方法的强近邻/归因/适用范围证据仍是成文关键缺口，辅助方向不自动变成更高优先级的训练投入。

源码认识更新：原生`_native_objective`调用的`NTXentLoss`按trial组织CVC，通常把其他trial置入负集合；`utils/contrastive_loss.py`同时已有`NTXentLossNeg/NTXentLossClass`草稿，但并非当前原生调用。其存在与SupCon/去偏对比、SCLDGN等先例说明，“普通同类mask”不能包装为新发现；草稿的相似度乘零也不等于数学上从分母删除。这里没有运行或修写上游代码，没有由方法存在推断训练效果。

用户要求继续围绕VCN-P1找辅助点。本轮建议保留一个可被简单对照推翻的问题：[XSN-R1跨来源同类负样本保护](xsn-r1/research_brief.md)。只在保标签的raw↔flip/freq CVC中，停止不同来源同类异trial之间的排斥，不添加同类吸引；CR/CMC、BN、分类头及raw部署保持。它不同于旧同trial跨目标碰撞图删边、固定BRN校正、分头或源类均值对齐，但**尚未证明来源限制有额外价值或有新颖性**。

选择后果：关键对照至少包含S0、XSN、匹配跨源随机删边、通用同类保护。若不能稳定超过S0，关闭固定实例；若与随机/通用保护相当，停止其选择性解释，不自动调mask或转推控制臂。若要声称VCN特异协同，另需B0与B0+XSN的配对交互；本次没有这些结果。保持同源同类排斥可能有害或保留来源捷径，是候选的核心风险。

本次只完成限定构思、直接近邻核对和brief，未调用完整查新/评审、未实现或训练、未分配新预算/Run/Claim。未选构思不登记为负结果；全部候选及实际适配范围见同目录`CANDIDATE_POOL.json`，来源与曝光说明见`CONTEXT_MANIFEST.json`。

### U6后续修正：限定查新与反驳并未支持立即投入

2026-10-29用户后续授权限定查新及独立科研反驳。Grok-4.6-build真实最终意见为查新`PROCEED WITH CAUTION`、科研`HOLD`，不是XSN科学失败。CALDA等已比较来源条件对比集合，普通标签/source对比不是新意；尚未核到完全同式论文，也不据此宣称首次。

P0删跨源同类，N0删所有同类，原P0−N0混入数量/分母质量。现行完整设计增加同类内等量随机保护R0，与S0/P0/C0/N0组成五臂；C0/R0随机化和非退化仍pending。保留同源同类排斥可能有害或保留捷径，不能作为机制事实；SCMA与XSN是相关source×class几何问题上的不同算子，而非完全独立的新问题。原SCMA正式NO_SIGNAL已本地回查，审查者仅见旧brief造成的经验结案pending限定原样保留，没有虚称复审改判。

判断后果：当前不支付完整矩阵，先解决零训练mask/度数可行性；本轮未执行该检查、未训练。五臂约58.677process-h外推，不是独占GPU成本；新预算未分配，原State/Run/275h计时不动。主Agent修订不等于独立接受，HOLD持续。直接报告：[限定查新与反驳](xsn-r1/review_20261029/NOVELTY_AND_REBUTTAL.md)、[对照与成本](xsn-r1/review_20261029/CONTROL_COST_ANALYSIS.md)。

后续用户授权零训练审计后：[144个训练元数据batch的审计](xsn-r1/mask_audit_20261029/REPORT.md)证明C0/R0在该有界样本中可合法匹配且非退化，保存的288控制图独立读取复核通过；R0平均仍与P0重合92.92%，N0比P0平均多删71.389条trial无向边。数量混淆确实存在，可构造性改善不提供效用证据。退化反例、非均匀交换链/未证明混合、未重放生产损失仍限制解释；维持设计HOLD、不直接投入90fit，未训练。加入当前CPU控制构造开销的假设性五臂规划约63.71process-h，4有效并发含假设缓冲约25.9–29.9h，非新GPU报价或预算授权。





## U7：等待XSN时的下一方向设计——CMC的两侧训练职责

2026-10-29，用户仅授权并行设计、不启动新训练。U6中的HOLD/预算描述保留原时点；之后用户选择XSN一次性探索和独立36h预算，精确执行身份见State与`xsn-r1/execution/PROGRESS.md`。本次没有读取其运行中效果，也未改变XSN的授权、合同或预算。

新设计依赖本页上方补充的论文/实现区别：本地CNN承担分类，Transformer为训练辅助；普通BN视图隔离不隔离CMC梯度。结合旧CVC×CMC整项诊断的INSUFFICIENT_DIAGNOSTIC_SUPPORT、VOGR未支持梯度冲突，以及SCVA子空间精确关闭，主Agent暂选[ACM-R1：分类主干为参照的单向CMC](next_auxiliary_20261029/research_brief.md)。这复用原候选池的stop-gradient备选，只改变CMC反向接收端，保留原前向/正负集合和CE/CVC，不改变推理或叠加XSN。

关键反驳为四臂：原双向、只更新辅助分支、只更新CNN、两侧均不反传。P0必须先有S0增量，再区分完全关闭CMC与反向职责；P0/N0首步CNN更新应一致，后续可经原CVC间接分化，因此不能宣称完全隔绝辅助影响。旧VOGR不构成“梯度有害”前提，旧因子实验也不证明删除CMC有利。stop-gradient/非对称学习已有充分一手先例，目前仅是有可辨识对照的开发候选，未建立新颖性或VCN特异互补。

此项依据是静态代码和历史开发证据，不是新科学结果。建议先做限定查新与独立反驳，再决定是否值得72fit；预算未分配，不借用XSN36h或VBR275h权限、不占用新seed、不建Run/Claim。未选备选保留，不登记为实验否定。设计登记ID：`MVCNET-VCN-NEXT-AUX-DESIGN-20261029`。


### U7后续修正：单侧保护已有近邻，先验证可辨识性

2026-10-29用户授权ACM-R1限定查新与独立科研反驳，仍不启动新训练。真实Grok-4.6-build最终结论为查新PROCEED WITH CAUTION（5/10）、科研HOLD_TRAINING；原文及trace见[next_auxiliary审查报告](next_auxiliary_20261029/review_20261029/NOVELTY_AND_REBUTTAL.md)。ADM（ICML2025）已有单侧特征梯度保护，NPL2024已有按部署角色安排不对称训练，故不能把detach或角色分工本身写成新贡献。可能的贡献只在本地CNN分类、训练辅助A经保留CVC耦合的具体发现，尚无效用证据。

P0对完全关闭N0、反向R0均没有实测支持；静态推导P0/N0首步F/C与Adam更新相同，后续可以经CVC分化，不能称全程隔离。旧CCF的CMC增量+0.491898pp未过冻结门，旧VOGR诊断不足；本轮不以它们证明删除有利或梯度有害，也不重开VBR/COC/SCVA/推理桥接融合实例。

设计后果：保持S0/P0/R0/N0四臂窄问题，先看宿主增量，再看关闭与反向的必要性；普通CMC衰减仍是竞争解释。第五固定0.5双侧衰减臂不是精确范数匹配，不自动增加。下一项有价值的是非XSN生产objective的零训练梯度表、S0 parity、P0/N0首步及CVC非退化检查，尚未执行，不记PASS。四臂历史工作量46.942process-h；有效并发4的条件规划20.67–23.60h、并发2为35.34–41.21h，含未实测倍率与6h统筹余量，不是上界或ACM报价。

ACM预算UNASSIGNED，无新seed/Run/Claim。XSN36h与VBR275h原起点/历史保持；未读取XSN中间效果、未probe或变更当前Run。后续即使图资格通过，也只说明设计可比较，仍需比较ACM与补充VCN/XSN证据的投入价值及独立执行预算。登记ID：MVCNET-ACM-R1-NOVELTY-REBUTTAL-20261029。


### U7零训练审计：接收端定义可实现，间接路径在合成实例中非退化

2026-10-29用户授权隔离副本审计、不启动新训练。记录[MVCNET-ACM-R1-GRADIENT-AUDIT-20261029](next_auxiliary_20261029/gradient_audit_20261029/REPORT.md)：53项主检查通过，保存核心张量另脚本复算两个原生实例；S0与原生参考的梯度/参数/Adam精确一致，P0/N0首步F/C及Adam一致。固定F只交换更新后A，CVC→F梯度差范数0.836111/0.428088；固定同一A或断开跨分支耦合后均为0。该计算依赖在所测实例非退化，不是分类效用。

覆盖实际大小模块、合成batch2/4和空Adam首步，未读真实EEG；10个临时副本各一次技术更新、0科学拟合。未验证生产增强、batch64、长期优化或Lab10成本。主Agent将判断从图资格HOLD推进为“有条件值得一次四臂开发探索”，仍不立即派发，预算UNASSIGNED。保持四臂与关闭/反向对照，不增第五臂、不调权救援。前轮Grok原HOLD和5/10查新意见保留，本次没有冒充外审改判。

成本仍采用历史46.942process-h；有效并发4为20.67–23.60h、并发2为35.34–41.21h的条件情景。小batch审计主体59.801秒及CUDA allocator峰值不转成生产报价。XSN当前Run和36h、VBR历史275h及各原起点保持，未访问XSN中间效果或远端。
