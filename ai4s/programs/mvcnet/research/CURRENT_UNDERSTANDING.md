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


## U8：XSN-R1未形成VCN-P1支架增量（2026-11-04）

MVCNET-XSN-R1-VERIFIED-NO-SIGNAL-20261104；证据 AI4S-MVCNET-mvcnet-xsn-r1-screen-lab10-r1-budget480。

XSN-R1同Run完成90fit+90check，0失败/0OOM；完整结果留Lab10，核心正式analysis_v1_recovery intake，原始logits两套独立复算一致NO_SIGNAL。P0−S0 −0.192901pp（seed71 −0.135031pp、seed72 −0.250772pp；5/9正向）。P0−C0 +0.096451pp、P0−N0 +0.212191pp未过门；P0−R0 +0.366512pp过门不能替代主门。关闭该固定开发配置，不新增训练或释放Claim。

证据引用 → 判断变化：CE改善0.011905且无≤−5pp被试，不能替代准确率门失败；没有充分依据把来源限制列为VCN-P1的必要辅助模块，不泛化为所有source-aware对比学习无效。来源：[正式结案报告](xsn-r1/execution/budget480/analysis_20261104/REPORT.md)、[精确关闭边界](xsn-r1/execution/budget480/analysis_20261104/PRECISE_CLOSURE.json)。

设计变化或停止 → 下一项对照：关闭此v01实例，不改mask、温度、seed或门救援，不自动改推N0/R0或叠加ACM。ACM零训练资格保留原角色、不继承训练授权；若另选问题，先比较与补齐VCN主方法证据的投入价值，本轮不创建新候选或对照。原480h预算、计时起点和全部失败修复历史保持。


## U9：XSN关闭后，ACM只保留为预算受控的职责检验

2026-11-04；MVCNET-ACM-R1-POST-XSN-REASSESSMENT-20261104。证据→判断：XSN负结果关闭删边v01，并未测试CMC接收端，不为ACM提供正支持。ACM既有53项合成资格和CVC间接作用非退化说明四臂可辨识；CCF/VOGR未建立CMC有害，既有不对称近邻限制仍在。详见[重评依据与关键对照](next_auxiliary_20261029/reassessment_20261104/REPORT.md)。

判断→设计/停止：仅有条件值得一次S0/P0/N0/R0开发探索，先宿主增量，再关闭和反向必要性；任何必要门失败收口固定实例，不调权/加seed/切方向或叠加XSN。四臂不排除一般减权、不证明VCN协同；主方法强近邻/范围证据仍比默认加模块更优先。下一项是将来获准后生产objective绑定与B64/目标机资格、固定判读/成本门，并非本轮执行。

成本→投入：近期XSN原生S0收据×4为37.512192process-h；K4含假设倍率/6h余量约17.72–20.07h，工作量敏感性及旧VBR参照约20.15–23.60h；K2情景29.45–41.21h。36h只是预算商议目标，不是已分配上限或实测可行性。ACM预算UNASSIGNED，XSN480h余款不转用。Program暂停/运行状态保持，本轮无新训练、seed、Run或Claim。


## U10：ACM 固定职责实例已按冻结门关闭（2026-11-05）

历史时点修正：U9 的有条件探索后，用户另授独立42h并完成执行/工程修复。证据 `AI4S-MVCNET-mvcnet-acm-r1-screen-lab10-checkrepair2`；[正式关闭报告](acm-r1/execution/checkrepair2/analysis_20261105/REPORT.md)。72个模型复用、72项check完成/独立logits复算，新增fit0；P0-S0 +0.733025pp主门通过，P0-N0 +0.868056pp通过，但P0-R0在seed74为-0.115741pp，关键逐seed门失败，NO_SIGNAL。保留开发效用线索，不升级为单向机制/论文Claim，不调参救援。原42h及起点/全部失败历史保持；截至既有最终结算累计23.823807h，不转为下一方向预算。

## U11：最小归一化结构已收束，不作新增主创新训练（2026-11-06）

证据 `MVCNET-VCN-MINIMAL-STRUCTURE-READONLY-ZERO-TRAINING-20261106`；[唯一研究brief](normalization_minimal_20261106/research_brief.md)与[结构化决定](normalization_minimal_20261106/DECISION.json)。绑定v3.3，只读设计/隔离CPU零训练，Program State与全部冻结Run/预算不改。

证据→判断：P1/N1已有scoped支持，不重训N1；两个真实IFNet+native CE/CVC/CMC合成实例中，仅合并三个增强running缓冲、保持四affine和raw统计独立，输出/全部已核查梯度/raw预测差均0，完整state_dict不同。共享增强affine具有非同view梯度和梯度求和关系，是活干预，但不能用可辨识性证明性能；固定bank置换只是重命名。B64、长期优化、checkpoint别名和完整gateway动态路径未覆盖，不为已放弃M补资格。

输入/近邻→设计后果：当前生产资产四hash闭合，当前X18条重放匹配trial[0,4)，非cue后[0,4)；历史S8 X字节身份未知，不改旧内部配对/曝光。已找到并核验5hash的既有cue资产候选，不需假设重建数据，但MVCNet协议未绑定，也不产生独立确认。AdvProp/DAJAT/DualBN已覆盖多BN及统计-affine拆分；真实Grok独立反驳STOP_MAIN_INNOVATION_INVESTMENT，原文/主Agent处置完整保留。

最终选择→下一行动：STOP_AS_NEW_MAIN_INNOVATION / HOLD_TRAINING，保留P1既有支架身份；只保留P1对普通raw/aug双BN作为未来论文补证草案，不自动开训、加随机路由臂、救援旧实例或借ACM/XSN预算。12-18h仅是旧100epoch两臂情景假设，非新Lab10报价/授权。下一步先收窄论文贡献与评价对象；cue对齐完整方法净收益是另一明确科学问题，不能冒充本题续跑。此为投入判断而非新科学NO_SIGNAL。


## U12：cue 对齐条件下先检验完整训练方案（2026-11-06）

证据 `MVCNET-CORE-DIRECTION-CUE-ALIGNED-DESIGN-20261106`。既有 P1 scoped、VBR/XSN/ACM 固定实例关闭及最小结构审计共同限制新增支架改动的论文价值。多视角构思初选逐试次参考，经独立科研反驳和限定近邻核对，将 cue 对齐下原生 MVCNet 相对同 IFNet、同增强监督的净增量列为下一设计优先项。P1−B0 另答支架适用范围。主比较成立后再考虑等长度窗口的提示时序依赖，窗口差只能先作条件性证据。

现成 cue 资产有五项哈希核验记录，当前数据 loader 与 role／trial-ID 仍需映射。下一步只读绑定资产、事件、划分、信息与训练预算，核对后决定是否值得配对比较。已暴露九被试限开发；预提示参考保留为次级问题，待因果输入、最近历史对照与独立评价可用性核查。范围及判别见[研究简报](next_core_direction_20261106/research_brief.md)与[决策](next_core_direction_20261106/DIRECTION_DECISION.json)。


### U12 只读绑定结果

证据 `MVCNET-CUE4-READONLY-PROTOCOL-BINDING-20261106`。cue 资产五文件与 18 份 MAT 哈希通过，5,184 条试次身份与旧 MVCNet 输入逐行对应，18 条抽样数组重放确认 500 点窗口偏移。MVCNet 的 18 份七源参考索引在 cue 元数据上闭合，DSAINet 的六源角色保持单独。旧生产入口依赖旧字段、哈希和 VCN-P1，cue 原生 B0 仍需独立生产接线资格。

历史 CVC×CMC 已在旧窗 P1 上比较四视图 CE，联合增量 +0.279707 pp 未过冻结门。当前唯一主比较草案是 cue 窗无 VCN B0 对同 IFNet、同四视图 CE。统一监督与更新步数，计算量单独计量。继续协议准备，暂缓完整训练。数据与角色证明、限制和投入处置见[绑定报告](next_core_direction_20261106/cue_protocol_binding/REPORT.md)。

### U12 两臂隔离零训练验收

证据 `MVCNET-CUE4-TWO-ARM-ZERO-TRAINING-20261106`。源 loader 通过 18 份参考划分闭合、B64 批次与角色拒绝检查。实际四视图匹配原生输入及标签规则，同 IFNet 的共享 CE 梯度一致，B0 的 CVC+CMC 确实进入梯度。两臂源侧 raw 验证共用正确数、平均 CE 和 epoch 的顺序，拒绝越权和不完整验证。真实 EEG 仅用于源批次与增强，模型前向及梯度使用合成输入。

隔离资格通过后，仍暂缓完整训练。仅在论文需要明确检验原生配方相对强增强监督的净增量时，考虑一次两臂开发比较。历史尺度约 19 process-hours。先定义有成文价值的最小差异和个体风险，再核验生产封存链及两臂 Lab10 实际成本。九个已暴露被试只提供开发证据。[验收与投入评估](next_core_direction_20261106/cue_two_arm_zero_training/REPORT.md)

### U12 生产接线、科学门冻结与 Lab10 启动故障

证据 `MVCNET-CUE4-PRODUCTION-QUALIFICATION-20261106`。已接通真实源拟合、raw 选模、全局 36 checkpoint 封存及逐折检查和判读消费者。平均 B0−CE 至少 +1 pp，两 seed 均值各为正，至少 6/9 被试改善，最差被试的两 seed 平均至少 −3 pp，并按 95% 被试配对 bootstrap 区分稳定开发信号和不确定。资格与科学合同身份分离。

Lab10 资格任务 0/4，全部在加载数据和更新前遭遇同一导入路径错误。原包和日志保留，入口修复及无训练的导入重放已完成。冻结的一次任务尝试均已消耗，后继资格须另授一次工程修复。成本仍缺两臂生产形状实测，科学拟合与开发检查均为零。[协议与工程收尾](next_core_direction_20261106/cue_production/REPORT.md)

### U13 sealfix2 后的逐试次参考核查

`sealfix2` 的 [NO_SIGNAL](mvcnet-cue-aligned-two-arm-full-lab10-r1-py39fix1-sealfix2_decision.yaml) 保留。新零训练核查在名义视觉 cue 下通过参考和任务截止的因果性回放，精确匹配实际派发包的 18 份划分，采集时序共同试次 4,860。严格同角色历史的源验证每折仅 3–10 项。同试次参考相对等长最近历史、等点数三窗替换的全任务窗被试中位距离优势均很小，较短历史时距层方向反转，现有时序无法分离试次身份与时间邻近。按原停止规则结束该方向的训练投入。四视图 CE 继续作为同期强基线，VCN-P1 保留为原历史协议的限定支持证据。[零训练核查与处置](next_core_direction_20261106/PRECUE_ZERO_TRAINING_AUDIT_20261108.md)

### U14 预提示参考关闭与下一候选

依用户指令正式关闭逐试次预提示参考，登记 `MVCNET-PRECUE-ZERO-TRAINING-CLOSE-20261108`。后续只设计一个独立候选，在四视图 CE 的固定一路训练增强上扰动提示起始，以等长错位遮挡和简单晚窗检验真实换提示时的净收益。现成 Supernumerary V3 有同被试双提示，但按日采集，资产触发、时钟和曝光须先验收。当前仅为选题草案，最低下一关为零训练。[候选与停止条件](next_core_direction_20261106/CUE_ONSET_AUGMENTATION_DESIGN_20261108.md)

### U15 提示起始遮挡在 Supernumerary V3 数据关停止

40 份 XDF 的 1,600 个左右想象任务窗可定位，未找到两条件的逐试次视觉提示起始标记。XDF 通道头与发行方位置表冲突，去抖显著改变 EEG 时间戳。匹配遮挡的合成控制通过，前向 SOS 与原生 Hilbert 频移在遮挡前处理都会把早段信息带到遮挡外。按原规则 `STOP_CUE_ONSET_UNVERIFIED`，当前资产和管线不进入四臂开发比较。这是数据与辨识性停止，未测模型效用。[实测与处置](next_core_direction_20261106/CUE_ONSET_ZERO_TRAINING_20261108.md)

### U16 提示起始遮挡方向收束

依用户指令关闭 Supernumerary V3 验证路径。限定替代核查未发现同时具有同被试双提示、两种实际起始事件、EEG 时钟和通道映射的公开资产。最接近的 BCIC IV Dataset 1 可由官方标签恢复大多数语音评分边界，仍缺实际呈现事件。DynamicCues 的原始发布未取得。登记 `CLOSED_NO_QUALIFIED_ALTERNATIVE`，提示起始遮挡不进入四臂开发比较。保留 Supernumerary 零训练回执、逐试次参考关闭及 sealfix2 的 `NO_SIGNAL`。[替代核查与收束](next_core_direction_20261106/CUE_ONSET_ALTERNATIVES_CLOSE_20261108.md)

### U17 用已有预测证伪训练配方分歧的错误辨识价值

选题草案。`sealfix2` 两臂准确率接近，已有正式核心包可在同一试次上区分两臂错误重叠和分歧。固定 CE 的预测，比较同 seed B0–CE 分歧与单 CE 置信度、普通 CE 重复分歧和相同两模型的平均概率置信度。最低验证是 CPU 离线等覆盖率错误排序，无新增模型前向、拟合或下载。两重复的源划分也不同，且九人已暴露，只支持开发证伪。两项时间参考/遮挡关闭和 `sealfix2` 判定保持原记录。[候选与对照](next_core_direction_20261106/PAIRED_RECIPE_ERROR_RANKING_DESIGN_20261108.md)

### U18 配对分歧未改善 CE 错误排序

CPU 核查已完成，处置 `STOP_NO_INCREMENTAL_ERROR_RANKING`。固定 CE 输出，每人保留 518/576，配对 JS 比单 CE 最大概率、同对平均置信度的错误率分别高 1.512、1.652 pp，两重复均落后，均为 0/9 人改善。相对普通 CE 重复为 −0.086 pp 改善，重复方向相反、5/9 人改善。匹配预测类别拒绝配额后仍落后主要置信度对照。结束当前路线的独立复核投入，保留所有旧结论。[正式核查](next_core_direction_20261106/PAIRED_RECIPE_ZERO_TRAINING_20261108.md)

### U19 配对分歧正式关闭与固定截止候选

依用户指令关闭 B0–CE 配对分歧方向，登记 `MVCNET-PAIRED-RECIPE-DIRECTION-CLOSE-20261109`，复用原零训练回执。下一候选只做设计，复用现有 2a 连续数据和试次闭集，在名义 cue 后 2 秒、4 秒共享因果主干并保留独立位置读出。最强反证为相同截止独立 CE 或等监督预算短窗 CE。先核前缀因果、训练梯度与缓存等价，再判断是否有资格申请比较。既有三项关闭和 `sealfix2` 的 `NO_SIGNAL` 保留。[设计与停止条件](next_core_direction_20261106/CAUSAL_TWO_DEADLINE_DESIGN_20261109.md)

### U20 双截止 CPU 技术核查通过

18 份 2a MAT、5,184 项嵌套共同试次及 18 个既有划分闭合。每个截止的原始波形扰动与截断回放一致。合成 IFNet 探针通过后缀扰动、完整输入梯度和缓存等价检查，长窗 CE 到共享主干梯度非零。处置 `PASS_ZERO_TRAINING_REVIEW_WORTHY`，仅建议进入独立科研复核，继续保留提示效应和训练预算匹配的反证。未执行外审或开发比较，既有关闭与 `sealfix2` 判定保留。[技术核查与边界](next_core_direction_20261106/CAUSAL_TWO_DEADLINE_ZERO_TRAINING_20261109.md)

### U21 双截止外审与开发派发前置门

Grok 4.6 对原设计独立反驳为 `HOLD_FOR_DESIGN_FIX`。修订后保留同截止独立 CE，加入两个 4 帧头的等监督短窗和 4/8 帧头的等容量短窗对照。匹配两项 CE、处理点数与更新，实测计算和共享梯度。合成 CPU 计时中 2×500 缓存的两次合计中位数下降约 23%，8×125 分块反而更慢。仅建议继续生产形状零训练接线、回放计时与成本资格，当前 `HOLD_BEFORE_DEVELOPMENT_DISPATCH`。早窗仅支持已知提示下的定时四分类。既有关闭和 `sealfix2 NO_SIGNAL` 保留。[外审及本地修订](next_core_direction_20261106/CAUSAL_TWO_DEADLINE_DEVELOPMENT_GATE_20261109.md)

### U22 五臂技术验收通过，含因果前端的合计 p95 未过门

显式候选经实际 adapter/gateway 接通 J/E2/E4/E/P。B64 源侧接线、等监督、J/P 活跃读出容量以及长窗到共享主干梯度均验收，零优化步。分包与前向 SOS 纳入后，2×500 缓存首段和第二段 p95 过门，合计 p95 从 2.2685 降到 2.0775 ms，仅省 8.42%。按原门 `STOP_TOTAL_P95_GATE`，不进入两重复开发或新 Lab10 资格。成本仅更新为历史锚点情景，保留全部旧关闭与 `sealfix2 NO_SIGNAL`。[生产形状资格与停止判断](next_core_direction_20261106/CAUSAL_TWO_DEADLINE_PRODUCTION_QUALIFICATION_20261109.md)

### U23 双截止正式关闭，转向同提示被动观看负对照

依用户指令登记 `MVCNET-CAUSAL-TWO-DEADLINE-DIRECTION-CLOSE-20261109`。五臂零训练资格与原始回执保留。含分包和前向 SOS 的合计 p95 仅下降 8.42%，低于 10% 预设门，关闭双截止开发投入。逐试次预提示参考、提示起始遮挡、B0–CE 配对分歧的关闭结论及 `sealfix2` 的 `NO_SIGNAL` 保留。[资格与停止记录](next_core_direction_20261106/CAUSAL_TWO_DEADLINE_PRODUCTION_QUALIFICATION_20261109.md)

`MVCNET-SAME-CUE-PASSIVE-SPECIFICITY-DESIGN-20261109` 仅登记设计。利用已公开的 HaLT 和 NoMT 记录，问主动训练的左右手解码器在同提示被动观看中是否仍能识别方向。目录可配同人七组，跨日与实际被动方向码和时钟尚待核查。先限定 Subject K 两份原始 MAT 做无拟合数据资格。通过后才核对全组，随后才有理由评审轻量读出。当前无原始波形读取、训练或新 Run。[设计与停止门](next_core_direction_20261106/SAME_CUE_PASSIVE_SPECIFICITY_DESIGN_20261109.md)

### U24 同提示负对照在 Subject K 实际提示资格停止

登记 `MVCNET-SAME-CUE-PASSIVE-K-ZERO-TRAINING-20261109`。HaLT、NoMT 两份原始 MAT 与作者采集脚本均通过发布哈希。两条件同为 200 Hz、相同 19 路 EEG 顺序，各有 163 个左发布码和 159 个右发布码。第 22 列使 322/322 个起点在 ±1 点内唯一复现，MAT 通道名为 X5，与论文 X3 的物理映射未核实。表 5 与发布 eGUI 的 2/3 图标索引冲突，缺逐试次界面源日志。独立硬件结束标记未验出，1 秒共同窗仅按发布 marker 成立。按硬门关闭这条 Kaya 发布资产路径，未扩到其余被试，未训练。保留既有关闭和 `sealfix2` 的 `NO_SIGNAL`。[实测与处置](next_core_direction_20261106/SAME_CUE_PASSIVE_K_ZERO_TRAINING_20261109.md)

### U25 同提示特异性方向因缺合格替代资产收束

登记 `MVCNET-SAME-CUE-PASSIVE-ALTERNATIVES-CLOSE-20261109`。限定核查八个候选，合格数为零。最接近的公开篮球数据同人有纯观察和纯想象。固定发行件实读表明运球事件按任务状态标码，缺可追溯的左右手码与实际显示起止；传球是复合序列。其余候选缺同提示被动条件、仅覆盖单手、改变呈现或当前无公开资产。按原规则收束方向，保留 Subject K 核查、Kaya 路径关闭和 `sealfix2` 的 `NO_SIGNAL`。零拟合，零新 Run。[替代资产核查](next_core_direction_20261106/SAME_CUE_PASSIVE_ALTERNATIVES_CLOSE_20261109.md)

### U26 以独立校准回放检验眼电扰动敏感性

登记 `MVCNET-OCULAR-REPLAY-DESIGN-20261109`。同提示方向关闭后，选择本地 2a 中独立于任务标签的前置眼电校准。对同一 EEG 波形作原波形、真实校准扰动、等能量空间对照三方配对，另以错位校准验证场估计。已查 A01T/E、A04T 的块结构及 seed73-outer0 的被试角色。拟在两名源被试 A03/A04 的 T/E 上做一次 CPU 零训练资格，先验证跨会话混合、等能量和非退化，失败即止。当前设计未执行资格或模型重放。保留同提示关闭、Subject K 核查及 `sealfix2` 的 `NO_SIGNAL`。[设计与停止门](next_core_direction_20261106/OCULAR_REPLAY_DESIGN_20261109.md)

### U27 眼电同步跨会话通过，固定注入因滤波越过试次边界停止

登记 `MVCNET-OCULAR-REPLAY-ZERO-TRAINING-20261109`。A03/A04 的 T/E 四份 MAT、10 个校准块、1,152 条源侧试次身份闭合。E 的 4 秒窗中，同步较错位的归一化平方误差中位优势为 A03 28.76%、A04 14.35%，逐校准块留一方向均为正。按本次固定中央 1 秒注入与 1,651 点连续 FIR，首条 A03T 的理论支撑越过下一试次起点 44 点。`STOP_PAIRED_INPUT_GATE`，尚未生成对照波形或核验等 RMS／拓扑，当前不进入设计复核。保留同提示关闭、Subject K 核查和 `sealfix2` 的 `NO_SIGNAL`。零模型前向、零训练、新 Run 为零。[核查报告](next_core_direction_20261106/OCULAR_REPLAY_ZERO_TRAINING_20261109.md)

### U28 共同位置解决固定试次边界，空间对照未过可辨门

登记 `MVCNET-OCULAR-REPLAY-SHARED-POSITION-20261109`。四条固定试次的共同安全起点为 cue 后 325–331 点，按中点取 328。首条 A03T 实测滤波隔离和等 RMS 通过，真实与分区置换对照的余弦 0.957123 超过既定 0.95，按规则结束固定方案，余下三条只核边界，未进入设计复核。保留上次试次隔离失败、同步通过、同提示关闭、Subject K 和 `sealfix2` 的 `NO_SIGNAL`。无模型前向、训练或新 Run。[补测与关闭记录](next_core_direction_20261106/OCULAR_REPLAY_SHARED_POSITION_20261109.md)

### U29 眼电固定方案正式关闭，登记同窗时间反演设计

依用户指令登记 `MVCNET-OCULAR-REPLAY-DIRECTION-CLOSE-20261109`。同步跨会话通过、原位置试次隔离失败、325–331 点共同安全位置和首条空间对照余弦 0.957123 超过 0.95 均保留。既有各方向关闭、Subject K 核查及 `sealfix2` 的 `NO_SIGNAL` 保留。

`MVCNET-TIME-REVERSAL-ODD-EVIDENCE-DESIGN-20261109` 仅设计。晚窗同试次正放与倒放的功率、实部互谱、零时滞协方差和幅值分布相同，有符号虚部反号。以共同平移、cue 前等长窗和 run 内置换排查伪差，先核其他源被试的 T 对照能否在留出人的 E 上复现。已从固定折元数据计得 403 条源验证左右手试次，未读取信号做效果统计、未拟合或启动 Run。[设计与停止门](next_core_direction_20261106/TIME_REVERSAL_ODD_EVIDENCE_DESIGN_20261109.md)

### U30 晚窗有符号时滞在跨人 T→E 门停止

登记 `MVCNET-TIME-REVERSAL-ODD-ZERO-TRAINING-20261109`。403 条源验证左右手试次及 806 个窗的正放、倒放和共同平移谱不变量通过，391 条进入双类 run 对照。晚窗仅 4/7 人同向，等权晚窗投影 +0.034532 低于置换 P95 +0.190738，晚窗相对 cue 前的增量 +0.189301 低于 P95 +0.360775。三门均未过，固定方案停止于零训练，未进入设计复核。保留眼电回放关闭、Subject K 核查、同提示关闭及 `sealfix2` 的 `NO_SIGNAL`。零模型前向、零训练、零新 Run。[核查与处置](next_core_direction_20261106/TIME_REVERSAL_ODD_ZERO_TRAINING_20261109.md)

### U31 晚窗有符号时滞正式关闭，登记主动未见类拒识设计

依用户指令登记 `MVCNET-TIME-REVERSAL-ODD-DIRECTION-CLOSE-20261109`。保留 403 条、806 窗的谱不变量通过、4/7 人同向及两项置换门失败。保留眼电回放关闭、Subject K 核查、既有关闭与 `sealfix2` 的 `NO_SIGNAL`。

`MVCNET-ACTIVE-UNSUPPORTED-COMMAND-DESIGN-20261109` 仅设计。固定左右手原型读出，比较到两类线段的距离、置信间隔、最近原型和中点距离。在六名源被试 T 定原型、六人 E 的已知手类定阈值，留出人 E 的同 run 双脚和舌只作最后评分。元数据已核对 33 个合格 E run、329 条候选试次。直接近邻已有同类被试内配置，论文投入以严格留人和风险匹配下对简单强对照的实质增量为前提。未做效果统计、拟合或派发。[设计与停止门](next_core_direction_20261106/ACTIVE_UNSUPPORTED_COMMAND_DESIGN_20261109.md)

### U32 主动未见类拒识因接收率不可比且无原型增量停止

登记 `MVCNET-ACTIVE-UNSUPPORTED-COMMAND-ZERO-TRAINING-20261109`。33 个同 run 匹配、329 条留出评分均闭合。其余六人仅以 T 左右手定原型、E 左右手定阈值。线段规则与最近原型和中点的 329 条接收决定完全一致。相对置信间隔，已知接收率相差 12.41 个百分点，超过 5 点门；双脚误放改善 7.82 点，舌 18.28 点，两类各仅 4/7 人改善。按 `STOP_REJECTION_GATE` 停止固定方案，未进入设计复核。保留晚窗时滞、眼电回放关闭、Subject K 核查及 `sealfix2` 的 `NO_SIGNAL`。零模型前向、零训练、零新 Run。[核查与处置](next_core_direction_20261106/ACTIVE_UNSUPPORTED_COMMAND_ZERO_TRAINING_20261109.md)

### U33 主动未见类拒识正式关闭，登记错误序列核查设计

依用户指令登记 `MVCNET-ACTIVE-UNSUPPORTED-COMMAND-DIRECTION-CLOSE-20261109`。保留 33 个匹配 run、329 条试次，线段与最近原型和中点的接收决定全部一致，已知类接收率相差 12.41 个百分点，高于 5 点可比性门。晚窗时滞和眼电回放的关闭、Subject K 核查和 `sealfix2` 的 `NO_SIGNAL` 均保留。[拒识核查](next_core_direction_20261106/ACTIVE_UNSUPPORTED_COMMAND_ZERO_TRAINING_20261109.md)

`MVCNET-ERROR-SEQUENCE-CLUSTERING-DESIGN-20261109` 仅设计。已核对 5,184 条、108 个 run 的时间索引，全部 5,076 对紧邻试次的连续 FIR 输入范围有重叠，全部 4,968 对隔一试次的输入范围分离。拟用两个不重叠子序列及保持真实类别、run 半段错误率和四模型同试次共错的联合置换，判断错误聚集是否值得独立复核。只做了资产和元数据核查，未计算错误序列效果或启动 Run。[设计与停止门](next_core_direction_20261106/ERROR_SEQUENCE_CLUSTERING_DESIGN_20261109.md)

### U34 错误序列聚集在偶数主检验与奇数复核的冻结门停止

登记 `MVCNET-ERROR-SEQUENCE-CLUSTERING-ZERO-TRAINING-20261109`。`sealfix2` 核心包、36 项已封存检查和 5,184 条唯一试次通过身份核验。两组各 2,376 对，分别完成 10,000 次分层联合置换。偶数 CE-73 过 P95，CE-74 和跨重复未过。两次 CE 平均超额 +0.611 个百分点，低于 +1 点门，7/9 人平均超额为正。奇数两次 CE 和跨重复超额均为负。按 `STOP_ERROR_SEQUENCE_CLUSTERING_GATE` 停止固定方向，未进入设计复核。既有关闭结论、Subject K 核查和 `sealfix2` 的 `NO_SIGNAL` 保留。零模型重放、零训练、零新 Run。[核查与处置](next_core_direction_20261106/ERROR_SEQUENCE_CLUSTERING_ZERO_TRAINING_20261109.md)

### U35 错误序列聚集正式关闭，登记 μ 与高 β 时间对齐设计

依用户指令登记 `MVCNET-ERROR-SEQUENCE-CLUSTERING-DIRECTION-CLOSE-20261109`。偶数主检验仅 CE-73 过 P95，两次 CE 平均超额 +0.611 个百分点，7/9 人同向。跨重复未过 P95，奇数复核三项均为负。保留既有关闭结论、Subject K 核查及 `sealfix2` 的 `NO_SIGNAL`。[错误序列核查](next_core_direction_20261106/ERROR_SEQUENCE_CLUSTERING_ZERO_TRAINING_20261109.md)

`MVCNET-MU-HIGH-BETA-ALIGNMENT-DESIGN-20261109` 仅设计。使用两个频带的完整功率轨迹、同维的试次内错位，以及同 run 同类、保留 cue 时钟的跨试次重配，检验同一试次 μ 与高 β 对齐的跨人 T→E 增量。固定折核对七名源被试、812 条 T 左右手和 E 的 20 个匹配 run、129 条。只做了元数据核对，未读取波形计算效应或启动 Run。[新方向与停止门](next_core_direction_20261106/MU_HIGH_BETA_ALIGNMENT_DESIGN_20261109.md)

### U36 μ 与高 β 时间对齐在冻结效果门停止

登记 `MVCNET-MU-HIGH-BETA-ALIGNMENT-ZERO-TRAINING-20261109`。14 份原始 MAT、812 条 T 源侧左右手、20 个 E 匹配 run 的 129 条评分试次和合成对照均通过。候选相对完整功率、同维错位的 AUROC 增量分别为 +0.011905、+0.003720，均低于 +0.03，只有 1/7 人同时改善。候选定向类差 −0.004130，低于 10,000 次同 run 同类重配的 P95 +0.191864。四项效果门均失败，按 `STOP_MU_HIGH_BETA_ALIGNMENT_GATE` 停止，未进入设计复核。保留既有关闭结论、Subject K 核查和 `sealfix2` 的 `NO_SIGNAL`。零模型重放、零训练、零新 Run。[核查与处置](next_core_direction_20261106/MU_HIGH_BETA_ALIGNMENT_ZERO_TRAINING_20261109.md)

### U37 μ 与高 β 时间对齐正式关闭，登记源验证 FIR 支撑隔离设计

依用户指令登记 `MVCNET-MU-HIGH-BETA-ALIGNMENT-DIRECTION-CLOSE-20261109`。保留 14 份 MAT 和合成对照通过、20 个匹配 run 和 129 条试次。两项 AUROC 增量未达 0.03，仅 1/7 人同时改善，定向类差未过重配 P95。既有关闭结论、Subject K 核查和 `sealfix2` 的 `NO_SIGNAL` 均保留。[核查与处置](next_core_direction_20261106/MU_HIGH_BETA_ALIGNMENT_ZERO_TRAINING_20261109.md)

`MVCNET-FIR-SUPPORT-DISJOINT-SELECTION-DESIGN-20261109` 仅设计。18 个冻结折每折有 761–772 条源验证试次与训练窗共享扩展 FIR 支撑，源侧与独立检查被试无同 run 相交。拟固定训练输入和验证身份，只限制验证可读原始采样。A03/A04 T/E 四份文件有 222 条验证，其中 67 条可做单侧相交的等长度反侧对照。仅完成元数据和文件可用性核对，未运行波形探针、复算选模、重放模型或启动训练。[新设计与停止门](next_core_direction_20261106/FIR_SUPPORT_DISJOINT_SELECTION_DESIGN_20261109.md)

### U38 源侧验证 FIR 支撑隔离过零训练门，设计复核暂缓选模比较

登记 `MVCNET-FIR-SUPPORT-DISJOINT-SELECTION-ZERO-TRAINING-20261109`。四份 MAT、222 条冻结验证裁剪及物理支撑通过，8 条实数据脉冲探针的共享响应在隔离验证端归零。67 条配对中，A03T、A03E、A04T、A04E 的隔离改变量中位数分别 8.81%、5.38%、3.97%、4.68%，全部通过 1% 门。进入只读设计复核。冻结 runner 与已验收清单只支持每次拟合的最终选中权重，缺可复用的多 epoch 轨迹，处置 `HOLD_MISSING_REUSABLE_EPOCH_TRAJECTORY`，不推断选模或准确率变化。保留既有关闭结论、Subject K 核查和 `sealfix2` 的 `NO_SIGNAL`。未重放模型、训练或建立 Run。[核查与设计复核](next_core_direction_20261106/FIR_SUPPORT_DISJOINT_SELECTION_ZERO_TRAINING_20261109.md)

### U39 源侧验证 FIR 支撑隔离正式关闭

依用户指令登记 `MVCNET-FIR-SUPPORT-DISJOINT-SELECTION-DIRECTION-CLOSE-20261109`。保留 222 条冻结数组重放、物理支撑与 8 条实数据脉冲核查通过，以及 67 对中四份记录的隔离中位数 8.81%、5.38%、3.97%、4.68%。设计复核确认 `sealfix2` 现有产物缺可复用的多 epoch 轨迹，选模和独立检查准确率的变化仍未检验。保留 Subject K 核查、既有关闭和 `sealfix2` 的 `NO_SIGNAL`。本次关闭不激活新方向。[核查与设计复核](next_core_direction_20261106/FIR_SUPPORT_DISJOINT_SELECTION_ZERO_TRAINING_20261109.md)

### U40 VCN-P1 保留为阶段性创新，登记逐视图 affine 可迁移性判据

`MVCNET-VCN-VIEW-AFFINE-TRANSFER-DESIGN-20261109` 仅设计。历史 P1−B0 和参数匹配 N1 支持仍限原协议。原最小结构结论 `STOP_AS_NEW_MAIN_INNOVATION` 保留。本次筛的是 raw 已独立后，增强 affine 的额外方向是否跨人和记录可复用，能否经共享主干更有利于另一会话的 raw 目标。关键对照为按增强视图分别计算批统计的 raw／增强双 BN、冻结方向后五种同范数错配和三路相同视图。固定折元数据支持七名源被试的 896 条 T 探针与 400 条 E 源验证。单份封存 CE 权重只由历史清单定位，本地精简包无该文件，未验证远端可达性。当前不确立新主线，未执行零训练探针或启动 Run。既有关闭结论与 `sealfix2` 的 `NO_SIGNAL` 均保留。[筛查设计与停止门](next_core_direction_20261106/VCN_VIEW_AFFINE_TRANSFER_DESIGN_20261109.md)
