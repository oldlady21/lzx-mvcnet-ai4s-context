# VBR-R1：VCN-P1 视图内训练归一化校正

2026-10-24；Program=mvcnet；chain=mvcnet-transaction-v3.2。
阶段：DISCOVER_NEXT / DESIGN_ONLY；候选设计，不是冻结合同或执行授权。

## 1. 问题与选择理由

VCN-P1 把不同增强视图的 BN 状态分开，但同一视图训练仍依赖当前 batch 统计，raw 部署则用 running 统计。下一问是：**在保持四视图路由、affine和原损失不变时，训练中对这个差异作有界校正，能否获得稳定的 raw 部署增量？** 该差异存在于算子定义中；其危害和可修复性是未证实假设。

引用 [CURRENT_UNDERSTANDING](../CURRENT_UNDERSTANDING.md) U1–U4：分头的平均正信号未通过稳定性/覆盖门，源内可读信息也未稳定转移，因此不再把下一轮放在读出容量、teacher或源类对齐上。对 VCN 最直接的剩余问题是归一化路径本身。按用户要求优先形成这一辅助研究设计；VCN 原创性和完整方法对照仍需补齐，辅助涨点不会自动解决主创新不足。

与继续分头调参相比，本方案改变训练 normalization 的前向与梯度，而不改头/CE分工；与单纯增加P1 seeds相比，它提出新问题。先保存完整设计，不以旧campaign结余自动支付。若机制差异不足以支持论文贡献，即使有效也只作为训练配方适配报告。

## 2. 实现身份与数学定义（拟议单配置）

intervention_id=`vcn-view-batch-renorm-v01`；module建议标识=`vbr_r1`，尚未注册/实现。沿 MVCNetAdapter → `model.args.ai4s_mvcnet_module` → AI4S gateway 接入，只改 stem 中的归一化，不同时打开旧 vcn_p1_variant。P0 保留 P1 四套独立 running mean/variance 和 affine；共用卷积、netC、增强标签、CVC/CMC及训练配方不变。只用source-training数据更新统计，validation/check/目标均eval，不做目标batch适配。

对活动 view v 的每通道，在 N×T 轴计算 batch mean μ_b 与 biased variance s²_b。取**当前forward更新前**该bank的 running mean m_v、running variance q_v：

    σ_b = sqrt(s²_b + eps), σ_v = sqrt(q_v + eps)
    r = stopgrad(clip(σ_b / σ_v, 1/r_max, r_max))
    d = stopgrad(clip((μ_b − m_v) / σ_v, −d_max, d_max))
    y = γ_v * (((x − μ_b) / σ_b) * r + d) + β_v

running mean/variance 的更新、momentum、eps、unbiased running variance约定和每次合法forward的计数均保持原PyTorch BN语义；r/d无梯度，batch mean/variance仍按BN反传。不能在原BN执行后再用已更新统计计算校正。该实现使用running variance，不声称逐字复现原论文的running standard deviation算法。eval为原BN running-stat公式，仅raw银行参与部署。

N0/B0沿宿主实际多视图forward顺序逐次更新同一共享bank：第k次调用取该次调用前的状态，允许它已包含第k−1次视图的合法更新；不统一取optimizer步开始的快照。P0/S0只更新当前活动bank，调用顺序与次数也不变。实际runner的调用序列须在资格检查中记录，不能假定每view恰好调用一次。T固定为计划的5000次optimizer更新，t随checkpoint保存/恢复，不以早停选中的epoch重算日程。

单个配置草案：100 epoch×50 optimizer steps、B64、FP32及原Adam等训练配方；所有臂完全一致。以optimizer进度 t/T 计日程（不是某bank调用次数）：前10%保持r_max=1,d_max=0；10%至50%线性升到r_max=3,d_max=5；之后固定。最终上下限借鉴原BRN，进度日程是本任务事前适配，未按效果选择，不做日程搜索。关闭增量走原BN实现，要求恢复完整宿主；不能只用r=1,d=0的表面公式代替数值/状态/梯度parity。

## 3. 最小关键比较：一个2×2设计

|臂|视图归一化状态|有界校正|回答的问题|
|---|---|---|---|
|B0|规范MVCNet的共享BN|无|原baseline|
|S0|原VCN-P1四bank|无|当前支架|
|N0|与B0相同共享BN|有，同一校正规则/日程|通用renorm是否已经足够|
|P0|与S0相同四bank|有，每view读取自己的状态|VCN宿主增量|

主比较 P0−S0；完整适配 P0−B0；通用效果 N0−B0；交互量 I=(P0−S0)−(N0−B0)。N0是本轮新语义，不是历史P1的N1。P0/S0、N0/B0各自训练参数量相同，无新head、teacher或第二套encoder前向；四臂本身的BN参数差异按宿主原定义保留。若I没有支持，不称VCN特异协同；P0优于S0但N0同样好，应优先考虑更简单通用训练适配，不把N0自动提升为新获准候选。

四臂比只跑S0/P0多成本，但能在本轮区分“修补VCN特有局限”和“一般归一化收益”。不追加C0，因为当前关键竞争解释不是新增容量。保持同seed初值、数据顺序、增强RNG、训练步数及source-validation选点规则配对；单独bank初始化按现有P1规则，不跨arm复用训练checkpoint。

## 4. 数据、成本与判定草案

延续七源developmental框架的角色，不读取外层目标结果：每个outer目标完全排除，另一个source subject封存作check，余7源中training/validation按原合法规则分开。9个outer身份×2个配对随机重复×4臂=72次完整拟合；72个末端check是评估，不是额外拟合。该结构仍是已暴露开发数据，不是普通八源LOSO或最终评价。具体seeds、split manifest/SHA、check被试映射、checkpoint选择与tie规则须在新冻结配置中明确；本轮不虚构已封存资产，也不沿用历史成绩作新配对对照。

目前零次新训练/零次EEG前向。训练代码、生产GPU资格、冻结合同、预算、执行器目的地和恢复额度均未建立。最近增强读出69次fresh拟合加3次合格复用的完整Run实际执行132:26:58，可作为成本风险提醒；不能把它当成本方案精确预计，更不能再直接采用早期11.86h/30h估计。新方法无额外encoder调用，但真实资源竞争、显存与实现开销未测；预算应在部署方案中估计、明确批准。旧Run余额不转入此设计。

以下为前瞻工作标准建议，未冻结、不能修改旧判定：

|判定|单位/阈值/依据|后果|
|---|---|---|
|完整性和作用路径|72拟合/72check闭集、真实产物/哈希、预测配对重放、正确bank更新、无eval更新/目标访问|缺失/错配/非有限或算子未生效：INVALID/未完成，不能按零效果聚合|
|支架效用与稳定性|P0−S0 accuracy_pp：被试先平均2seed再跨9被试，均值≥+0.5pp、两seed均值>0、≥6/9被试>0；平均CE不恶化|仅全部满足才形成developmental SCREEN_POSITIVE；阈值沿既有开发工作口径，不是功效计算或临床阈值|
|风险|P0−S0 被试均值≤−5pp的数量≤1|超限停止，不能靠平均收益抵消|
|归因与机制|完整报告P0−B0、N0−B0、交互I、seed/被试差和被试bootstrap区间；均为预登记解释指标|无交互支持则不称VCN特异协同；仅表征差异下降不证明它造成ACC收益|

训练内按固定间隔记录r/d截断比例、|r−1|/|d|和同一activation上校正前后与running-stat公式的归一化差异，不另开科学诊断阶段、不中途读取效果选配置。该差异在同一activation上比较，不切整个模型train/eval混入dropout/上游变化。它是算子作用证据，不是因果证明；不因其大小调日程。只在完整末端intake后按预登记比较读效果。

“算子未生效”仅指配置被忽略、错误路由、非恒等合成输入却没有预期校正等可验证实现错误；r≈1、d≈0或clip比例为0本身不是INVALID，可能只是batch与running统计接近。资格阶段用合成的已知非恒等输入检查实现与解析公式相符，数值容差依dtype/真实kernel事前固定；不能用真实效应大小重新定义工程有效性。可在后续授权中用合格P1 checkpoint的source-training批次、内存副本和固定batch清单进行不更新权重/不读标签效果的作用量检查，辅助决定是否值得投入；成熟checkpoint近恒等不能证伪早期训练作用，也不自动构成放弃整个机制的科学门。本轮未读取这些产物或运行该检查。

停止：固定实例无稳定效用或风险超限，关闭该日程/配置，不补seed、加校正强度或换bank分组救援。完整正筛查也只比较是否值得做独立复核/机制控制，不自动执行；若只产生通用BRN收益，记录适配发现并重新判断论文价值。已有证据不足以把任何分支写成正式Claim。

## 5. 执行前资格与旧边界

需要覆盖的最小工程验证：关闭增量恢复B0/S0的forward、loss、梯度、running state及计数；r/d detach但BN主路径梯度保留；只更新活动bank；读取pre-update状态；验证/检查raw-only且统计不动；optimizer步进日程跨bank一致；断点恢复计数无漂移；完整真实runner/checkpoint导出/分析器闭环及生产shape GPU smoke。当前均NOT_RUN，不将源码推理写成通过。

最近一步只建议design_qualification（实现身份、原生parity、作用量及生产资源/预算核查），不直接投入72次拟合。72次是资格后拟议完整研究矩阵；为保持初筛解释完整，本稿不采用事后见P0−S0收益再挑选是否加入通用控制的方案。若成本不能接受，可另行前瞻设计分阶段比较及停止规则，但不能把缺少N0/B0的结果称为互补归因。

旧augmentation-readout、head-meta、SCMA、SED等精确关闭不变。父级old_moment_only_route不重开：本候选不做旧moment诊断再晋级、不改变P1 bank划分、不从其历史结果选阈值；它是在原P1状态布局内引入新的训练校正算子。仍需设计外审核对此边界。旧Run在State仍存在，真正派发前必须按原v3.1绑定解决其控制器/intake收尾，不以激活或新设计覆盖。

## 6. 近邻与原创性边界

Ioffe的 [Batch Renormalization, NeurIPS 2017](https://proceedings.neurips.cc/paper/2017/file/c54e7837e0cd0ced286cb5995327d1ab-Paper.pdf) 已提出用停止梯度的有界校正减少batch统计与推理统计差异。本方案沿用其机制，主要适配为P1的view bank、现有variance状态及统一步进日程，不声称原创renorm。

[SAFE, 2026-01预印本](https://arxiv.org/abs/2601.05789) 已在联邦EEG中采用local batch-specific normalization；其设置与这里source-only、raw-only部署不同，不能据名称差异证明首次。[DSBN, CVPR2019](https://openaccess.thecvf.com/content_CVPR_2019/html/Chang_Domain-Specific_Batch_Normalization_for_Unsupervised_Domain_Adaptation_CVPR_2019_paper.html) 是多bank归一化的直接近邻，目标参与UDA不能迁入本设计。2026-10-24的有限检索用于确认已有机制与近期方向，不构成完整查新。当前method_originality=known_mechanism_adaptation，novelty=unverified/likely_limited。

## 7. 交接

question_id=`vcn-within-view-train-inference-normalization`；data_scope_id=`prospective-seven-source-developmental-vbr-r1-unfrozen`；contrast=`P0-S0`，关键控制为2×2交互；next_stage=`design_qualification`；next_transition=`DISCOVER_NEXT`；run_action=`DESIGN_ONLY`；execution_authorized=false。

source_evidence_ids：VCN-P1-S8-SCOPED-2026-08-28（身份/范围）、VCN-V0-1-PARENT-BOUNDARY-2026-08-21（禁闭边界），以及CURRENT_UNDERSTANDING U2/U3/U4列出的精确AI4S evidence IDs。本brief和总览的直接来源/SHA见SOURCE_MANIFEST.json；不建立正式评估计划或Claim Ledger。


### 2026-10-24 执行资格补充（冻结前）

用户已授权200h累计执行与Lab10独立Run。仅四个短资格任务不足以测量本科学矩阵所需并发，因此技术scope采用四臂各两个同配置资格实例：seed69、outer0、5×50步、B64/FP32，副本显式后缀`-replica1`并独立调度/计时/遥测。它们只使用原source训练与source验证，不读取held check效能，不新增科学候选或seed，不进入科学效应均值。任务总数154=source-import1+资格8+科学train72+seal1+check72；科学门与72/72矩阵不变。

技术阶段累计执行仍最多6h（以Transaction真实计时排除纯资源等待，不把并发进程小时当累计耗时）。成本从真实五epoch完整原生训练/源验证/clone和source576 raw消费外推，记录串行及并发场景；八资格实例所得有效并发只作为同目标机器的条件预测依据，并计入科学调度首轮四家族渐进启动成本与10%余量。未来服务能力不保证，实际SCI188h与campaign200h硬上限仍生效；观测不足或预计超限时只记技术/成本未通过，不伪装为科学NO_SIGNAL。


### 冻结前预算收敛与外审处置

Grok最终只读复核返回ACCEPT_WITH_LIMITS，指出SCI固定188h可能侵占profile按预测预留的分析时长。为落实总200h，最终整个Transaction执行上限为188h（含TECH与SCI），包外验收/分析硬预留6h；注册/启动仍要求至少194h余额，实际准备时间自04:35:51 UTC计入，不重置，因此最多6h前期准备可进入本次188h Transaction。TECH最多6h，SCI节点最多188h，但实际科学阶段上限同时受Transaction剩余与200h账户扣6h约束。6/188/6原计划保留为设计历史，实际不把这三项再叠加到前期准备之上。该修正没有降低100epoch/5000steps或缩小72/72矩阵；成本不合格即0科学派发。最终科学/资源/成本规则以唯一冻结Transaction为准。
