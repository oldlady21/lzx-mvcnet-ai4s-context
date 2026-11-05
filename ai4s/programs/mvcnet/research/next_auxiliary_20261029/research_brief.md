# VCN-P1 下一辅助方向：ACM-R1 设计备忘录

> **执行授权更新（2026-11-04，独立42h）**：用户已选择ACM-R1一次性四臂探索。当前生产准备，先冻结判读并完成Lab10资格/成本门，二者通过后同一Run启动72fit+72check；不调参救援、不转用XSN余额。独立累计起点2026-11-04T13:54Z，42h总额，Transaction32h、外留6h收尾。此前36h为重评商议目标，现以用户42h授权为准；下文各设计/审查历史时点保留。XSN关闭与全部历史不变。授权记录见[ACM执行授权](../acm-r1/execution/AUTHORIZATION.json)。


> **2026-11-04最新重评（DESIGN_ONLY）**：XSN已正式NO_SIGNAL关闭；ACM保留为有条件值得一次四臂开发探索的候选，尚未生产接线/冻结/分配预算。最新依据、对照和成本以[收尾后重评](reassessment_20261104/REPORT.md)及第11节为准。以下2026-10-29内容保留其历史时点，未覆盖原Grok HOLD意见。


2026-10-29；Program `mvcnet`，新设计链 `mvcnet-transaction-v3.3`。状态 **AUDIT_ONLY / 限定合成梯度资格通过 / 未接入生产、未冻结、未授权训练**。

最新进展：用户授权的[隔离梯度与非退化审计](gradient_audit_20261029/REPORT.md)已完成，53项主检查及两个原生实例的保存张量复算通过。S0原生一致、P0/N0首步F/C及Adam精确等价；交换A引起CVC→F梯度差，固定A或断开跨分支耦合时差值为零。主Agent现判断为有条件值得一次四臂开发探索；效用未知，尚无生产batch64/Lab10成本资格和独立预算，当前不派发训练。以下查新/反驳段保留原时点，Grok原HOLD意见未改写。


后续用户已授权限定查新、独立反驳、对照与成本修订，现已完成。当前建议是先做零训练图资格审计，暂不投入72次完整fit。详见[本轮完整结论](review_20261029/NOVELTY_AND_REBUTTAL.md)及[关键对照与成本](review_20261029/CONTROL_COST_ANALYSIS.md)。初稿逐字保存在[外审前版本](review_20261029/research_brief_before_review.md)，不覆盖原始设计与Grok原意见。

本次用户要求：等待XSN-R1时，并行设计下一方向，给出选择依据和关键对照，不启动新训练。本记录不转换活动campaign，不占用训练seed，不修改当前XSN合同、预算、源码或Run。XSN没有作为已完成科学结果输入本次选型。

## 1. 建议与实际做法

优先准备 **ACM-R1：以分类主干为参照的单向跨模型对比（CMC）**。它深化此前候选池中的 `cmc_stop_gradient_cnn_training_only`，不是凭空新增一条已获支持路线。

1. 保留VCN-P1四个BN bank、四视图、分类头、CE、CVC和raw部署。
2. 在CMC调用处，把CNN特征作为本次反向不更新的参照；Transformer及原投影器继续接受CMC梯度。
3. CNN仍由分类CE和原CVC训练；不冻结CNN、BN或整个辅助分支，不增加教师、EMA、预测头或损失权重搜索。
4. 用四个配对条件区分：双向CMC、只更新辅助分支、只更新CNN、CMC两侧均不反传。

拟回答：**对当前“CNN负责分类、Transformer只辅助训练”的实现，CMC的两侧是否需要不同的更新职责？** 潜在价值是为VCN统计隔离支架补充一条可解释的训练约束；效用、方向必要性及VCN特异互补均未证实。当前值得把设计和反驳准备完整，不建议直接排入训练。

## 2. 基模事实核对：论文与当前实现必须分开

原文v4描述双分支融合分类；本地阅读材料 `refs/paper-cards/mvcnet/MVCNet-note.md` 也按该论文解释raw推理保留两分支。**这不能直接用作当前VCN-P1执行路径的描述。**

本次直接核对：

- `src/baselines/mvcnet.py:212` 的forward为 `classifier(features(raw))`，不调用encoder/projector；训练中的四视图分类也只消费CNN特征。
- 真实原生训练参考 `_work/augmentation_readout_source_r1/ai4s/programs/mvcnet/research/augmentation_readout/train.py:28` 的 `NativeModel.forward` 同样是 `netC(netF(x))[1]`；其 `_native_objective`（第183行起）独立生成encoder/projector表示。
- `xsn-r1/execution/native_runner.py:114` 的构造沿用该NativeModel与VCN-P1，check可不构造辅助模块。本轮仅静态阅读源码，没有读取活动Run的中间效果。
- 原生CVC把CNN与Transformer投影沿**batch轴**拼接，再比较raw和各增强；CMC把四视图分别拼接，对CNN与投影表示做NT-Xent。CVC仍含跨分支的负项耦合，不能简化为两条彼此独立的视图损失。
- `src/models/proposed/vcn_p1_mvcnet.py` 的P1只分离stem BN running state与affine；不能把它解释成分离了损失梯度。

因此以下设计绑定本地release-owned IFNet/VCN-P1路径，不绑定论文双分支分类图，也不把两种实现的绝对成绩混作配对证据。L3的基模阅读核对项不覆盖或修改历史执行合同。

## 3. 历史证据如何改变选择

| 直接证据/来源 | 已有认识与边界 | 对本设计的影响 |
|---|---|---|
| `VCN-P1-S8-SCOPED-2026-08-28`；Program的SATURATED登记 | 支架有范围限定的效用/归因支持；不是主创新资格或跨数据结论；同一稳定性seed扩展已经饱和 | 沿用宿主身份，不再靠给P1加seed形成新研究；旧LOSO效应不用于选本次配置 |
| `AI4S-MVCNET-mvcnet-vbr-r1-screen-lab10-r1-repair3`；`vbr-r1/execution/repair3/analysis_20261029/REPORT.md` | 固定VBR P0−S0约+0.019pp，seed方向/覆盖等冻结门失败，NO_SIGNAL | 不再换BRN日程、限幅或seed；转向不同的梯度作用路径，保留其精确关闭 |
| `AI4S-MVCNET-mvcnet-augmentation-readout-screen-lab10-r1-resume69-repair2`；研究总览U2 | 分离增强读出未形成稳定宿主增量 | 不新增视图分类头，不把共享头冲突当作已证明前提 |
| `AI4S-MVCNET-mvcnet-contrastive-factorial-diagnostic-lab10-r3`；`contrastive_factorial/PLAN.md`、`REPORT.md` | CVC/CMC整项开关未建立稳定效应；也没有证明删掉CMC更好或等效 | 完全关闭CMC必须是本次对照；旧开关臂不能直接充作新配对结果；新问题只细分CMC的反向职责 |
| `AI4S-MVCNET-mvcnet-vogr-d0-diagnostic-lab10-r1`；`VOGR_D0_PROTOCOL.md`及对应decision | source-only梯度/虚拟SGD诊断为INSUFFICIENT_DIAGNOSTIC_SUPPORT，没有训练路由 | 不以“CMC梯度有害已证明”开题，也不先增加动态梯度门；固定方向比按梯度相似度调权更少选择自由度 |
| `AI4S-MVCNET-mvcnet-scva-cvc-s0p0-screen-lab10-v0.2`；归档合同 | CVC共享rank-128投影精确实例NO_SIGNAL | 不优先添加另一个低维投影；只换到CMC不足以证明与旧假设独立 |
| `IPB-V0-1-NEGATIVE-2026-08-20`、`MPRF-P1-R10-NEGATIVE-2026-08-24` | 已关闭推理桥接与融合实例 | 本设计不改变推理路径、不恢复桥接或双分支分类融合 |
| `AI4S-MVCNET-mvcnet-cross-objective-collision-screen-lab10-r8`；`cross_objective_collision/PLAN.md` | 旧实验删除同trial跨目标直接正负碰撞边，精确NO_SIGNAL | ACM保持原正负集合，区别在梯度接收端；不借换名称重跑删边 |
| XSN-R1设计/冻结身份（不是结果） | 当前在检验保标签CVC中的跨来源同类负项选择 | 不叠加XSN；新方向先以未加XSN的S0为宿主，不用运行中或未来结果暗中选方向 |

上述区别说明它不是原实例的相同配置重试，不等于已经有证据证明该机制有效。旧负结果不推广为全部归一化、头、投影、梯度或对比学习无效。

## 4. 四臂定义：只改变CMC的反向权限

令 `F` 为四视图CNN特征拼接，`A` 为四视图Transformer→原投影器输出拼接，`sg` 为仅在本次CMC消费处停止梯度。使用原生NT-Xent的全部anchor、正负项、温度及归一化：

`L = CE_four_views + 0.1 * CVC_native + 0.1 * NTXent(F_gate, A_gate)`。

0.1为原生配方沿用建议，不是调优结果；未来精确配置仍须冻结。这里的“单向”指梯度权限，**不是**把双向NT-Xent改成一半anchor的损失。

| 暂用arm | CMC输入 | CMC更新CNN | CMC更新encoder/projector | 回答的问题 |
|---|---|---:|---:|---|
| S0 | F, A | 是 | 是 | 原生VCN-P1宿主 |
| P0（拟议ACM） | sg(F), A | 否 | 是 | 分类主干作参照、辅助分支追随是否有益 |
| R0 | F, sg(A) | 是 | 否 | 反向职责是否同样有效或更好 |
| N0 | sg(F), sg(A) | 否 | 否 | 完全去除CMC训练作用能否解释收益 |

名称仅属于本设计，不能覆盖XSN同名arm的含义。N0仍计算同一CMC前向数值，只使它不贡献梯度；参数更新应与 `lambda_cmc=0` 等价，不靠删辅助forward来节省成本。各臂的CE、CVC完整保留。

在同一参数、输入和随机状态下，四臂CMC前向值应相同。CMC独立梯度分解必须得到表中结果；P0与N0的CNN/分类头**第一步**总梯度及Adam更新应相同，encoder/projector不同。之后若两臂CNN轨迹不同，只能通过保留的CVC等训练耦合传播，不能把detach描述成整个训练中CNN与辅助分支隔绝。

关键边界：P0停止的是**全部四视图F在CMC中的梯度**（包括其上游共享卷积和对应BN affine），不是只停止raw视图或只停止共享卷积。BN状态仍由原forward正常更新；CE/CVC对这些参数照常反传。只在CMC调用处使用非原位detach，禁止把供CE/CVC共用的features整体detach。

## 5. 关键比较与能推翻方案的结果

1. **P0−S0：宿主增量。** 没有稳定效用则不作为辅助创新；控制臂优势不能代替它。
2. **P0−N0：是否只是关闭CMC。** 这是最重要的反驳。如果没有可辨识增量，就没有依据保留“辅助分支追随”的新增机制；不能把相近均值叫等效，也不能自动改推N0。
3. **P0−R0：职责方向是否必要。** 若未建立方向差异，不支持“分类主干应该作参照”的特定解释；不得看结果后选择R0为新主方法。

四臂构成CMC两侧梯度开关的2×2，记录 `S0−P0−R0+N0` 可帮助解释，但它是梯度权限的交互，**不是VCN与ACM的协同**。欲声称VCN特异互补，另需前瞻B0/B0+ACM与S0/P0的配对交互，不能借用旧B0成绩。本轮不把这些额外臂塞入核心矩阵。

与一般减弱正则化相比，N0直接覆盖“CNN侧CMC完全去掉”这个最强简单解释；四臂并不穷尽所有中间权重、梯度范数或有效优化强度解释。因此首轮只判断固定职责分配的开发效用，不宣称已经证明梯度冲突、信息互补或优于所有权重调节。将来若论文主张需要排除一般调权，须事前公平给S0/完整方法同等配置机会，而非事后添加一个容易击败的权重对照。

建议结果分支（不是已冻结门）：

| 结果 | 应有处置 |
|---|---|
| P0对S0未建立稳定效用，或个体风险不可接受 | 关闭这一个固定ACM实例；不切视图、改方向、加权重或seed救援 |
| P0相对S0有信号，但未区分N0/R0 | 保留有限效用记录，停止方向必要性故事；不自动支付确认预算 |
| P0同时优于S0/N0/R0且风险可接受 | 只支持开发阶段继续投入的判断；再比较主方法证据价值与强近邻，不直接发布Claim |
| 仅CMC值、梯度几何或源内CE改变 | 记录解释性观测；不替代raw分类效用 |
| 技术上detach边界、原生parity或优化器路径不合格 | 工程未就绪，不记科学NO_SIGNAL |

未来若进入实施，建议效用最小门以P0−S0均值≥0.5pp、两seed均正、被试内合并后≥6/9正向、CE不恶化及预定个体风险为起草依据；方向比较可从≥0.25pp及相同稳定性要求起草。它们只是延续已有开发决策尺度的建议，不是显著性/等效界值，也未绑定可执行分类器；不得由本文件直接派发。冻结前须确定风险阈值、区间、必要比较及专属判读映射，不能自动调用XSN五臂分类器或为旧accuracy分类器补造字段。

## 6. 初稿有限文献核对与创新边界（后续已补限定查新）

初稿做了直接一手近邻与含2025/2026年份的有限检索，当时尚未做独立反驳。后续限定查新和真实Grok反驳已完成，新增最强近邻为ADM（ICML2025）、按部署角色组织的不对称蒸馏（NPL2024）、ReC-TTT及AF。结论为PROCEED WITH CAUTION，5/10，且HOLD_TRAINING；完整来源与限制见本轮报告。以下保留初稿基础近邻，未找到同式命中仍不证明首次。

- [SimSiam](https://arxiv.org/abs/2011.10566)：stop-gradient已是成熟手段，不能作为本方法的原创算子；其无负样本防塌缩结论不能直接移到有CE、NT-Xent及异构分支的这里。
- [On the Importance of Asymmetry for Siamese Representation Learning, CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/html/Wang_On_the_Importance_of_Asymmetry_for_Siamese_Representation_Learning_CVPR_2022_paper.html)：已有对不对称编码器职责的研究。本设计不同在本地分类/辅助角色及CMC单项反向权限，但这一宿主区别本身不足以证明新颖。
- [Adapting Auxiliary Losses Using Gradient Similarity](https://arxiv.org/abs/1812.02224) 与 [PCGrad](https://arxiv.org/abs/2001.06782)：辅助目标与主任务梯度协调已有成熟近邻。本轮不采用动态相似度门或梯度投影，也不把旧VOGR未支持的冲突当事实。
- [Implicit Contrastive Representation Learning with Guided Stop-gradient](https://arxiv.org/abs/2503.09058)：arXiv于2025发布，但条目注明NeurIPS 2023，不能误报成2025新会议；说明停止梯度的细化也已有先例。
- [Dual Perspectives on Non-Contrastive Self-Supervised Learning](https://arxiv.org/abs/2507.01028) v3（2026-02-05）：其理论范围为非对比学习/所研究动力系统；提示相同前向标量不意味着同一优化过程，不为本设计提供泛化保证。
- [MVCNet原文v4](https://arxiv.org/abs/2502.17482v4)：用于原论文身份与实现差异，不把原论文完整模型结果当作本地ACM支持。

潜在贡献须是“在明确的训练辅助分支条件下，职责分配带来可归因的稳定效用或清楚的适用条件”。仅加detach、仅一次均值涨点、仅与弱对照比较均不够。目前新颖性薄、效用未知；更准确的定位是**有清楚反驳条件的辅助机制候选**。

## 7. 备选与暂选依据

完整合并池见 `CANDIDATE_POOL.json`。已有候选全部保留，未选不等于实验否定。

| 实际选项 | 本次优先级判断与理由 |
|---|---|
| ACM固定CMC梯度方向 | 首选设计：复用原生全部表示与对比集合，四臂能直接挑战“关闭CMC已足够”及“反向也一样”；零新增部署参数 |
| raw-CE引导的CMC梯度投影 | 备选：与PCGrad/辅助梯度文献接近，且VOGR尚未支持冲突；额外backward与Adam解释成本更高，不先做 |
| 来源均衡batch | 备选：同时改BN估计、负集合及曝光；要拆清需匹配曝光并交叉BN分组/对比集合，控制规模更大 |
| CMC低维共享子空间 | 保留：与已关闭SCVA选择子空间假设近，新增容量和rank选择自由度；先核清实质新问题 |
| 原Transformer投影器线性化 | 简化/效率备选：需区分容量与非线性，且非劣效及成本判定不是普通优效筛查器；不与ACM合并 |
| 补VCN完整方法强对手、范围与成文证据 | 始终有效的投入选项；若ACM只有一般训练技巧价值，这项优先于继续堆辅助模块 |

原池中普通同类mask/多正对、CR约束消融及XSN保留原处置；本轮不复制当前负项选择研究，也不为旧碰撞/MLEV更名救援。

## 8. 最小充分规模、成本与隔离

若以后明确选择实施：建议四臂×2枚新配对seed×9个developmental检查角色，**72次完整fit+72次check**，100epoch原生配方；七源训练/源内验证、另一个来源check、outer封存；先全局选择封存后check。两枚具体seed、分割资产和阈值尚未分配。独立训练S0，不复用当前XSN的S0成绩或checkpoint构成配对；新seed也不能把已多轮暴露的被试恢复成独立确认集。

仅规划复用 `xsn-r1/review_20261029/CONTROL_COST_ANALYSIS.md` 所引用的已完成VBR收据：72fit+72check约 **46.942 process-h**。这是不同算子的历史数量级，含进程内CPU/I/O，既不是独占GPU-h，也不是ACM实测报价。

假设有效并发4、速度损失25–50%、另留6h准备/工程/验收分析，规划约 **20.7–23.6h**；有效并发2则约 **35.3–41.2h**。并发、尾部、checkpoint I/O与准备成本不确定，以上不是上界。不能因为detach减少一侧梯度就承诺同比例提速：CE/CVC仍需两条分支反传。真实资格与成本门尚未执行。

ACM预算 **UNASSIGNED**。本设计不把XSN36小时或VBR275小时余额解释为ACM授权；现有计时继续按各自合同计算，原起点与累计历史保留。未来若运行，仍遵守服务器完整核验/保存、本地只回拉分析核心的规则。当前没有新合同、Bundle、Run、GPU操作或训练。

## 9. 接下来什么才有价值

针对“训练期异构辅助分支的固定单向CMC”的限定查新与独立反驳现已按用户后续请求完成。Grok-4.6-build建议HOLD训练，主Agent采纳：下一步只值得先解决零训练图资格；普通detach或角色不对称不能作为新算子。四臂足以回答窄问题，第五固定对称衰减臂不自动加入。若决定实施，最小资格核对需验证实际生产objective的梯度表、P0/N0首步主干等价、S0原生parity、非原位detach、四视图BN/RNG顺序、实际分析分支及完整进程成本。这里没有执行这些检查，不把静态推导记成PASS。

XSN结束后先按其原合同intake与判读。若XSN建立清楚支持，优先比较补其关键证据与启动ACM的价值；若XSN负/不明确，它也不会自动授权ACM或证明ACM正确。两者组合是另一个未设计的问题，不自动叠加。

初稿的构思trace、候选池及保护对象清单保留；本次新增限定查新、跨模型独立反驳与对照/成本报告。Grok的查新谨慎继续不等于科学有效性或执行验收通过。没有新建评估计划、Claim Ledger、Foundry、执行状态机或Run，没有运行forward/backward或训练；XSN保持原链。


## 10. 零训练审计后的设计取舍（2026-10-29）

本轮实际执行合成前向/反向和10个临时副本各一次Adam更新，0真实数据拟合。完整原生模块大小保留，native batch仅2/4；B64只在D8表示层检查，不是生产容量验收。四臂定义、关键比较、阈值草案及历史关闭边界不变；没有增加D0、调权或选择新的科研seed。审计证明P0/N0不是在所测实例中始终等价，不证明P0有益，更不证明优于R0。

主审计主体41.058秒、重放18.743秒，不含完整进程启动和准备分析；峰值CUDA allocated约1.158GiB、reserved1.750GiB不可外推到batch64。四臂历史成本仍46.942process-h，有效并发4的条件规划20.67–23.60h、并发2为35.34–41.21h。详见[成本更新](gradient_audit_20261029/COST_UPDATE.json)。

下一设计判断：已消除“首步相同所以后续必退化”的技术阻碍，有条件值得一次性、无调参救援的四臂探索；仍先比较XSN收尾后的证据价值，后续生产接线、判读冻结、Lab10资格/成本和独立预算各自需要完成。本轮不启动这些后继工作。前述第9节“尚未检查”仅描述外审前时点，现以本审计限定覆盖为准；原文快照已保存。


## 11. XSN负结果后的重评（2026-11-04）

MVCNET-ACM-R1-POST-XSN-REASSESSMENT-20261104。

XSN固定跨来源负项保护P0−S0 −0.192901pp、两seed负、5/9正向，已关闭；C0/N0未过门、R0单项通过不能替代效用。ACM不叠加/救援XSN，四臂保留原CE/CVC/CMC正负集合，只改CMC接收端。53项合成审计和CVC间接非退化证据使该窄问题可比较，不构成分类效用或新颖性。

判断为有条件值得一次无调参救援的四臂开发探索；在整篇论文投入排序上，补VCN主方法强近邻/归因/适用范围仍优先。三项固定比较P0−S0→P0−N0→P0−R0均需支持；四臂不证明优于一般CMC减权或VCN特有协同，不增加D0/B0，不选择控制臂替换候选。门仍为设计草案，预算和科学seed未分配。

成本更新采用刚完成XSN的原生S0 18fit+18check共9.378048process-h，×4为37.512192process-h。有效并发4、倍率1.25–1.5及6h统筹余量为17.72–20.07h；全XSN敏感性为20.15–22.99h，旧VBR情景20.67–23.60h。并发2对应29.45–41.21h的情景范围，均非ACM实测/上界。36h仅是将来投入商议目标，不是预算授权；真实成本不支持时先HOLD、不自动扩额。

ACM预算UNASSIGNED，暂不生产接线或训练。XSN480h和VBR275h及原起点/关闭/历史保持；不转用余额。详细选择证据、对照、限制、后续资格和完整成本表见[本轮报告](reassessment_20261104/REPORT.md)、[可复算成本](reassessment_20261104/COST_ESTIMATE.json)。前次查新/外审复用其覆盖，不声称本轮重新查新或外审改判。
