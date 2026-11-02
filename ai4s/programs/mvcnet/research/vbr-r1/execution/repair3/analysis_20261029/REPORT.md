# VBR-R1 repair3-packfix2 最终验收与结论

2026-10-29，Program=mvcnet。精确证据 `AI4S-MVCNET-mvcnet-vbr-r1-screen-lab10-r1-repair3`。

本轮完整训练与验收完成，冻结判定 **NO_SIGNAL**，已通过原控制器登记。Program 为 `COMPLETE_MAX_ROUNDS`、暂停，无活动 Run/operation；固定干预 `vcn-view-batch-renorm-v01` 已登记 `FROZEN_EVIDENCE_ONLY`。没有新增训练、参数救援或论文 Claim。

## 结果与科学门

72/72 fresh 科学拟合、72/72 check 完成，running/pending/failed 均0；8项资格按冻结来源复用。9个 outer × seeds69/70 × B0/S0/N0/P0 的原矩阵不变。先在被试内平均两个 seed，再汇总9名被试，不把18行当作18名独立被试。

| P0−S0 冻结门 | 实际值 | 判定 |
|---|---:|---|
| 平均增益 ≥ +0.5pp | +0.019290pp | 未通过 |
| 两个 seed 均值均 > 0 | seed69 −0.250772；seed70 +0.289352pp | 未通过 |
| 正向被试 ≥6/9 | 4/9 | 未通过 |
| 平均 CE gain ≥0 | +0.012652578 | 通过 |
| 被试平均差 ≤−5pp 的人数 ≤1 | 0 | 通过 |

四臂平均 ACC：B0 56.278935%、S0 58.275463%、N0 56.066744%、P0 58.294753%。P0−B0 +2.015818pp，N0−B0 −0.212191pp，交互差 +0.231481pp，均仅作预登记描述，不能替换 P0−S0 主门。P0−B0 的差异主要没有转化成超越 S0 支架的增量；CE改善也不能补偿三个失败门。

该结果关闭的是本次固定 BRN 日程/限幅/配方的七源 developmental 实例，不证明所有归一化校正无效，不构成独立最终 LOSO 确认或创新性证明。后续处置见 [PRECISE_CLOSURE.json](PRECISE_CLOSURE.json)：不追加 seed、调日程/阈值或更名救援；把该结果作为 VCN 主方法证据与写作的设计约束，不自动创建下一实验。

## 完整远端验证与核心 intake

用户本轮指令：“按新规则，完整结果留在服务器，分析后只拉回核心产物”。旧冻结合同没有精简回传字段，因此使用明确、单 Run、哈希固定的 [ADDENDUM.json](intake_recovery_v1/ADDENDUM.json)，没有修改原合同、Bundle或原 artifact manifest，也没有把 engineering snapshot 改称正式结果。

- Run：`2026-10-28_mvcnet_vbr_r1_lab10_r1_repair3_packfix2-full-w1`。
- 原完整结果保留于 Lab10 `[REDACTED_MACHINE]<Run>`；完整229文件 Bundle 仍保留于对应 `ai4s-runs/2026-10-28_mvcnet_vbr_r1_lab10_bundle_r1_repair3_packfix2`。
- 远端原清单1,242文件、46,825,495,710 bytes（约46.83GB），含72个 checkpoint，真实逐文件哈希核验通过。
- 在独立 `ai4s-analysis/<Run>` 目录，先完整验证，再运行包内冻结分析，另用独立 math/logsumexp 实现重算41,472条预测，并对72个 check 的正确数/分母/CE/预测文件 SHA逐项核对。未训练或执行模型前向，原 Run 字节不变。
- 核心包959个清单文件；加 core manifest/receipt 共961文件、27,222,739 bytes（约27.22MB）。保留154任务收据、3节点结果/排队时钟、全局 student seal、资格来源、全部 raw check logits、训练日志/元数据、两份分析与完整验证证明、实际执行的独立复算源码。没有拉回 checkpoint 或数据数组。
- 统一 `advance fetch → advance intake` 已真实通过，标准核心结果路径为 `results/logs/remote/<Run>`，正式 marker 为 `tmp/remote_download/<Run>/verification.json`。scope=`analysis_v1_recovery`，`complete_remote=true`，**`full_payload_local=false`**。包内运输收据不能替代标准 marker。
- 本地从正式核心包 logits 再计算72 check/41,472预测，correct/count/预测 SHA精确一致，CE差≤1e−10，见 [LOCAL_CORE_REPLAY.json](LOCAL_CORE_REPLAY.json)。

先前全量 SCP 已中止；唯一临时隔离目录的802文件/172,041,878 bytes在远端完整证据及核心 intake 成功后删除，处置记录 [ABORTED_FETCH_DISPOSITION.json](ABORTED_FETCH_DISPOSITION.json)。原失败Run、OOM、字段修复、未派发候选及全部历史日志均保留。

## 成本与预算

Lab10 资格成本门 PASS 的保守预测为236.987060h，冻结 TX上限237.25h、包外验收预留6h。最终该 Run 实际排队剔除后的 Transaction 墙钟执行6.432206h，资源排队252.244秒；72训练进程时间累计46.424703h、72 check累计0.517232h。并发和资格阶段资源竞争不同，所以进程时间之和、并发墙钟及保守预测不是同一成本单位，不能用其比值宣称方法训练加速。

截至本报告快照，原275h累计预算约已用39.425h、余235.575h，核实可剔除的资源排队约70.508h；没有 stopped-time 豁免。原计时起点 **2026-10-24T04:35:51+00:00**、全部修复成本均保留，没有重置。此次无需增加预算。控制器 `can_schedule=false` 包含暂停、轮次已满及无法再预留一整轮237.25h+6h的原因，**不表示本次已超支**，也不授予使用余额继续训练的权限。结构化快照见 [RESULT_SUMMARY.json](RESULT_SUMMARY.json)。

## 复核与可追溯性

Grok `grok-4.6-build` 对新增核心验收路径给出真实 `ACCEPT_WITH_LIMITS`，非正式科学 Claim审查；两次达到turn上限的 incomplete trace也原样保留。限制与处理见 [REVIEW_DISPOSITION.md](REVIEW_DISPOSITION.md)，实际最终原文见 [GROK_CORE_REVIEW_FINAL_REPORT.md](GROK_CORE_REVIEW_FINAL_REPORT.md)。实物集成6项、共享transport/controller12项、Transaction21项及helper4项通过；精确命令和真实输出见 [VALIDATION_REPORT.md](intake_recovery_v1/VALIDATION_REPORT.md)。

| 对象 | SHA-256 |
|---|---|
| 冻结合同 canonical | `57dfc4bd89fc84bad17a024df6ec1a313e7ac468e401443884209e56fb33ffdf` |
| 原 Bundle manifest | `8ed6eb37ce031dc04af05fd31695670e9acb082d8fe55f50415bdfbb4337d946` |
| 原完整 artifact manifest | `206ac78625979fe528c4061543e8158f3a41a5595451ff724a95bf543116812f` |
| 核心 manifest | `8f23d0cf6a3c85870189581f8bb578816bcc4cb10bb4cd2778aa461adba0a516` |
| 冻结分析结果 | `834903a7816809b4a585f8621d3b7ba7ec30dce5c264e78e2e6520cc0892c749` |
| 独立重算结果 | `f6b2c48f13e0c872c0596d1526070299cb9d1d868936177e1c9b1a9b2ab6a1d9` |

独立分析实际执行源码SHA为`652e5c7894d2640cae8a7d6babcc28e65a68877e071861f5f6032b9d12645d0b`，在核心包 `_analysis/INDEPENDENT_SOURCE.py`；本地后来增强的脚本未用于此次结果，见 [INDEPENDENT_SOURCE_DISPOSITION.json](INDEPENDENT_SOURCE_DISPOSITION.json)。原分析与其绑定不覆写，新增逐任务一致性仅形成派生绑定。
