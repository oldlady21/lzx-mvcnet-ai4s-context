# XSN-R1 累计480小时预算续接与完整训练启动

截至2026-10-31T04:56:32.651330+08:00，新Run `2026-10-31_mvcnet_xsn_r1_lab10_r1_budget480-full-w1` 已真实启动。source-import PASS，10项历史资格复用节点PASS，冻结成本门PASS，完整训练diagnostic节点RUNNING。科学拟合：完成0、运行0、待运行90、失败0；check90项均未开始，须全90fit封存后才能执行。当前训练开始标记0项；**进入完整训练队列不等于已经发生首个GPU拟合**，状态以本快照和精确Run看板为准。

## 授权、预算与历史

用户先要求“增加预算，启动实验”，随后明确选择“累计480小时，增加工程余量”。这是原36h总上限增加444h，不是另开480h余额。新Transaction432h、收尾预留6h；2026-10-29 13:20UTC（21:20北京时间）原累计起点不重置。State已登记授权及扩额历史，max_rounds1和round_index0保留，同一科学阶段。

原Run导入失败和repair1成本停止均已保留为非科学INVALID记录；repair1的10项资格全部成功，科学0，成本STOP419.2589h超过原24h Transaction余额，未把其写成科学负结果。停机结算保留原9.536963h，新增已核验成本停止至首次后续状态查询的2.602477h，共12.139440h；两Run累计纯资源等待16.505126h另计，区间/源ledger哈希验证且不与执行重叠。截止点后的状态核对、准备、修复、验收均计入。

本启动快照累计计费2.992h、剩余477.008h；执行和排队持续变动，权威值以同Run最新probe为准。没有转用旧VBR275h或抹掉原36h成本。

## 十项资格复用和真实成本门

完整资格工件留原服务器。服务器逐项重算10份selected.pt SHA、task_result/task/capacity摘要和旧103文件源码哈希；仅回拉核心技术证明，没有下载checkpoint或读取科学效果。`QUALIFICATION_REUSE_SOURCE_VERIFIED.json`保留原远端验收原文，最终清单再显式绑定每个task的source/result/remote checkpoint三方摘要相等。新科学拟合全部从原协议初始化，**不warm-start资格checkpoint**。

新qualification scope为cpu_analysis：读取十项已验证收据，写明historical source/reused provenance、规范技术判定字段，运行原成本函数并由实际Transaction emit闭环；0次新资格训练、无资格GPU调度。服务器真实成本预测419.258885h、可用科学执行窗口431.994343h，故PASS。最慢epoch、K=1.279301、1.25安全倍率、100epoch缩放和90fit+90check原公式保持，没有用平均epoch或假定K=10救门。预算总额480h没有改变科学成功门。

## 冻结与工程验收

五臂S0/P0/C0/N0/R0、seed71/72、九outer、18划分、loss/mask/增强/模型/训练长度和全部科学门与repair1逐字段/源码哈希保持；仅预算时限、资格证据复用、时间结算验证及启动接口发生工程变化。source digest及完整比较见`CONTRACT_BINDING_PACKFIX1.json`。

最终包 `2026-10-31_mvcnet_xsn_r1_lab10_bundle_r1_budget480_packfix1`；未派发prelaunch0合同/包及原始外审均保留。Grok-4.6-build限定复核原文REVISE提出两项证据闭合要求，均在派发前实施：两份来源Run都核验qualification终止、无scientific节点、墙钟终点闭合；10份checkpoint SHA必须在source/result/remote映射逐项相等。新增负例与真实保存收据CPU消费者闭环通过，不将外审改写为READY/PASS。

最终限定回归与62项共享接口通过，实际完整node成本门/emit、资格逻辑10和science argv90/90、实际worker导入和111文件上传manifest均通过。共享controller仅增加精确budget480的can_schedule分支，原36h历史路径保留；同campaign原起点扩额，不调用start_campaign重置计时。Lab10通过既有等价跳板访问相同账号/数据/项目。

## 执行边界与后续

后台链source-import→资格证据复用/成本门→90fit→全局seal→90check。资源满足后同Run自动继续；432h执行上限和480h累计上限仍是硬界，纯排队排除。失败时保留日志、阻断依赖，不自动调参、加seed或另建Run。完整payload留服务器，完成后只回analysis_v1核心并验收分析。

此快照尚无科学结果或Claim。只读观察命令保存于`MONITOR.ps1`：

```powershell
& '[REDACTED_MACHINE]' './scripts\ai4s\monitor.py' '--host' '[REDACTED_MACHINE]' '--remote-python' '[REDACTED_MACHINE]' '--output' '[REDACTED_MACHINE]' '--run-id' '2026-10-31_mvcnet_xsn_r1_lab10_r1_budget480-full-w1'
```

合同canonical SHA：`1c199413e43a6c42faa6badcee0db2ef1f056838527818063b47a6948618f3a8`；source_manifest SHA：`df03f73977bc5363ee5955b48b8a2cbf2dc98762f5ad3fa1153438ae1fe7cfac`。
