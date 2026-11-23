# 自动化可靠性实证记录 R4-A

2026-11-23 UTC。本轮只检查通用队列、宿主回执和归档路径。

|能力|实际结果|
|---|---|
|唯一小时脚本|本机492c2e4d-66be-43c3-ad61-5d1a44d10fe0仍active，0 * * * * UTC；其他电脑离线，不能断言全账号唯一|
|最新小时自然触发|run 760c4756-b040-4556-92c9-4b136d8e365a，failed，script exited with code 1: RuntimeError: report divergence; retained local files|
|R3一次性代理自然触发|run fd8a16ac-ad1d-4aff-9e79-fae35a347380，failed；queued heartbeat Codex thread provider identity does not match the current session route|
|代理执行完整单元|该次唤醒未通过，不能宣称执行成功|
|CWC自动交接|未验收；既有人工DONE不能替代|
|本轮隔离测试|python -m unittest discover -s tests -p 'test*queue*.py'：14 tests，OK，退出0；不是自然唤醒证据|

tests/test_research_queue.py的临时目录测试覆盖完成后选择B2、重复完成不改字节、坏哈希拒绝、锁冲突、失败保留、依赖与路径限制、只读选择。其他dummy测试覆盖异常/中断，不接触实验数据。

小时归档失败与Provider失败分属不同路径。hourly_supervisor.py的archive在报告不一致时主动抛错并保留文件，不能关闭这一保护来制造成功。本轮没有覆盖有分歧的报告，没有改Provider身份检查，没有创建新自动化。归档差异尚待逐文件对账；宿主身份故障仍BLOCKED_BY_RUNTIME。

保留原时钟；历史脚本成功不可覆盖本次失败。本轮测试仅证明被测软件分支，不证明无人值守端到端链路。修复成功必须另有实际自然运行回执。

