# V2 MTP greedy verify 实验

设置 `VLLM_ASCEND_EXPERIMENT_GREEDY_VERIFY=1` 后，仅 Model Runner V2 的 `method=mtp`、整批 temperature=0 且未启用水印、synthetic、block、FLy 或 entropy verify，并且 draft 词表不小于 target 词表时，使用实验 greedy verifier。其他情况回退到原实现；默认关闭。安装 patch 会包装 upstream `RejectionSampler.__init__`，从 `spec_config.method` 获取方法，未知方法安全回退。

测试：在运行服务的、包含 vLLM、torch_npu、triton-ascend 与 pytest 的环境中执行：

```bash
pytest -q tests/ut/worker/v2/test_greedy_verify_experiment.py
pytest -q tests/e2e/nightly/single_node/ops/singlecard_ops/triton/test_greedy_verify_experiment.py
```

第二项直接对照 NPU 原 rejection_sample 的有效 token 和数量，覆盖不同请求长度、placeholder、拒绝、bonus、跨 block 平局、NaN/Inf、非整除词表。NaN/Inf 或 bonus 分块差异可能暴露不一致；必须实际通过后再判断精度。不要只以 Python mock 测试代替 NPU 对照。

当前变更引入 MTP 和 entropy verify 守卫后，之前的性能、正确性和 trace 数据未绑定此提交，需要重新进行相同模型、负载与代码版本下的验证。V2 greedy verify 优化的是 verify 后采样，不等同于已缩短 MTP 到下一轮 verify 的设备空泡。本地仅完成静态检查，未完成 NPU 测试。
