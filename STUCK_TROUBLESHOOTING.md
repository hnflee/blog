# 程序卡住问题排查指南

## 🔴 你遇到的问题

程序在显示以下信息后**卡住不动**：

```
2025-11-18 17:20:33,667 INFO tensorboardx.py:45 -- pip install "ray[tune]" to see TensorBoard files.
2025-11-18 17:20:33,667 WARNING unified.py:56 -- Could not instantiate TBXLogger: No module named 'tensorboardX'.
[2025-11-18 17:20:33,728 E 75330 2394350] core_worker.cc:2200: Actor with class name: 'MultiAgentEnvRunner'...
[2025-11-18 17:21:03,807 E 75330 2394847] core_worker_process.cc:825: Failed to establish connection...
```

然后就没有任何输出了。

---

## 🎯 问题原因

程序卡在了 **`Algorithm.from_checkpoint()`** 这一步，可能的原因：

### 1. Ray 尝试创建过多的 Workers（最常见）
- `Algorithm.from_checkpoint()` 默认会创建多个 EnvRunner workers
- 这些 workers 需要时间初始化
- 如果配置不当，可能会卡住

### 2. Checkpoint 文件问题
- Checkpoint 文件损坏
- Checkpoint 路径不正确
- Checkpoint 与当前环境不匹配

### 3. 资源不足
- 内存不够
- CPU 资源被占用
- Ray 无法分配足够的资源给 workers

### 4. 环境定义问题
- 环境注册失败
- 环境与训练时不一致
- 环境初始化太慢

---

## ✅ 解决方案

### 方案1：使用轻量级版本（推荐）⭐

```bash
python qln_multi_flight_game_v2_inference_lightweight.py
```

**优势**：
- ✅ 使用 `num_env_runners=0` 配置（不创建额外的 workers）
- ✅ 加载速度快（30-60秒）
- ✅ 资源占用少
- ✅ 专门为推理优化

**工作原理**：
```python
config = (
    PPOConfig()
    .environment("multi_flight_v0")
    .env_runners(
        num_env_runners=0,  # ← 关键：推理时不需要额外的 workers
        num_envs_per_env_runner=1,
    )
    .resources(num_gpus=0)
)

algo = config.build()
algo.restore(checkpoint_path)  # ← 只恢复权重，不创建 workers
```

---

### 方案2：使用调试版本（排查问题）

```bash
python qln_multi_flight_game_v2_inference_debug.py
```

**功能**：
- ✅ 分步显示加载进度
- ✅ 详细的错误信息
- ✅ 120秒超时保护
- ✅ 多种加载方法自动尝试

**适合**：
- 第一次运行，想知道卡在哪里
- 排查具体的错误原因
- 验证 checkpoint 是否有效

---

### 方案3：手动修复原代码（最小改动）

在你的原始代码中，**关键是添加 `num_env_runners=0`**：

```python
# 在加载 checkpoint 之前添加
from ray.rllib.algorithms.ppo import PPOConfig

# 创建推理配置
config = (
    PPOConfig()
    .environment("multi_flight_v0")
    .env_runners(num_env_runners=0)  # ← 这是关键！
    .resources(num_gpus=0)
)

# 使用配置加载
algo = config.build()
algo.restore(checkpoint_path)
policy = algo.get_policy("default_policy")
```

---

## 🔍 诊断步骤

### 第1步：确认 Checkpoint 是否存在

```bash
ls -lh '/Users/lifeng/ray_results/FlightPPO_Final_2025/PPO_multi_flight_v0_f4f6f_00000_0_2025-11-18_16-31-19/checkpoint_000000'
```

应该看到类似：
```
algorithm_state.pkl
policies/
rllib_checkpoint.json
```

### 第2步：检查 Ray 版本

```bash
pip show ray
```

推荐版本：Ray 2.5+ 或 2.9+

### 第3步：检查系统资源

```bash
# 内存使用
free -h

# CPU 使用
top

# Ray 进程
ps aux | grep ray
```

### 第4步：清理旧的 Ray 进程

```bash
# 停止所有 Ray 进程
ray stop

# 清理临时文件
rm -rf /tmp/ray

# 重新运行
python qln_multi_flight_game_v2_inference_lightweight.py
```

---

## 📊 三个版本对比

| 版本 | 加载速度 | 资源占用 | 适用场景 | 推荐度 |
|-----|---------|---------|---------|--------|
| `lightweight.py` | ⚡ 快（30-60秒） | 💚 低 | 日常推理 | ⭐⭐⭐⭐⭐ |
| `debug.py` | 🐢 中（60-120秒） | 💛 中 | 问题排查 | ⭐⭐⭐⭐ |
| `fixed.py` | 🐌 慢（可能卡住） | 💔 高 | 不推荐 | ⭐⭐ |

---

## 🚨 常见错误及解决

### 错误1：程序卡在加载（超过2分钟）

**症状**：
```
正在加载训练好的 PPO 模型 checkpoint...
[然后就没有输出了]
```

**解决**：
1. 按 `Ctrl+C` 中断
2. 运行 `ray stop` 清理
3. 使用 `lightweight.py` 版本

---

### 错误2：OOM (Out of Memory)

**症状**：
```
ray::RolloutWorker.init() killed
```

**解决**：
```python
ray.init(
    num_cpus=2,  # 减少 CPU
    object_store_memory=1000000000,  # 限制内存为 1GB
)
```

---

### 错误3：Workers 无法启动

**症状**：
```
The actor died unexpectedly before finishing this task
```

**解决**：
使用 `num_env_runners=0` 禁用额外的 workers

---

### 错误4：Checkpoint 版本不兼容

**症状**：
```
ValueError: Could not deserialize checkpoint
```

**解决**：
1. 检查 Ray 版本是否与训练时一致
2. 重新导出 checkpoint：
```python
algo.save_to_path("/path/to/new/checkpoint")
```

---

## 💡 性能优化建议

### 1. 训练时就优化 Checkpoint

训练时添加：
```python
config = (
    PPOConfig()
    .checkpointing(
        export_native_model_files=True,  # 导出原生模型
    )
)
```

### 2. 导出仅包含 Policy 的 Checkpoint

```python
# 训练后
algo.export_policy_checkpoint("/path/to/policy/only")

# 推理时
policy = Policy.from_checkpoint("/path/to/policy/only")
```

### 3. 使用 ONNX 格式（最快）

```python
# 导出为 ONNX
algo.export_policy_model("/path/to/model.onnx", onnx=True)

# 推理时使用 ONNX Runtime（速度提升 3-10x）
```

---

## 📝 总结

### 立即解决方案

```bash
# 1. 清理环境
ray stop

# 2. 运行轻量级版本
python qln_multi_flight_game_v2_inference_lightweight.py

# 3. 如果还是卡住，运行调试版本看详细信息
python qln_multi_flight_game_v2_inference_debug.py
```

### 关键要点

1. **问题根源**：`Algorithm.from_checkpoint()` 默认创建多个 workers，导致卡住
2. **解决方案**：使用 `num_env_runners=0` 配置
3. **最佳实践**：推理时不需要额外的 workers
4. **加载方式**：先创建配置，再 restore 权重

### 预期效果

使用轻量级版本后，你应该看到：

```
[1/5] 验证文件...
✓ Checkpoint 文件存在
✓ 环境文件导入成功

[2/5] 初始化 Ray（轻量级配置）...
✓ Ray 2.9.0 初始化成功

[3/5] 注册环境...
✓ 环境 'multi_flight_v0' 注册成功

[4/5] 加载模型...
提示: 这一步可能需要30-90秒，请耐心等待...
[进行中] 正在加载... (方法1: 轻量级配置)
  → 创建 Algorithm 对象...
  → 从 checkpoint 恢复权重...
  → 获取 policy...
✓ 加载成功！(方法1)

[5/5] 设置推理...
✓ 推理测试成功！

开始推理测试
运行 10 个回合...
  第  1 回合 → 总收益 XXX,XXX | ...
  第  2 回合 → 总收益 XXX,XXX | ...
  ...
```

**整个过程应该在 1-2 分钟内完成！**

如果还有问题，请提供：
1. 运行 `debug.py` 的完整输出
2. Ray 版本
3. 系统内存信息
4. Checkpoint 目录结构（`ls -R checkpoint_path`）
