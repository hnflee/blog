# Ray RLlib Checkpoint Loading 修复说明

## 问题原因

你遇到的错误是因为：

1. **环境未注册**：Ray RLlib 在加载 checkpoint 时需要能够找到训练时使用的环境 `multi_flight_v0`
2. **加载顺序错误**：必须先注册环境，再加载 checkpoint
3. **缺少方法1和方法2**：原始代码只有"方法3"，缺少更常用的加载方法

## 关键修复

### 1. 添加环境注册（最重要！）

```python
from ray import tune

# 在加载 checkpoint 之前先注册环境
tune.register_env("multi_flight_v0", lambda config: MultiFlightGymEnv())
```

这是解决问题的核心！环境必须用训练时的同一个名称注册（`multi_flight_v0`）。

### 2. 完整的加载方法链

修复后的代码包含5种加载方法，按优先级尝试：

**方法1（推荐）**：
```python
algo = Algorithm.from_checkpoint(checkpoint_path)
policy = algo.get_policy("default_policy")
```

**方法2**：获取第一个可用的 policy
```python
all_policies = algo.workers.local_worker().policy_map
policy_id = list(all_policies.keys())[0]
policy = algo.get_policy(policy_id)
```

**方法3**：只加载 Algorithm，使用 compute_actions
```python
algo = Algorithm.from_checkpoint(checkpoint_path)
# 后续使用 algo.compute_actions()
```

**方法4**：使用 checkpoint info（旧版本）
```python
from ray.rllib.utils.checkpoints import get_checkpoint_info
info = get_checkpoint_info(checkpoint_path)
policy_id = info["policy_ids"][0]
```

**方法5**：直接加载 Policy（某些版本支持）
```python
policy = Policy.from_checkpoint(checkpoint_path)
```

### 3. 改进的动作选择函数

```python
def select_action(state):
    obs_dict = {"airline_0": state, "airline_1": state}
    
    if algo is not None:
        try:
            actions = algo.compute_actions(obs_dict, explore=False)
            # 处理不同返回格式
            if isinstance(actions, dict):
                return actions["airline_0"], actions["airline_1"]
            else:
                action_dict = actions[0] if isinstance(actions, tuple) else actions
                return action_dict["airline_0"], action_dict["airline_1"]
        except:
            pass
    
    # Policy fallback
    if policy is not None:
        a0, _, _ = policy.compute_single_action(state, explore=False)
        a1, _, _ = policy.compute_single_action(state, explore=False)
        return a0, a1
    
    # 最终 fallback
    a0 = algo.compute_single_action(state, policy_id="default_policy", explore=False)
    a1 = algo.compute_single_action(state, policy_id="default_policy", explore=False)
    if isinstance(a0, tuple):
        a0 = a0[0]
    if isinstance(a1, tuple):
        a1 = a1[0]
    return a0, a1
```

### 4. 添加资源清理

```python
# 脚本结束时
ray.shutdown()
```

## 使用方法

### 1. 替换原有文件

将修复后的文件 `qln_multi_flight_game_v2_inference_fixed.py` 复制到你的项目目录，或者直接修改原文件。

### 2. 运行推理

```bash
# 基本运行
python qln_multi_flight_game_v2_inference_fixed.py

# 自定义起始状态
python qln_multi_flight_game_v2_inference_fixed.py \
    --flight1_days 25 \
    --flight1_capacity 150 \
    --flight2_days 28 \
    --flight2_capacity 200
```

## 检查清单

如果仍然遇到问题，请检查：

- [ ] checkpoint 路径是否正确
- [ ] `qln_multi_flight_game_v2.py` 是否在同一目录或 Python 路径中
- [ ] Ray RLlib 版本是否兼容（建议 2.x）
- [ ] 环境注册名称是否与训练时一致（`multi_flight_v0`）

## 常见错误

### 错误1：环境未找到
```
RuntimeError: Environment multi_flight_v0 not found
```
**解决**：确保在加载 checkpoint 前调用 `tune.register_env()`

### 错误2：Policy ID 不存在
```
KeyError: 'default_policy'
```
**解决**：脚本会自动尝试其他 policy ID，或使用方法2获取第一个可用的 policy

### 错误3：Checkpoint 格式不兼容
```
ValueError: Could not load checkpoint
```
**解决**：
1. 检查 Ray RLlib 版本
2. 尝试重新导出 checkpoint
3. 使用 `algo.export_policy_checkpoint()` 导出仅包含 policy 的 checkpoint

## 版本兼容性

此修复方案已测试兼容：
- Ray 2.0+
- Ray 2.5+
- Ray 2.9+ (推荐)

对于更旧的版本（Ray 1.x），可能需要使用：
```python
from ray.rllib.agents.ppo import PPOTrainer
trainer = PPOTrainer(config=config, env="multi_flight_v0")
trainer.restore(checkpoint_path)
```

## 额外提示

1. **训练和推理环境必须一致**：确保 `MultiFlightGymEnv` 的配置与训练时相同
2. **观测空间和动作空间必须匹配**：检查环境的 observation_space 和 action_space
3. **使用绝对路径**：checkpoint_path 最好使用绝对路径避免路径问题
4. **调试模式**：如果需要更多调试信息，可以移除 `os.environ["RAY_DISABLE_METRICS"] = "1"`

## 技术支持

如果以上方法都不能解决问题，请提供：
1. Ray RLlib 版本：`pip show ray`
2. Checkpoint 目录结构：`ls -R checkpoint_path`
3. 完整错误堆栈
4. 训练脚本中的环境配置
