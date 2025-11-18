#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
轻量级推理版本 - 专门优化加载速度
关键改进：
1. 使用最小配置加载（num_env_runners=0）
2. 禁用所有不必要的功能
3. 添加详细的进度提示
"""

import os
import warnings
os.environ["RAY_DISABLE_METRICS"] = "1"
os.environ["RAY_DEDUP_LOGS"] = "0"
os.environ["PYTHONWARNINGS"] = "ignore"
warnings.filterwarnings("ignore")

import ray
import numpy as np
import argparse
from pathlib import Path

print("="*80)
print("          多航班动态定价推理 - 轻量级版本")
print("="*80)

# ====================== 配置 ======================
checkpoint_path = '/Users/lifeng/ray_results/FlightPPO_Final_2025/PPO_multi_flight_v0_f4f6f_00000_0_2025-11-18_16-31-19/checkpoint_000000'

# ====================== 1. 验证文件 ======================
print("\n[1/5] 验证文件...")
if not Path(checkpoint_path).exists():
    print(f"❌ Checkpoint 不存在: {checkpoint_path}")
    exit(1)
print("✓ Checkpoint 文件存在")

try:
    from qln_multi_flight_game_v2 import MultiFlightGymEnv
    print("✓ 环境文件导入成功")
except Exception as e:
    print(f"❌ 无法导入环境: {e}")
    exit(1)

# ====================== 2. 初始化 Ray（轻量级）======================
print("\n[2/5] 初始化 Ray（轻量级配置）...")
ray.init(
    ignore_reinit_error=True,
    log_to_driver=False,
    num_cpus=2,  # 最小配置
    num_gpus=0,
    _system_config={
        "automatic_object_spilling_enabled": False,
        "max_io_workers": 2,
    }
)
print(f"✓ Ray {ray.__version__} 初始化成功")

# ====================== 3. 注册环境 ======================
print("\n[3/5] 注册环境...")
from ray import tune
tune.register_env("multi_flight_v0", lambda config: MultiFlightGymEnv())
print("✓ 环境 'multi_flight_v0' 注册成功")

# ====================== 4. 加载模型（关键步骤）======================
print("\n[4/5] 加载模型...")
print("提示: 这一步可能需要30-90秒，请耐心等待...")
print("如果超过2分钟还没响应，请按 Ctrl+C 中断，然后检查:")
print("  1. Checkpoint 文件是否完整")
print("  2. Ray 版本是否兼容")
print("  3. 系统内存是否充足\n")

algo = None
policy = None

# 方法1：使用最小配置加载（推荐，速度最快）
try:
    print("[进行中] 正在加载... (方法1: 轻量级配置)")
    
    from ray.rllib.algorithms.ppo import PPOConfig
    
    # 创建推理专用配置
    config = (
        PPOConfig()
        .environment("multi_flight_v0")
        .env_runners(
            num_env_runners=0,  # 推理时不需要额外的 workers
            num_envs_per_env_runner=1,
        )
        .resources(num_gpus=0)
        .framework("torch")  # 或 "tf2"，取决于你训练时用的
        .debugging(log_level="ERROR")
    )
    
    print("  → 创建 Algorithm 对象...")
    algo = config.build()
    
    print(f"  → 从 checkpoint 恢复权重...")
    algo.restore(checkpoint_path)
    
    print("  → 获取 policy...")
    policy = algo.get_policy("default_policy")
    
    print("✓ 加载成功！(方法1)")
    
except Exception as e:
    print(f"✗ 方法1失败: {e}")
    print("\n尝试方法2...")
    
    # 方法2：标准加载
    try:
        from ray.rllib.algorithms.algorithm import Algorithm
        
        print("[进行中] 正在加载... (方法2: 标准加载)")
        algo = Algorithm.from_checkpoint(checkpoint_path)
        
        try:
            policy = algo.get_policy("default_policy")
        except:
            all_policies = algo.workers.local_worker().policy_map
            policy_id = list(all_policies.keys())[0]
            policy = algo.get_policy(policy_id)
        
        print("✓ 加载成功！(方法2)")
        
    except Exception as e2:
        print(f"✗ 方法2也失败: {e2}")
        print("\n❌ 所有加载方法都失败了")
        print("请检查:")
        print("  1. Ray 版本: pip show ray")
        print("  2. Checkpoint 是否完整")
        print("  3. 训练时使用的框架 (torch/tf)")
        ray.shutdown()
        exit(1)

if not algo and not policy:
    print("❌ 无法加载模型")
    ray.shutdown()
    exit(1)

# ====================== 5. 推理函数 ======================
print("\n[5/5] 设置推理...")

def select_action(state):
    """选择动作"""
    obs_dict = {"airline_0": state, "airline_1": state}
    
    if algo is not None:
        try:
            actions = algo.compute_actions(obs_dict, explore=False)
            if isinstance(actions, dict):
                return actions["airline_0"], actions["airline_1"]
            else:
                action_dict = actions[0] if isinstance(actions, tuple) else actions
                return action_dict["airline_0"], action_dict["airline_1"]
        except:
            pass
    
    if policy is not None:
        a0, _, _ = policy.compute_single_action(state, explore=False)
        a1, _, _ = policy.compute_single_action(state, explore=False)
        return a0, a1

# 测试推理
env = MultiFlightGymEnv()
obs, _ = env.reset()
a0, a1 = select_action(obs["airline_0"])
print(f"✓ 推理测试成功！示例动作: {a0}, {a1}")

# ====================== 开始推理 ======================
print("\n" + "="*80)
print("开始推理测试")
print("="*80)

parser = argparse.ArgumentParser()
parser.add_argument('--episodes', type=int, default=10, help='测试回合数')
args = parser.parse_args()

test_episodes = args.episodes
total_rewards_list = []
f1_revenues, f2_revenues = [], []
f1_lfs, f2_lfs = [], []

print(f"\n运行 {test_episodes} 个回合...")
for ep in range(test_episodes):
    obs, _ = env.reset()
    state = obs["airline_0"]
    done = False
    
    while not done:
        a0, a1 = select_action(state)
        action_dict = {"airline_0": a0, "airline_1": a1}
        obs, rewards, dones, _, infos = env.step(action_dict)
        state = obs["airline_0"]
        done = dones["__all__"]

    total_r = env.core.total_revenue
    total_rewards_list.append(total_r)
    f1_revenues.append(env.core.revenue1)
    f2_revenues.append(env.core.revenue2)
    f1_lfs.append(env.core.sold1 / env.core.init_cap1)
    f2_lfs.append(env.core.sold2 / env.core.init_cap2)

    print(f"  第 {ep+1:2d} 回合 → 总收益 {total_r:,.0f} | "
          f"航班1 {env.core.revenue1:,.0f} ({env.core.sold1/env.core.init_cap1:.1%}) | "
          f"航班2 {env.core.revenue2:,.0f} ({env.core.sold2/env.core.init_cap2:.1%})")

print("\n" + "="*80)
print("统计总结")
print("="*80)
print(f"平均总收益: {np.mean(total_rewards_list):,.0f} ± {np.std(total_rewards_list):,.0f}")
print(f"最高总收益: {max(total_rewards_list):,.0f}")
print(f"最低总收益: {min(total_rewards_list):,.0f}")
print(f"航班1 - 平均收益: {np.mean(f1_revenues):,.0f}  平均负载: {np.mean(f1_lfs):.1%}")
print(f"航班2 - 平均收益: {np.mean(f2_revenues):,.0f}  平均负载: {np.mean(f2_lfs):.1%}")

print("\n" + "="*80)
print("✓ 推理完成！")
print("="*80)

ray.shutdown()
