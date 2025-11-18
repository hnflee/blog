#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import warnings
import time
from pathlib import Path

# 静默配置
os.environ["RAY_DISABLE_METRICS"] = "1"
os.environ["RAY_DEDUP_LOGS"] = "0"
os.environ["PYTHONWARNINGS"] = "ignore"
warnings.filterwarnings("ignore")

import ray
import numpy as np
import argparse
from ray import tune
from ray.rllib.policy.policy import Policy
from ray.rllib.algorithms.algorithm import Algorithm

# ====================== 第1步：检查文件是否存在 ======================
print("="*80)
print("第1步：验证环境和文件")
print("="*80)

checkpoint_path = '/Users/lifeng/ray_results/FlightPPO_Final_2025/PPO_multi_flight_v0_f4f6f_00000_0_2025-11-18_16-31-19/checkpoint_000000'

print(f"\n检查 checkpoint 路径...")
print(f"路径: {checkpoint_path}")

checkpoint_dir = Path(checkpoint_path)
if not checkpoint_dir.exists():
    print(f"❌ 错误：Checkpoint 路径不存在！")
    print(f"请检查路径是否正确")
    sys.exit(1)
else:
    print(f"✓ Checkpoint 目录存在")
    
    # 列出 checkpoint 目录内容
    print(f"\nCheckpoint 目录内容:")
    for item in checkpoint_dir.iterdir():
        print(f"  - {item.name}")

# 检查环境文件
print(f"\n检查环境文件...")
try:
    from qln_multi_flight_game_v2 import MultiFlightGymEnv
    print(f"✓ 环境文件导入成功")
except Exception as e:
    print(f"❌ 错误：无法导入环境文件")
    print(f"错误信息: {e}")
    sys.exit(1)

# ====================== 第2步：初始化 Ray ======================
print("\n" + "="*80)
print("第2步：初始化 Ray")
print("="*80)

print("\n配置 Ray 参数...")
ray_config = {
    "ignore_reinit_error": True,
    "log_to_driver": False,
    "num_cpus": 4,  # 限制 CPU 数量，避免资源问题
    "num_gpus": 0,
    "object_store_memory": 1000000000,  # 1GB
    "_temp_dir": "/tmp/ray",
}

print(f"Ray 配置: {ray_config}")
print(f"\n正在初始化 Ray...")

try:
    ray.init(**ray_config)
    print(f"✓ Ray 初始化成功")
    print(f"  Ray 版本: {ray.__version__}")
    print(f"  可用资源: {ray.available_resources()}")
except Exception as e:
    print(f"❌ Ray 初始化失败: {e}")
    sys.exit(1)

# ====================== 第3步：注册环境 ======================
print("\n" + "="*80)
print("第3步：注册环境")
print("="*80)

print("\n正在注册环境 'multi_flight_v0'...")
try:
    tune.register_env("multi_flight_v0", lambda config: MultiFlightGymEnv())
    print("✓ 环境注册成功")
    
    # 测试环境是否可以创建
    test_env = MultiFlightGymEnv()
    print(f"✓ 环境测试创建成功")
    print(f"  观测空间: {test_env.observation_space}")
    print(f"  动作空间: {test_env.action_space}")
except Exception as e:
    print(f"❌ 环境注册或创建失败: {e}")
    ray.shutdown()
    sys.exit(1)

# ====================== 第4步：加载 Checkpoint ======================
print("\n" + "="*80)
print("第4步：加载 Checkpoint（这一步可能需要1-2分钟）")
print("="*80)

policy = None
algo = None

# ------------------- 方法1：最标准的方式 -------------------
print("\n尝试方法1: Algorithm.from_checkpoint()...")
print("  这一步可能需要30-60秒，请耐心等待...")

start_time = time.time()
try:
    print("  [进行中] 正在加载 checkpoint...")
    
    # 添加超时保护
    import signal
    
    class TimeoutError(Exception):
        pass
    
    def timeout_handler(signum, frame):
        raise TimeoutError("加载超时")
    
    # 设置120秒超时（2分钟）
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(120)
    
    try:
        algo = Algorithm.from_checkpoint(checkpoint_path)
        signal.alarm(0)  # 取消超时
        
        elapsed = time.time() - start_time
        print(f"  ✓ Algorithm 加载成功！耗时 {elapsed:.1f} 秒")
        
        # 尝试获取 policy
        print("  [进行中] 正在获取 policy...")
        try:
            policy = algo.get_policy("default_policy")
            print(f"  ✓ Policy 获取成功 (ID: default_policy)")
        except:
            # 尝试获取第一个可用的 policy
            try:
                all_policies = algo.workers.local_worker().policy_map
                policy_id = list(all_policies.keys())[0]
                policy = algo.get_policy(policy_id)
                print(f"  ✓ Policy 获取成功 (ID: {policy_id})")
            except Exception as e:
                print(f"  ⚠ 无法获取 policy，但可以使用 compute_actions: {e}")
    
    except TimeoutError:
        signal.alarm(0)
        print(f"  ❌ 加载超时（超过120秒）")
        print(f"     这可能是因为:")
        print(f"     1. Checkpoint 文件过大")
        print(f"     2. 系统资源不足")
        print(f"     3. Checkpoint 文件损坏")
        algo = None
    
except Exception as e:
    signal.alarm(0)
    elapsed = time.time() - start_time
    print(f"  ❌ 方法1失败 (耗时 {elapsed:.1f} 秒): {e}")
    algo = None

# ------------------- 方法2：轻量级加载 -------------------
if algo is None:
    print("\n尝试方法2: 轻量级加载（禁用 workers）...")
    try:
        from ray.rllib.algorithms.ppo import PPOConfig
        
        # 创建最小配置
        config = (
            PPOConfig()
            .environment("multi_flight_v0")
            .env_runners(
                num_env_runners=0,  # 推理时不需要额外的 workers
                num_envs_per_env_runner=1,
            )
            .resources(num_gpus=0)
        )
        
        algo = config.build()
        algo.restore(checkpoint_path)
        
        print(f"  ✓ 轻量级加载成功")
        policy = algo.get_policy("default_policy")
        
    except Exception as e:
        print(f"  ❌ 方法2失败: {e}")
        algo = None

# ------------------- 方法3：直接加载 Policy 权重 -------------------
if algo is None and policy is None:
    print("\n尝试方法3: 直接加载 Policy 权重...")
    try:
        # 尝试从 checkpoint 目录中找到 policy 文件
        policy_checkpoint = checkpoint_dir / "policies" / "default_policy"
        if policy_checkpoint.exists():
            policy = Policy.from_checkpoint(str(policy_checkpoint))
            print(f"  ✓ Policy 权重加载成功")
        else:
            print(f"  ❌ 找不到 policy 文件")
    except Exception as e:
        print(f"  ❌ 方法3失败: {e}")

# ------------------- 最终检查 -------------------
if policy is None and algo is None:
    print("\n" + "="*80)
    print("❌ 所有加载方法都失败了")
    print("="*80)
    print("\n可能的原因:")
    print("1. Checkpoint 文件损坏或不完整")
    print("2. Ray RLlib 版本不兼容")
    print("3. 环境定义与训练时不一致")
    print("\n建议:")
    print("1. 重新训练模型并保存 checkpoint")
    print("2. 检查 Ray RLlib 版本: pip show ray")
    print("3. 使用 algo.save_to_path() 重新导出模型")
    
    ray.shutdown()
    sys.exit(1)

print("\n" + "="*80)
print("✓✓✓ Checkpoint 加载成功！✓✓✓")
print("="*80)

if algo:
    print(f"Algorithm 类型: {type(algo).__name__}")
if policy:
    print(f"Policy 类型: {type(policy).__name__}")

# ====================== 第5步：动作选择函数 ======================
print("\n" + "="*80)
print("第5步：设置推理函数")
print("="*80)

def select_action(state):
    """共享策略：两个航班观测相同，动作也相同"""
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
    
    raise RuntimeError("无可用的 policy 或 algorithm")

# 测试推理
print("\n测试推理功能...")
try:
    env = MultiFlightGymEnv()
    obs, _ = env.reset()
    test_state = obs["airline_0"]
    
    print(f"  测试观测值: {test_state[:5]}... (前5个值)")
    
    a0, a1 = select_action(test_state)
    print(f"  ✓ 推理成功！动作: airline_0={a0}, airline_1={a1}")
except Exception as e:
    print(f"  ❌ 推理测试失败: {e}")
    ray.shutdown()
    sys.exit(1)

# ====================== 第6步：正式推理 ======================
print("\n" + "="*80)
print("第6步：开始正式推理")
print("="*80)

parser = argparse.ArgumentParser(description='多航班收益管理推理')
parser.add_argument('--episodes', type=int, default=10, help='测试回合数')
parser.add_argument('--flight1_days', type=int, default=30, help='航班1剩余天数')
parser.add_argument('--flight1_capacity', type=int, default=197, help='航班1剩余座位数')
parser.add_argument('--flight2_days', type=int, default=30, help='航班2剩余天数')
parser.add_argument('--flight2_capacity', type=int, default=230, help='航班2剩余座位数')
args = parser.parse_args()

# --------------------- 完整回合测试 ---------------------
print(f"\n测试策略（{args.episodes} 个回合）:")
test_episodes = args.episodes
total_rewards_list = []
f1_revenues, f2_revenues = [], []
f1_lfs, f2_lfs = [], []

for ep in range(test_episodes):
    obs, _ = env.reset()
    state = obs["airline_0"]
    done = False
    step_count = 0
    
    while not done:
        a0, a1 = select_action(state)
        action_dict = {"airline_0": a0, "airline_1": a1}
        obs, rewards, dones, _, infos = env.step(action_dict)
        state = obs["airline_0"]
        done = dones["__all__"]
        step_count += 1

    total_r = env.core.total_revenue
    total_rewards_list.append(total_r)
    f1_revenues.append(env.core.revenue1)
    f2_revenues.append(env.core.revenue2)
    f1_lfs.append(env.core.sold1 / env.core.init_cap1)
    f2_lfs.append(env.core.sold2 / env.core.init_cap2)

    print(f"  第 {ep+1:2d} 回合 ({step_count:2d} 步) → 总收益 {total_r:,.0f} | "
          f"航班1 {env.core.revenue1:,.0f} (负载 {env.core.sold1/env.core.init_cap1:.2%}) | "
          f"航班2 {env.core.revenue2:,.0f} (负载 {env.core.sold2/env.core.init_cap2:.2%})")

print("\n统计总结:")
print(f"   平均总收益: {np.mean(total_rewards_list):,.0f} ± {np.std(total_rewards_list):,.0f}")
print(f"   最高总收益: {max(total_rewards_list):,.0f}")
print(f"   航班1 平均收益: {np.mean(f1_revenues):,.0f}  平均负载: {np.mean(f1_lfs):.2%}")
print(f"   航班2 平均收益: {np.mean(f2_revenues):,.0f}  平均负载: {np.mean(f2_lfs):.2%}")

# --------------------- 最优回合详细过程 ---------------------
print(f"\n最优回合详细决策过程（从 20 局中选收益最高的一局）:")
best_reward = -1
best_data = None
search_rounds = 20

for _ in range(search_rounds):
    obs, _ = env.reset()
    state = obs["airline_0"]
    episode_records = []
    prev_sold1 = prev_sold2 = 0

    while True:
        a0, a1 = select_action(state)
        action_dict = {"airline_0": a0, "airline_1": a1}
        obs, rewards, dones, _, infos = env.step(action_dict)

        sold1_now = env.core.sold1
        sold2_now = env.core.sold2
        tickets1 = sold1_now - prev_sold1
        tickets2 = sold2_now - prev_sold2
        prev_sold1, prev_sold2 = sold1_now, sold2_now

        p1 = infos["airline_0"]["price"]
        p2 = infos["airline_1"]["price"]

        record = {
            "day": env.core.init_days - env.core.day_left + 1,
            "p1": p1, "p2": p2,
            "t1": tickets1, "t2": tickets2,
            "lf1": env.core.sold1 / env.core.init_cap1,
            "lf2": env.core.sold2 / env.core.init_cap2,
            "rev1": rewards["airline_0"],
            "rev2": rewards["airline_1"],
        }
        episode_records.append(record)
        state = obs["airline_0"]
        if dones["__all__"]:
            total = env.core.total_revenue
            if total > best_reward:
                best_reward = total
                best_data = episode_records
            break

if best_data:
    print(f"   最佳总收益: {best_reward:,.0f}")
    print("   天数  航班1价格  航班2价格  航班1售票  航班2售票  航班1负载   航班2负载   当天收益1   当天收益2   当天总收益")
    print("   ----  ---------  ---------  ---------  ---------  ----------  ----------  ----------  ----------  ----------")
    for r in best_data:
        print(f"   {r['day']:3d}   {r['p1']:8.0f}   {r['p2']:8.0f}   "
              f"{r['t1']:8d}   {r['t2']:8d}   {r['lf1']:8.2%}   {r['lf2']:8.2%}   "
              f"{r['rev1']:9.0f}   {r['rev2']:9.0f}   {r['rev1']+r['rev2']:9.0f}")

print("\n" + "="*80)
print("✓✓✓ 推理完成！✓✓✓")
print("="*80)

# 清理
ray.shutdown()
