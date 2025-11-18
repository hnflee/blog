#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
多航班动态定价推理 - 修复循环导入版本
文件重命名为 flight_inference.py 避免与 qln_multi_flight_game_v2.py 冲突
"""

import os
import sys
import warnings

# 静默配置
os.environ["RAY_DISABLE_METRICS"] = "1"
os.environ["RAY_DEDUP_LOGS"] = "0"
os.environ["PYTHONWARNINGS"] = "ignore"
warnings.filterwarnings("ignore")

print("="*80)
print("          多航班动态定价推理 - 修复版本")
print("="*80)

# ====================== 配置 ======================
checkpoint_path = '/Users/lifeng/ray_results/FlightPPO_Final_2025/PPO_multi_flight_v0_f4f6f_00000_0_2025-11-18_16-31-19/checkpoint_000000'

# ====================== 1. 验证文件 ======================
print("\n[1/5] 验证文件...")

from pathlib import Path
if not Path(checkpoint_path).exists():
    print(f"❌ Checkpoint 不存在: {checkpoint_path}")
    print("\n请检查路径是否正确")
    sys.exit(1)
print("✓ Checkpoint 文件存在")

# 修复循环导入：延迟导入环境
print("✓ 准备导入环境...")

# ====================== 2. 初始化 Ray（必须在导入环境前）======================
print("\n[2/5] 初始化 Ray...")

import ray
ray.init(
    ignore_reinit_error=True,
    log_to_driver=False,
    num_cpus=2,
    num_gpus=0,
    _system_config={
        "automatic_object_spilling_enabled": False,
        "max_io_workers": 2,
    }
)
print(f"✓ Ray {ray.__version__} 初始化成功")

# ====================== 3. 现在安全地导入和注册环境 ======================
print("\n[3/5] 导入和注册环境...")

try:
    # 使用绝对导入
    import qln_multi_flight_game_v2
    from qln_multi_flight_game_v2 import MultiFlightGymEnv
    print("✓ 环境模块导入成功")
except ImportError as e:
    print(f"❌ 无法导入环境模块: {e}")
    print("\n可能的原因:")
    print("1. qln_multi_flight_game_v2.py 不在当前目录")
    print("2. 环境文件有语法错误")
    print("3. 缺少依赖包")
    print(f"\n当前目录: {os.getcwd()}")
    print(f"Python 路径: {sys.path[:3]}")
    ray.shutdown()
    sys.exit(1)

from ray import tune
tune.register_env("multi_flight_v0", lambda config: MultiFlightGymEnv())
print("✓ 环境 'multi_flight_v0' 注册成功")

# ====================== 4. 加载模型 ======================
print("\n[4/5] 加载模型...")
print("提示: 这一步可能需要30-90秒，请耐心等待...\n")

algo = None
policy = None

# 方法1：轻量级配置（推荐）
try:
    print("[方法1] 使用轻量级配置加载...")
    
    from ray.rllib.algorithms.ppo import PPOConfig
    
    config = (
        PPOConfig()
        .environment("multi_flight_v0")
        .env_runners(
            num_env_runners=0,  # 推理时不需要额外的 workers
            num_envs_per_env_runner=1,
        )
        .resources(num_gpus=0)
        .debugging(log_level="ERROR")
    )
    
    print("  → 创建 Algorithm 对象...")
    algo = config.build()
    
    print(f"  → 从 checkpoint 恢复权重...")
    algo.restore(checkpoint_path)
    
    print("  → 获取 policy...")
    policy = algo.get_policy("default_policy")
    
    print("✓ 方法1加载成功！\n")
    
except Exception as e:
    print(f"✗ 方法1失败: {e}\n")
    
    # 方法2：标准加载
    try:
        print("[方法2] 使用标准方式加载...")
        from ray.rllib.algorithms.algorithm import Algorithm
        
        algo = Algorithm.from_checkpoint(checkpoint_path)
        
        try:
            policy = algo.get_policy("default_policy")
        except:
            all_policies = algo.workers.local_worker().policy_map
            policy_id = list(all_policies.keys())[0]
            policy = algo.get_policy(policy_id)
        
        print("✓ 方法2加载成功！\n")
        
    except Exception as e2:
        print(f"✗ 方法2也失败: {e2}\n")
        print("❌ 所有加载方法都失败")
        ray.shutdown()
        sys.exit(1)

if not algo and not policy:
    print("❌ 无法加载模型")
    ray.shutdown()
    sys.exit(1)

# ====================== 5. 推理函数 ======================
print("[5/5] 设置推理...")

import numpy as np

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
    
    raise RuntimeError("无可用的 policy 或 algorithm")

# 测试推理
env = MultiFlightGymEnv()
obs, _ = env.reset()
a0, a1 = select_action(obs["airline_0"])
print(f"✓ 推理测试成功！示例动作: {a0}, {a1}")

# ====================== 参数解析 ======================
import argparse

parser = argparse.ArgumentParser(description='多航班收益管理推理')
parser.add_argument('--episodes', type=int, default=10, help='测试回合数')
parser.add_argument('--flight1_days', type=int, default=30, help='航班1剩余天数')
parser.add_argument('--flight1_capacity', type=int, default=197, help='航班1剩余座位数')
parser.add_argument('--flight2_days', type=int, default=30, help='航班2剩余天数')
parser.add_argument('--flight2_capacity', type=int, default=230, help='航班2剩余座位数')
args = parser.parse_args()

is_custom = (
    args.flight1_days != env.core.init_days or
    args.flight1_capacity != env.core.init_cap1 or
    args.flight2_days != env.core.init_days or
    args.flight2_capacity != env.core.init_cap2
)

# ====================== 开始推理 ======================
print("\n" + "="*80)
print("开始推理测试")
print("="*80)

test_episodes = args.episodes
total_rewards_list = []
f1_revenues, f2_revenues = [], []
f1_lfs, f2_lfs = [], []

print(f"\n运行 {test_episodes} 个完整回合...")
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

# ====================== 最优回合详细过程 ======================
if not is_custom:
    print("\n" + "="*80)
    print("最优回合详细决策过程")
    print("="*80)
    
    best_reward = -1
    best_data = None
    search_rounds = 20
    
    print(f"正在搜索 {search_rounds} 个回合中的最优策略...")
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
        print(f"\n最佳总收益: {best_reward:,.0f}")
        print("\n天数  航班1价格  航班2价格  航班1售票  航班2售票  航班1负载   航班2负载   当天收益1   当天收益2   当天总收益")
        print("----  ---------  ---------  ---------  ---------  ----------  ----------  ----------  ----------  ----------")
        for r in best_data:
            print(f"{r['day']:3d}   {r['p1']:8.0f}   {r['p2']:8.0f}   "
                  f"{r['t1']:8d}   {r['t2']:8d}   {r['lf1']:8.2%}   {r['lf2']:8.2%}   "
                  f"{r['rev1']:9.0f}   {r['rev2']:9.0f}   {r['rev1']+r['rev2']:9.0f}")

# ====================== 自定义起始状态模拟 ======================
if is_custom:
    print("\n" + "="*80)
    print("自定义起始状态模拟")
    print("="*80)
    
    print(f"自定义参数:")
    print(f"  航班1: {args.flight1_days} 天, {args.flight1_capacity} 座位")
    print(f"  航班2: {args.flight2_days} 天, {args.flight2_capacity} 座位\n")
    
    env.core.day_left = min(args.flight1_days, args.flight2_days)
    env.core.cap1 = args.flight1_capacity
    env.core.sold1 = env.core.init_cap1 - args.flight1_capacity
    env.core.cap2 = args.flight2_capacity
    env.core.sold2 = env.core.init_cap2 - args.flight2_capacity
    env.core.last_p1 = env.core.last_p2 = 1000

    obs = {agent: env.core._get_obs() for agent in env.agents}
    state = obs["airline_0"]
    custom_records = []
    prev_sold1, prev_sold2 = env.core.sold1, env.core.sold2

    while True:
        a0, a1 = select_action(state)
        action_dict = {"airline_0": a0, "airline_1": a1}
        obs, rewards, dones, _, infos = env.step(action_dict)

        sold1_now = env.core.sold1
        sold2_now = env.core.sold2
        t1 = sold1_now - prev_sold1
        t2 = sold2_now - prev_sold2
        prev_sold1, prev_sold2 = sold1_now, sold2_now

        p1 = infos["airline_0"]["price"]
        p2 = infos["airline_1"]["price"]

        custom_records.append({
            "day": env.core.init_days - env.core.day_left + 1,
            "p1": p1, "p2": p2, "t1": t1, "t2": t2,
            "lf1": env.core.sold1 / env.core.init_cap1,
            "lf2": env.core.sold2 / env.core.init_cap2,
            "rev1": rewards["airline_0"],
            "rev2": rewards["airline_1"],
        })
        state = obs["airline_0"]
        if dones["__all__"]:
            break

    total_custom = env.core.total_revenue
    print(f"自定义状态总收益: {total_custom:,.0f}")
    print(f"最终负载因子 → 航班1: {env.core.sold1/env.core.init_cap1:.2%}   航班2: {env.core.sold2/env.core.init_cap2:.2%}\n")
    print("天数  价格1    价格2   售票1  售票2  负载1    负载2    收益1     收益2     小计")
    print("----  -------  -------  -----  -----  -------  -------  --------  --------  --------")
    for r in custom_records:
        print(f"{r['day']:3d}  {r['p1']:7.0f}  {r['p2']:7.0f}  "
              f"{r['t1']:5d}  {r['t2']:5d}  {r['lf1']:6.2%}  {r['lf2']:6.2%}  "
              f"{r['rev1']:8.0f}  {r['rev2']:8.0f}  {r['rev1']+r['rev2']:8.0f}")

print("\n" + "="*80)
print("✓ 推理完成！")
print("="*80)

ray.shutdown()
