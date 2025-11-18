#!/usr/bin/env python
# -*- coding: utf-8 -*-

import ray
import os
import numpy as np
import argparse
from ray import tune
from ray.rllib.policy.policy import Policy
from ray.rllib.algorithms.algorithm import Algorithm
from qln_multi_flight_game_v2 import MultiFlightGymEnv  # 你的环境文件

os.environ["RAY_DISABLE_METRICS"] = "1"
ray.init(ignore_reinit_error=True, log_to_driver=False)

# ====================== 重要：先注册环境 ======================
print("正在注册自定义环境...")
tune.register_env("multi_flight_v0", lambda config: MultiFlightGymEnv())
print("✓ 环境注册成功！")

# ====================== 1. 万能加载最优 checkpoint（2025 终极版）======================
print("正在加载训练好的 PPO 模型 checkpoint...")

checkpoint_path = '/Users/lifeng/ray_results/FlightPPO_Final_2025/PPO_multi_flight_v0_f4f6f_00000_0_2025-11-18_16-31-19/checkpoint_000000'

# ------------------- 终极容错加载逻辑 -------------------
policy = None
algo = None

# 方法1：最推荐的方式（RLlib 2.x+）
if policy is None:
    try:
        algo = Algorithm.from_checkpoint(checkpoint_path)
        policy = algo.get_policy("default_policy")
        print("✓ 通过 Algorithm.from_checkpoint + get_policy('default_policy') 加载成功")
    except Exception as e:
        print(f"  方法1失败: {e}")
        algo = None

# 方法2：尝试获取第一个可用的 policy
if policy is None and algo is not None:
    try:
        # 如果 algo 已经加载，尝试获取任意可用的 policy
        all_policies = algo.workers.local_worker().policy_map
        policy_id = list(all_policies.keys())[0]
        policy = algo.get_policy(policy_id)
        print(f"✓ 通过获取第一个 policy 加载成功（policy_id: {policy_id}）")
    except Exception as e:
        print(f"  方法2失败: {e}")

# 方法3：尝试直接加载 Algorithm 并使用 compute_actions
if algo is None:
    try:
        algo = Algorithm.from_checkpoint(checkpoint_path)
        print("✓ 通过 Algorithm.from_checkpoint 加载成功（将使用 compute_actions）")
    except Exception as e:
        print(f"  方法3失败: {e}")

# 方法4：可选命中率最高的后备方案（极少数极端旧版本）
if policy is None and algo is None:
    try:
        from ray.rllib.utils.checkpoints import get_checkpoint_info
        info = get_checkpoint_info(checkpoint_path)
        policy_id = info["policy_ids"][0] if info.get("policy_ids") else "default_policy"
        algo = Algorithm.from_checkpoint(checkpoint_path)
        policy = algo.get_policy(policy_id)
        print(f"✓ 通过 get_checkpoint_info 方式加载成功（policy_id: {policy_id}）")
    except Exception as e:
        print(f"  方法4失败: {e}")

# 方法5：尝试 Policy.from_checkpoint（某些版本支持）
if policy is None:
    try:
        policy = Policy.from_checkpoint(checkpoint_path)
        print("✓ 通过 Policy.from_checkpoint 加载成功")
    except Exception as e:
        print(f"  方法5失败: {e}")

if policy is None and algo is None:
    raise RuntimeError("经过所有方式都无法加载 Policy，请检查 checkpoint 路径或重新导出模型")

print("✓ 最终成功获得可用的 Policy/Algorithm 对象！")
if policy:
    print(f"   Policy 类型: {type(policy)}")
if algo:
    print(f"   Algorithm 类型: {type(algo)}")
print()

# ====================== 2. 统一的动作选择函数 ======================
def select_action(state):
    """共享策略：两个航班观测相同，动作也相同"""
    obs_dict = {"airline_0": state, "airline_1": state}
    
    if algo is not None:
        # Algorithm 模式更快（支持 batch 推理）
        try:
            actions = algo.compute_actions(obs_dict, explore=False)
            # compute_actions 可能返回不同格式，需要处理
            if isinstance(actions, dict):
                return actions["airline_0"], actions["airline_1"]
            else:
                # 如果返回的是 tuple (actions, states, infos)
                action_dict = actions[0] if isinstance(actions, tuple) else actions
                return action_dict["airline_0"], action_dict["airline_1"]
        except:
            pass
    
    # Policy 模式一定可用（fallback）
    if policy is not None:
        a0, _, _ = policy.compute_single_action(state, explore=False)
        a1, _, _ = policy.compute_single_action(state, explore=False)
        return a0, a1
    
    # 最后的 fallback：使用 algo.compute_single_action
    a0 = algo.compute_single_action(state, policy_id="default_policy", explore=False)
    a1 = algo.compute_single_action(state, policy_id="default_policy", explore=False)
    if isinstance(a0, tuple):
        a0 = a0[0]
    if isinstance(a1, tuple):
        a1 = a1[0]
    return a0, a1

# ====================== 3. 环境与参数 ======================
env = MultiFlightGymEnv()

parser = argparse.ArgumentParser(description='多航班收益管理推理（支持自定义起始状态）')
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

# ====================== 4. 正式推理开始 ======================
print("\n" + "="*80)
print("          使用训练好的最优模型进行多航班动态定价推理")
print("="*80)

# --------------------- 1. 标准完整回合测试 ---------------------
print("\n1. 最终策略测试（10 个完整回合）:")
test_episodes = 10
total_rewards_list = []
f1_revenues, f2_revenues = [], []
f1_lfs, f2_lfs = [], []

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
          f"航班1 {env.core.revenue1:,.0f} (负载 {env.core.sold1/env.core.init_cap1:.2%}) | "
          f"航班2 {env.core.revenue2:,.0f} (负载 {env.core.sold2/env.core.init_cap2:.2%})")

print("\n2. 测试统计总结:")
print(f"   平均总收益: {np.mean(total_rewards_list):,.0f} ± {np.std(total_rewards_list):,.0f}")
print(f"   最高总收益: {max(total_rewards_list):,.0f}")
print(f"   航班1 平均收益: {np.mean(f1_revenues):,.0f}  平均负载: {np.mean(f1_lfs):.2%}")
print(f"   航班2 平均收益: {np.mean(f2_revenues):,.0f}  平均负载: {np.mean(f2_lfs):.2%}")

# --------------------- 3. 最优回合详细过程 ---------------------
print("\n3. 最优回合详细决策过程（从 20 局中选收益最高的一局）:")
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
    print(f"   最佳总收益: {best_reward:,.0f}（在 {search_rounds} 局中）")
    print("   天数  航班1价格  航班2价格  航班1售票  航班2售票  航班1负载   航班2负载   当天收益1   当天收益2   当天总收益")
    print("   ----  ---------  ---------  ---------  ---------  ----------  ----------  ----------  ----------  ----------")
    for r in best_data:
        print(f"   {r['day']:3d}   {r['p1']:8.0f}   {r['p2']:8.0f}   "
              f"{r['t1']:8d}   {r['t2']:8d}   {r['lf1']:8.2%}   {r['lf2']:8.2%}   "
              f"{r['rev1']:9.0f}   {r['rev2']:9.0f}   {r['rev1']+r['rev2']:9.0f}")

# --------------------- 4. 自定义起始状态模拟 ---------------------
if is_custom:
    print("\n" + "="*80)
    print("自定义起始状态模拟")
    print("="*80)
    
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
    print(f"   自定义状态总收益: {total_custom:,.0f}")
    print(f"   最终负载因子 → 航班1: {env.core.sold1/env.core.init_cap1:.2%}   航班2: {env.core.sold2/env.core.init_cap2:.2%}")
    print("   天数  价格1    价格2   售票1  售票2  负载1    负载2    收益1     收益2     小计")
    for r in custom_records:
        print(f"   {r['day']:3d}  {r['p1']:7.0f}  {r['p2']:7.0f}  "
              f"{r['t1']:5d}  {r['t2']:5d}  {r['lf1']:6.2%}  {r['lf2']:6.2%}  "
              f"{r['rev1']:8.0f}  {r['rev2']:8.0f}  {r['rev1']+r['rev2']:8.0f}")

print("\n" + "="*80)
print("多航班动态博弈推理全部完成！策略已就绪，收益起飞！")
print("="*80)

# 清理资源
ray.shutdown()
