import os
import numpy as np
import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_env

import config
# OVERRIDE per l'evaluation
config.ALPHA = 0.5
config.NUM_WORKERS = 5
config.CLOUD_AVAILABLE = False

from env.edge_continuum import EdgeContinuumEnv
from train_autonomous import env_creator, RandomPolicy

# ==========================================
# IL CHECKPOINT FINALE DELLA FASE 5
# ==========================================
CHECKPOINT_PATH = "/var/folders/gb/b_8vls29731bgmdz63sxbtnw0000gn/T/tmphjzo0w03" 

# ==============================================================================
# IL VIGILE URBANO (Fase 5: 5 Smart, 0 Random)
# ==============================================================================
def policy_mapping_fn(agent_id, episode, worker, **kwargs):
    if agent_id == "worker_0": return "ppo_policy_0"
    elif agent_id == "worker_1": return "ppo_policy_1"
    elif agent_id == "worker_2": return "ppo_policy_2"
    elif agent_id == "worker_3": return "ppo_policy_3"
    elif agent_id == "worker_4": return "ppo_policy_4"
    else: return "random_policy"

if __name__ == "__main__":
    ray.init()
    register_env("edge_continuum_auto_env", env_creator)

    temp_env = env_creator({})
    obs_space = temp_env.observation_space["worker_0"]
    act_space = temp_env.action_space["worker_0"]

    # ==============================================================================
    # IL DIZIONARIO DELLE POLICY (5 cervelli indipendenti)
    # ==============================================================================
    policies = {
        "ppo_policy_0": (None, obs_space, act_space, {}),
        "ppo_policy_1": (None, obs_space, act_space, {}),
        "ppo_policy_2": (None, obs_space, act_space, {}),
        "ppo_policy_3": (None, obs_space, act_space, {}),
        "ppo_policy_4": (None, obs_space, act_space, {}),
        "random_policy": (RandomPolicy, obs_space, act_space, {})
    }

    algo_config = (
        PPOConfig()
        .api_stack(enable_rl_module_and_learner=False, enable_env_runner_and_connector_v2=False)
        .environment(env="edge_continuum_auto_env")
        .multi_agent(
            policies=policies,
            policy_mapping_fn=policy_mapping_fn,
        )
    )
    
    algo = algo_config.build()
    algo.restore(CHECKPOINT_PATH)
    print(f"\nCheckpoint FINALE (FASE 5) caricato con successo da: {CHECKPOINT_PATH}")

    env = EdgeContinuumEnv()
    obs, info = env.reset()
    
    dones = {a: False for a in env.possible_agents}
    
    tasks_total = 0
    tasks_rejected = 0
    tasks_type_counts = {0: 0, 1: 0, 2: 0}
    tasks_type_rejected = {0: 0, 1: 0, 2: 0}

    print("Inizio simulazione di Evaluation (Fase 5: 5 Smart, 100% MARL)...")
    
    while not all(dones.values()):
        actions = {}
        for agent_id, agent_obs in obs.items():
            policy_id = policy_mapping_fn(agent_id, 0, None)
            actions[agent_id] = algo.compute_single_action(
                observation=agent_obs,
                policy_id=policy_id
            )
            
        obs, rewards, terminations, truncations, infos = env.step(actions)
        dones = {a: terminations[a] or truncations[a] for a in env.possible_agents}
        
        tasks_total += 1
        current_type = env.current_task_type
        tasks_type_counts[current_type] += 1
        
        if any(info.get("rejected", False) for info in infos.values()):
            tasks_rejected += 1
            tasks_type_rejected[current_type] += 1

    gamma = ((tasks_total - tasks_rejected) / max(1, tasks_total)) * 100
    gamma_0 = ((tasks_type_counts[0] - tasks_type_rejected[0]) / max(1, tasks_type_counts[0])) * 100
    gamma_1 = ((tasks_type_counts[1] - tasks_type_rejected[1]) / max(1, tasks_type_counts[1])) * 100
    gamma_2 = ((tasks_type_counts[2] - tasks_type_rejected[2]) / max(1, tasks_type_counts[2])) * 100
    
    norm_batts = [env.batteries[a] / env.initial_batteries[i] for i, a in enumerate(env.possible_agents)]
    sigma = np.var(norm_batts)
    delta = max(norm_batts) - min(norm_batts)
    m = env.timestep

    print("\n" + "="*95)
    print(" RIGA PRONTA PER LA TABELLA DELLA TESI (FASE 5: 5 Smart, 0 Random)")
    print("="*95)
    print(f" Alpha |   σ    |   δ    |   m   |   γ   |  γ_0  |  γ_1  |  γ_2  |  ts   |")
    print("-" * 95)
    print(f" {config.ALPHA:.2f}  | {sigma:.4f} | {delta:.2f}   |  {m:4d} | {gamma:5.1f} | {gamma_0:5.1f} | {gamma_1:5.1f} | {gamma_2:5.1f} | {tasks_total:5d} |")
    print("="*95)
    
    ray.shutdown()