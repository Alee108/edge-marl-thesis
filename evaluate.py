import os
import numpy as np
import ray
from ray.rllib.algorithms.algorithm import Algorithm
from ray.tune.registry import register_env
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv

# Importiamo l'ambiente e le configurazioni
from env.edge_continuum import EdgeContinuumEnv
import config

config.MAX_TIMESTEPS = 999999

def env_creator(args):
    return ParallelPettingZooEnv(EdgeContinuumEnv())

if __name__ == "__main__":
    ray.init(log_to_driver=False)
    register_env("edge_continuum_env", env_creator)

    # =================================================================
    # ATTENZIONE: INSERISCI QUI IL PATH DEL CHECKPOINT
    # =================================================================
    CHECKPOINT_PATH = "/var/folders/gb/b_8vls29731bgmdz63sxbtnw0000gn/T/tmptnk5vyho"

    print(f"\nCaricamento modello da: {CHECKPOINT_PATH}...")
    try:
        algo = Algorithm.from_checkpoint(CHECKPOINT_PATH)
    except Exception as e:
        print(f"Errore: {e}")
        exit()

    env = EdgeContinuumEnv()
    obs, info = env.reset()

    ts = 0                  
    first_death_step = None 
    delta = 0.0             
    
    total_tasks = 0
    executed_tasks = 0
    total_by_type = {0: 0, 1: 0, 2: 0}
    executed_by_type = {0: 0, 1: 0, 2: 0}

    print("\nInizio simulazione di valutazione...")

    while env.agents:
        actions = {}
        for agent in env.agents:
            action = algo.compute_single_action(
                observation=obs[agent],
                policy_id="shared_policy",
                explore=False 
            )
            actions[agent] = action
            
        obs, rewards, terminations, truncations, infos = env.step(actions)
        
        if infos:
            first_worker = list(infos.keys())[0]
            task_info = infos[first_worker]
            t_type = task_info["task_type"]
            
            total_tasks += 1
            total_by_type[t_type] += 1
            
            if not task_info["rejected"] and not task_info["to_cloud"]:
                executed_tasks += 1
                executed_by_type[t_type] += 1

        ts += len(env.agents)
        
        if first_death_step is None and any(b <= 0 for b in env.batteries.values()):
            first_death_step = env.timestep
            max_b = max(env.batteries.values())
            min_b = min(env.batteries.values())
            delta = max_b - min_b
            
    batteries_list = list(env.batteries.values())
    sigma = np.var(batteries_list)
    
    gamma = (executed_tasks / total_tasks) * 100 if total_tasks > 0 else 0.0
    gamma_0 = (executed_by_type[0] / total_by_type[0]) * 100 if total_by_type[0] > 0 else 0.0
    gamma_1 = (executed_by_type[1] / total_by_type[1]) * 100 if total_by_type[1] > 0 else 0.0
    gamma_2 = (executed_by_type[2] / total_by_type[2]) * 100 if total_by_type[2] > 0 else 0.0

    # =================================================================
    # CALCOLO m e M (Lifespans)
    # =================================================================
    m_lifespan = env.timestep # Il momento in cui muore il primo nodo
    
    projected_lifespans = []
    for a in env.possible_agents:
        idx = int(a.split("_")[1])
        initial_b = env.initial_batteries[idx]
        remaining_b = env.batteries[a]
        consumed = initial_b - remaining_b
        
        if consumed > 0:
            # Calcoliamo il consumo medio per step e proiettiamo la vita residua
            avg_drain_rate = consumed / m_lifespan
            projected_m = m_lifespan + (remaining_b / avg_drain_rate)
            projected_lifespans.append(projected_m)
        else:
            projected_lifespans.append(m_lifespan)
            
    M_lifespan = max(projected_lifespans)

    # =================================================================
    # STAMPA DELLA RIGA PER LA TABELLA DELLA TESI
    # =================================================================
    print("\n" + "="*85)
    print(" RIGA PRONTA PER LA TABELLA DELLA TESI")
    print("="*85)
    print(" Alpha |   σ    |   δ   |   m   |   M   |   γ   |  γ_0  |  γ_1  |  γ_2  |   ts  |")
    print("-" * 85)
    print(f"  {config.ALPHA:.2f}  | {sigma:.4f} | {delta:.2f}  | {int(m_lifespan):>5} | {int(M_lifespan):>5} | {gamma:>5.1f} | {gamma_0:>5.1f} | {gamma_1:>5.1f} | {gamma_2:>5.1f} | {ts:>5} |")
    print("="*85 + "\n")

    ray.shutdown()