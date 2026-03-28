import os
import numpy as np
import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_env
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.rllib.policy.policy import Policy

# ==============================================================================
# 1. OVERRIDE DINAMICO DEL CONFIG
# ==============================================================================
import config
config.ALPHA = 0.5
config.NUM_WORKERS = 5
config.CLOUD_AVAILABLE = False

from env.edge_continuum import EdgeContinuumEnv

# ==============================================================================
# 2. DEFINIZIONE DELLA RANDOM POLICY (Mantenuta per compatibilità, ma non la useremo!)
# ==============================================================================
class RandomPolicy(Policy):
    def __init__(self, observation_space, action_space, config):
        super().__init__(observation_space, action_space, config)

    def compute_actions(self, obs_batch, *args, **kwargs):
        actions = np.array([self.action_space.sample() for _ in range(len(obs_batch))])
        return actions, [], {}

    def compute_single_action(self, obs, *args, **kwargs):
        action = self.action_space.sample()
        return action, [], {}

    def learn_on_batch(self, samples): return {}
    def get_weights(self): return {}
    def set_weights(self, weights): pass

# ==============================================================================
# 3. CREAZIONE DELL'AMBIENTE
# ==============================================================================
def env_creator(args):
    env = EdgeContinuumEnv()
    return ParallelPettingZooEnv(env)

if __name__ == "__main__":
    ray.init()
    env_name = "edge_continuum_auto_env"
    register_env(env_name, env_creator)

    temp_env = env_creator({})
    obs_space = temp_env.observation_space["worker_0"]
    act_space = temp_env.action_space["worker_0"]

    # ==============================================================================
    # 4. IL DIZIONARIO DELLE POLICY (CINQUE cervelli indipendenti! Nessun random.)
    # ==============================================================================
    policies = {
        "ppo_policy_0": (None, obs_space, act_space, {}), 
        "ppo_policy_1": (None, obs_space, act_space, {}), 
        "ppo_policy_2": (None, obs_space, act_space, {}),
        "ppo_policy_3": (None, obs_space, act_space, {}),
        "ppo_policy_4": (None, obs_space, act_space, {}), # <- L'ultimo arrivato!
        "random_policy": (RandomPolicy, obs_space, act_space, {})
    }

    # ==============================================================================
    # 5. IL VIGILE URBANO (Policy Mapping) - FASE 5 (Totalmente Autonomo)
    # ==============================================================================
    def policy_mapping_fn(agent_id, episode, worker, **kwargs):
        if agent_id == "worker_0": return "ppo_policy_0"
        elif agent_id == "worker_1": return "ppo_policy_1"
        elif agent_id == "worker_2": return "ppo_policy_2"
        elif agent_id == "worker_3": return "ppo_policy_3"
        elif agent_id == "worker_4": return "ppo_policy_4" # <- Tutti intelligenti!
        else: return "random_policy" # Fallback di sicurezza

    # ==============================================================================
    # 6. CONFIGURAZIONE DELL'ALGORITMO
    # ==============================================================================
    algo_config = (
        PPOConfig()
        .api_stack(enable_rl_module_and_learner=False, enable_env_runner_and_connector_v2=False)
        .environment(env=env_name)
        .env_runners(num_env_runners=1) 
        .multi_agent(
            policies=policies,
            policy_mapping_fn=policy_mapping_fn,
            # Addestriamo tutti e 5 insieme per l'accordo finale!
            policies_to_train=["ppo_policy_0", "ppo_policy_1", "ppo_policy_2", "ppo_policy_3", "ppo_policy_4"], 
        )
        .training(gamma=0.99, lr=5e-4, train_batch_size=4000, minibatch_size=128)
        .debugging(log_level="ERROR")
    )

    algo = algo_config.build()

    # ==============================================================================
    # LA MAGIA: TRASFERIMENTO DEI PESI dalla Fase 4 all'intero cluster
    # ==============================================================================
    CHECKPOINT_FASE_4 = "/var/folders/gb/b_8vls29731bgmdz63sxbtnw0000gn/T/tmp9znkiffs"
    
    old_policy_path = os.path.join(CHECKPOINT_FASE_4, "policies", "ppo_policy_0") 
    print("\nEstrazione della conoscenza dalla Fase 4 in corso...")
    old_policy = Policy.from_checkpoint(old_policy_path)
    old_weights = old_policy.get_weights()
    
    # Iniettiamo l'esperienza a tutti e cinque
    algo.get_policy("ppo_policy_0").set_weights(old_weights)
    algo.get_policy("ppo_policy_1").set_weights(old_weights)
    algo.get_policy("ppo_policy_2").set_weights(old_weights)
    algo.get_policy("ppo_policy_3").set_weights(old_weights)
    algo.get_policy("ppo_policy_4").set_weights(old_weights)
    
    print("[!] Conoscenza trasferita. Il cluster è 100% Edge-Intelligent!\n")

    # ==============================================================================
    # 7. CICLO DI ADDESTRAMENTO
    # ==============================================================================
    print("="*65)
    print(" INIZIO ADDESTRAMENTO CURRICULUM - FASE 5 (GRAN FINALE)")
    print(" 5 Agenti INDIPENDENTI PPO (100% MARL Autonomo)")
    print("="*65 + "\n")

    num_iterations = 800 
    
    for i in range(num_iterations):
        result = algo.train()
        
        # Monitoriamo le 5 reward
        r0 = result.get('policy_reward_mean', {}).get('ppo_policy_0', 'N/A')
        r1 = result.get('policy_reward_mean', {}).get('ppo_policy_1', 'N/A')
        r2 = result.get('policy_reward_mean', {}).get('ppo_policy_2', 'N/A')
        r3 = result.get('policy_reward_mean', {}).get('ppo_policy_3', 'N/A')
        r4 = result.get('policy_reward_mean', {}).get('ppo_policy_4', 'N/A')
        
        print(f"Iter: {i+1:03d} | W0:{r0} | W1:{r1} | W2:{r2} | W3:{r3} | W4:{r4}")

        if (i + 1) % 20 == 0:
            checkpoint_dir = algo.save()
            print(f" -> Checkpoint FASE 5 salvato in: {checkpoint_dir}")

    print("\nAddestramento Fase 5 completato! IL CURRICULUM È FINITO.")
    ray.shutdown()