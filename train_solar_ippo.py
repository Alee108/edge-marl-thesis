import os
import numpy as np
import ray
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_env
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from ray.rllib.policy.policy import Policy

# ==============================================================================
# 1. OVERRIDE DINAMICO DEL CONFIG (IL NUOVO SETUP SOLARE)
# ==============================================================================
import config
config.ALPHA = 0.5
config.NUM_WORKERS = 8  # <-- SCALABILITÀ: Passiamo a 8 Nodi! (2 Sud, 2 Est, 2 Ovest, 2 Nord)
config.CLOUD_AVAILABLE = False # Costringiamoli a collaborare senza scappatoie

from env.solar_edge_env import SolarEdgeContinuumEnv # <-- Importiamo il nuovo ambiente

# ==============================================================================
# 2. DEFINIZIONE DELLA RANDOM POLICY (Per eventuali test futuri)
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
    env = SolarEdgeContinuumEnv()
    return ParallelPettingZooEnv(env)

if __name__ == "__main__":
    ray.init(ignore_reinit_error=True)
    env_name = "solar_edge_continuum_env"
    register_env(env_name, env_creator)

    temp_env = env_creator({})
    obs_space = temp_env.observation_space["worker_0"]
    act_space = temp_env.action_space["worker_0"]

    # ==============================================================================
    # 4. IL DIZIONARIO DELLE POLICY (OTTO cervelli indipendenti!)
    # ==============================================================================
    policies = {
        f"ppo_policy_{i}": (None, obs_space, act_space, {}) for i in range(config.NUM_WORKERS)
    }
    policies["random_policy"] = (RandomPolicy, obs_space, act_space, {})

    # ==============================================================================
    # 5. IL VIGILE URBANO (Policy Mapping)
    # ==============================================================================
    def policy_mapping_fn(agent_id, episode, worker, **kwargs):
        if agent_id.startswith("worker_"):
            idx = agent_id.split("_")[1]
            return f"ppo_policy_{idx}"
        return "random_policy"

    # ==============================================================================
    # 6. CONFIGURAZIONE DELL'ALGORITMO PPO
    # ==============================================================================
    algo_config = (
        PPOConfig()
        .api_stack(enable_rl_module_and_learner=False, enable_env_runner_and_connector_v2=False)
        .environment(env=env_name)
        .env_runners(num_env_runners=1) 
        .multi_agent(
            policies=policies,
            policy_mapping_fn=policy_mapping_fn,
            # Addestriamo tutti e 8 simultaneamente per far emergere l'Onda Solare
            policies_to_train=[f"ppo_policy_{i}" for i in range(config.NUM_WORKERS)], 
        )
        # Aumentiamo leggermente il batch size perché 8 agenti generano più dati
        .training(gamma=0.99, lr=5e-4, train_batch_size=8000, minibatch_size=256)
        .debugging(log_level="ERROR")
    )

    algo = algo_config.build()

    # ==============================================================================
    # 7. CICLO DI ADDESTRAMENTO
    # ==============================================================================
    print("="*75)
    print(" INIZIO ADDESTRAMENTO SOLARE - L'ONDA SOLARE (8 NODI)")
    print(" Gli agenti impareranno a inseguire il sole (Sud, Est, Ovest, Nord)")
    print("="*75 + "\n")

    num_iterations = 1000 # Potrebbe servire qualche iterazione in più per l'Onda Solare
    
    for i in range(num_iterations):
        result = algo.train()
        
        # Stampiamo un riassunto compatto delle reward
        rewards = []
        for w in range(config.NUM_WORKERS):
            r = result.get('env_runners', {}).get('policy_reward_mean', {}).get(f'ppo_policy_{w}', 0.0)
            # Formattazione per avere numeri allineati e leggibili
            rewards.append(f"W{w}:{r:6.2f}")
            
        print(f"Iter: {i+1:04d} | " + " | ".join(rewards))

        # Salviamo il cervello ogni 50 iterazioni
        if (i + 1) % 50 == 0:
            checkpoint_dir = algo.save()
            print(f"\n[💾] Checkpoint Solare salvato in: {checkpoint_dir}\n")

    print("\nAddestramento Solare completato! Il cluster è ora Zero-Energy Edge-Intelligent.")
    ray.shutdown()