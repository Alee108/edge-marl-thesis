import os
import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.tune.registry import register_env
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv

# Importiamo il nostro ambiente e le configurazioni
from env.edge_continuum import EdgeContinuumEnv
import config

# 1. Funzione per creare l'ambiente
def env_creator(args):
    # RLlib richiede che l'ambiente PettingZoo sia "avvolto" (wrapped)
    # per essere compatibile con i suoi standard interni.
    env = EdgeContinuumEnv()
    return ParallelPettingZooEnv(env)

if __name__ == "__main__":
    # Inizializza Ray (il motore di calcolo distribuito)
    ray.init()

    # Registra l'ambiente personalizzato in RLlib
    env_name = "edge_continuum_env"
    register_env(env_name, env_creator)

    # Crea un'istanza temporanea
    temp_env = env_creator({})
    
    # 1. MODIFICA QUI: Estraiamo lo spazio di UN SOLO worker, non di tutti
    obs_space = temp_env.observation_space["worker_0"]
    act_space = temp_env.action_space["worker_0"]

    policies = {
        "shared_policy": (None, obs_space, act_space, {})
    }

    def policy_mapping_fn(agent_id, episode, worker, **kwargs):
        return "shared_policy"

    # 3. Configurazione dell'algoritmo PPO (MAPPO)
    algo_config = (
        PPOConfig()
        # 2. MODIFICA QUI: Disattiviamo il nuovo stack sperimentale di Ray 
        # che causa conflitti con PettingZoo
        .api_stack(
            enable_rl_module_and_learner=False,
            enable_env_runner_and_connector_v2=False,
        )
        .environment(env=env_name)
        .env_runners(num_env_runners=1) 
        .multi_agent(
            policies=policies,
            policy_mapping_fn=policy_mapping_fn,
        )
        .training(
            gamma=0.99, 
            lr=5e-4,    
            train_batch_size=4000, 
            minibatch_size=128,
        )
        .debugging(log_level="ERROR")
    )

    # Costruisci l'algoritmo
    algo = algo_config.build()

    # 4. Ciclo di Addestramento
    print("\n" + "="*50)
    print(f"Inizio Addestramento MAPPO - Alpha: {config.ALPHA}")
    print("="*50 + "\n")

    num_iterations = 400 # Numero di cicli di training
    
    for i in range(num_iterations):
        result = algo.train()
        
        # Estraiamo le metriche principali: la reward media dell'episodio
        # che ci fa capire se stanno imparando a cooperare
        reward = result.get('episode_reward_mean', 'N/A')
        
        print(f"Iterazione: {i+1:02d} | Reward Media Episodio: {reward}")

        # Salviamo il modello ogni 10 iterazioni
        if (i + 1) % 10 == 0:
            checkpoint_dir = algo.save()
            print(f" -> Checkpoint salvato in: {checkpoint_dir}")

    print("\nAddestramento completato!")
    ray.shutdown()