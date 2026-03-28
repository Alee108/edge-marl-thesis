# config.py
import os
# ==========================================
# Parametri dell'Ambiente Edge-to-Cloud
# Basati sul paper di riferimento
# ==========================================

NUM_WORKERS = 3
MAX_QUEUE = 5  # Capacità massima della coda (K) per ogni worker

# Parametri fisici dei dispositivi (Scenario con Cloud da Fig. 5)
# S_i: Moltiplicatore di velocità del worker (1.0 è il più veloce)
WORKER_SPEEDS = [1.8, 1.7, 1.4, 1.6, 1.5]

# B_i: Capacità iniziale della batteria in Watt-ora (Wh)
WORKER_BATTERIES = [9.0, 8.0, 7.0, 8.5, 7.5]

# ==========================================
# Impostazioni della Simulazione
# ==========================================
CLOUD_AVAILABLE = True
MAX_TIMESTEPS = 10000  # Durata della simulazione (s) presa dai grafici del paper

# ==========================================
# Parametri Reinforcement Learning
# ==========================================
# ALPHA bilancia l'importanza tra:
# 1.0 = Solo ottimizzazione FPS (prestazioni)
# 0.0 = Solo bilanciamento batterie (risparmio energetico)
ALPHA = 0.5