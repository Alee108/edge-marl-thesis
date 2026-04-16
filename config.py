# config.py
import os

# ==========================================
# Parametri dell'Ambiente Edge-to-Cloud
# Versione 2.0: Energy Harvesting & 8 Nodi
# ==========================================

NUM_WORKERS = 8  # Scalabilità aumentata per la tesi
MAX_QUEUE = 5    # Capacità massima della coda (K) per ogni worker

# Parametri fisici dei dispositivi (Eterogeneità Hardware)
# P_completion: Probabilità di completare 1 task per step (più alto = più veloce)
# 4 coppie di nodi con prestazioni diverse
WORKER_SPEEDS = [
    0.95, 0.95,  # Nodi molto potenti (es. Jetson Orin)
    0.80, 0.80,  # Nodi medi (es. Jetson Nano)
    0.65, 0.65,  # Nodi standard (es. Raspberry Pi 4)
    0.50, 0.50   # Nodi a basso consumo (es. ESP32/Micro-controller)
]

# B_i: Capacità massima della batteria in Wh
# Nota: Questi valori ora rappresentano il tetto massimo (100%)
WORKER_BATTERIES = [10.0, 10.0, 8.0, 8.0, 7.0, 7.0, 5.0, 5.0]

# ==========================================
# Impostazioni della Simulazione
# ==========================================
CLOUD_AVAILABLE = False # Disattivato per forzare la cooperazione tra nodi Edge
MAX_TIMESTEPS = 10000   # Durata estesa per vedere più cicli giorno/notte

# ==========================================
# Parametri Reinforcement Learning
# ==========================================
# ALPHA = 0.5 bilancia equamente prestazioni e sopravvivenza energetica
ALPHA = 0.5