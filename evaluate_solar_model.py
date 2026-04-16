import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from ray.rllib.policy.policy import Policy

# ==============================================================================
# 1. SETUP E OVERRIDE CONFIG
# ==============================================================================
import config
config.NUM_WORKERS = 8
config.CLOUD_AVAILABLE = False
from env.solar_edge_env import SolarEdgeContinuumEnv

CHECKPOINT_PATH = "/var/folders/gb/b_8vls29731bgmdz63sxbtnw0000gn/T/tmp4vzjtd_t/policies"

# ==============================================================================
# 2. CARICAMENTO DELL'AMBIENTE E DEI CERVELLI
# ==============================================================================
print("Caricamento dell'Ambiente di Valutazione...")
env = SolarEdgeContinuumEnv(training_mode=False)

print("Estrazione della conoscenza dai Checkpoint...")
policies = {}
try:
    for i in range(config.NUM_WORKERS):
        policy_dir = os.path.join(CHECKPOINT_PATH, f"ppo_policy_{i}")
        policies[f"worker_{i}"] = Policy.from_checkpoint(policy_dir)
except Exception as e:
    print(f"\n[ERRORE] Impossibile caricare il checkpoint.\nDettaglio: {e}")
    exit()

# ==============================================================================
# 3. CICLO DI SIMULAZIONE
# ==============================================================================
SIMULATION_STEPS = 336  # 2 settimane

# Strutture dati
history_batt = {f"worker_{i}": [] for i in range(config.NUM_WORKERS)}
history_variance = []
history_queues = {f"worker_{i}": [] for i in range(config.NUM_WORKERS)}
history_solar = {'nord': [], 'est': [], 'sud': [], 'ovest': []}
history_actions = {f"worker_{i}": [] for i in range(config.NUM_WORKERS)}
history_rejected_steps = []
total_rejected = 0

print(f"Avvio simulazione per {SIMULATION_STEPS} step ({SIMULATION_STEPS//24} giorni)...")
obs, _ = env.reset(seed=42)

for step in range(SIMULATION_STEPS):
    actions = {}
    for agent_id, observation in obs.items():
        action, _, _ = policies[agent_id].compute_single_action(observation, explore=False)
        actions[agent_id] = action

    for a in env.possible_agents:
        history_actions[a].append(actions.get(a, 0))

    ws = env.weather_step
    for orient in ['nord', 'est', 'sud', 'ovest']:
        history_solar[orient].append(env.solar_profiles[orient][ws] * env.solar_augmentation)

    next_obs, rewards, terminations, truncations, infos = env.step(actions)

    current_batts = []
    for i, a in enumerate(env.possible_agents):
        norm_b = env.batteries[a] / env.initial_batteries[i]
        history_batt[a].append(norm_b)
        current_batts.append(norm_b)
        history_queues[a].append(env.queues[a])

    history_variance.append(np.var(current_batts))

    rejected = infos['worker_0']['rejected']
    history_rejected_steps.append(1 if rejected else 0)
    if rejected:
        total_rejected += 1

    obs = next_obs
    if all(terminations.values()) or all(truncations.values()):
        print(f"[WARNING] System terminated at step {step}.")
        break

print("Simulation complete! Generating plots...")

# ==============================================================================
# 4. SETUP COMUNE
# ==============================================================================
orient_map = {
    'North': ['worker_0', 'worker_1'],
    'West':  ['worker_2', 'worker_3'],
    'East':  ['worker_4', 'worker_5'],
    'South': ['worker_6', 'worker_7'],
}
orient_colors = {'North': '#4575b4', 'East': '#fdae61', 'South': '#d73027', 'West': '#91bfdb'}
orient_solar_key = {'North': 'nord', 'East': 'est', 'South': 'sud', 'West': 'ovest'}

num_steps = len(history_variance)
x = np.arange(num_steps)

day_ticks = np.arange(0, num_steps, 24)
day_labels = [f'D{d+1}' for d in range(len(day_ticks))]

plt.style.use('seaborn-v0_8-whitegrid')

def add_night_shading(ax, num_steps):
    for day_start in range(0, num_steps, 24):
        night_start = day_start + 20
        night_end = day_start + 24
        if night_start < num_steps:
            ax.axvspan(night_start, min(night_end, num_steps), alpha=0.08, color='navy', zorder=0)
        morning_end = day_start + 6
        if day_start < num_steps:
            ax.axvspan(day_start, min(morning_end, num_steps), alpha=0.08, color='navy', zorder=0)

def setup_xaxis(ax):
    ax.set_xticks(day_ticks)
    ax.set_xticklabels(day_labels)
    ax.set_xlabel('Simulated Days', fontsize=12)

# ==============================================================================
# FIGURE 1: Battery Levels by Orientation
# ==============================================================================
fig1, ax1 = plt.subplots(figsize=(14, 5))
add_night_shading(ax1, num_steps)

for orient_name, workers in orient_map.items():
    color = orient_colors[orient_name]
    avg_batt = np.mean([history_batt[w] for w in workers], axis=0)
    min_batt = np.min([history_batt[w] for w in workers], axis=0)
    max_batt = np.max([history_batt[w] for w in workers], axis=0)
    ax1.plot(x, avg_batt, color=color, linewidth=2.5, label=f'{orient_name} (mean)')
    ax1.fill_between(x, min_batt, max_batt, color=color, alpha=0.15)

ax1.set_ylabel('Normalized Battery Level', fontsize=12)
ax1.set_ylim(0.0, 1.05)
ax1.set_title('Battery Levels by Panel Orientation (pair mean + range)', fontsize=14, fontweight='bold')
ax1.legend(loc='lower left', ncol=4, fontsize=10)
setup_xaxis(ax1)
fig1.tight_layout()
fig1.savefig("plot_batteries.png", dpi=300, bbox_inches='tight')
print("[1/4] plot_batteries.png saved")

# ==============================================================================
# FIGURE 2: Solar Irradiance Profiles
# ==============================================================================
fig2, ax2 = plt.subplots(figsize=(14, 5))
add_night_shading(ax2, num_steps)

for orient_name in ['East', 'South', 'West', 'North']:
    key = orient_solar_key[orient_name]
    color = orient_colors[orient_name]
    ax2.plot(x, history_solar[key], color=color, linewidth=1.5, alpha=0.85, label=orient_name)
    ax2.fill_between(x, history_solar[key], color=color, alpha=0.08)

ax2.set_ylabel('Normalized Irradiance', fontsize=12)
ax2.set_ylim(-0.02, 1.05)
ax2.set_title('Real Solar Irradiance Profiles (PVGIS - Rome 2023)', fontsize=14, fontweight='bold')
ax2.legend(loc='upper right', ncol=4, fontsize=10)
setup_xaxis(ax2)
fig2.tight_layout()
fig2.savefig("plot_solar.png", dpi=300, bbox_inches='tight')
print("[2/4] plot_solar.png saved")

# ==============================================================================
# FIGURE 3: Cluster Energy Variance + Acceptance Rate
# ==============================================================================
fig3, ax3 = plt.subplots(figsize=(14, 5))
add_night_shading(ax3, num_steps)

window = 12
variance_smooth = np.convolve(history_variance, np.ones(window)/window, mode='same')
ax3.fill_between(x, history_variance, color='red', alpha=0.15, label='Variance (instantaneous)')
ax3.plot(x, variance_smooth, color='darkred', linewidth=2.5, label=f'Variance ({window}h moving avg)')
ax3.set_ylabel('Energy Variance', fontsize=12, color='darkred')
ax3.set_ylim(-0.001, max(max(history_variance)*1.3, 0.02))

ax3b = ax3.twinx()
accept_window = 24
for orient_name, workers in orient_map.items():
    color = orient_colors[orient_name]
    accept_rate = np.mean([history_actions[w] for w in workers], axis=0)
    accept_smooth = np.convolve(accept_rate, np.ones(accept_window)/accept_window, mode='same')
    ax3b.plot(x, accept_smooth, color=color, linewidth=1.5, linestyle='--', alpha=0.7)

ax3b.set_ylabel('Acceptance Rate (24h avg)', fontsize=12, color='gray')
ax3b.set_ylim(-0.05, 1.05)

lines1, labels1 = ax3.get_legend_handles_labels()
orient_patches = [mpatches.Patch(color=orient_colors[o], alpha=0.5, label=f'Accept. {o}') for o in orient_map]
ax3.legend(handles=lines1 + orient_patches, loc='upper left', ncol=3, fontsize=9)
ax3.set_title('Cluster Energy Variance and Task Acceptance Rate by Orientation', fontsize=14, fontweight='bold')
setup_xaxis(ax3)
fig3.tight_layout()
fig3.savefig("plot_variance_acceptance.png", dpi=300, bbox_inches='tight')
print("[3/4] plot_variance_acceptance.png saved")

# ==============================================================================
# FIGURE 4: Queue Lengths + Rejected Tasks
# ==============================================================================
fig4, ax4 = plt.subplots(figsize=(14, 5))
add_night_shading(ax4, num_steps)

for orient_name, workers in orient_map.items():
    color = orient_colors[orient_name]
    avg_q = np.mean([history_queues[w] for w in workers], axis=0)
    ax4.plot(x, avg_q, color=color, linewidth=2, label=f'{orient_name}')

rejected_steps = [i for i, r in enumerate(history_rejected_steps) if r == 1]
if rejected_steps:
    ax4.scatter(rejected_steps, [0.1]*len(rejected_steps), color='red', marker='x',
                s=60, zorder=5, label=f'Rejected tasks ({total_rejected})')

ax4.axhline(y=config.MAX_QUEUE, color='red', linestyle='--', alpha=0.5, linewidth=1, label='Max queue capacity')
ax4.set_ylabel('Average Queue Length', fontsize=12)
ax4.set_ylim(-0.1, config.MAX_QUEUE + 0.5)
ax4.set_title(f'Queue Management by Orientation (total rejected: {total_rejected}/{num_steps})',
              fontsize=14, fontweight='bold')
ax4.legend(loc='upper right', ncol=3, fontsize=10)
setup_xaxis(ax4)
fig4.tight_layout()
fig4.savefig("plot_queues.png", dpi=300, bbox_inches='tight')
print("[4/4] plot_queues.png saved")

# ==============================================================================
# SUMMARY
# ==============================================================================
print(f"\n{'='*50}")
print(f"  Rejected tasks:    {total_rejected}/{num_steps}")
print(f"  Mean variance:     {np.mean(history_variance):.6f}")
print(f"  Min battery (end): {min(history_batt[a][-1] for a in env.possible_agents):.3f}")
print(f"  Max battery (end): {max(history_batt[a][-1] for a in env.possible_agents):.3f}")
print(f"{'='*50}")
plt.show()
