import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Impostazioni di stile eleganti per la tesi
sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.size': 12, 'font.family': 'sans-serif'})

# Dati estratti dalle 5 Fasi
fasi = ['Phase 1\n(1S, 4R)', 'Phase 2\n(2S, 3R)', 'Phase 3\n(3S, 2R)', 'Phase 4\n(4S, 1R)', 'Phase 5\n(5S, 0R)']
varianza = [0.0048, 0.0112, 0.0062, 0.0028, 0.0007]
vita_m = [683, 650, 680, 695, 715]
throughput = [97.7, 97.8, 98.8, 99.6, 99.4]

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))

# 1. Grafico della Varianza (σ)
sns.lineplot(x=fasi, y=varianza, marker="o", markersize=10, linewidth=3, color="#D9534F", ax=ax1)
ax1.set_title("Energy Imbalance Evolution (Variance σ)", fontweight='bold', pad=15)
ax1.set_ylabel("Variance (σ)")
ax1.set_xlabel("Curriculum Learning Phases")
# Evidenziamo il Tipping point (Attrito da transizione)
ax1.axvspan(0.8, 1.2, color='gray', alpha=0.1) 
ax1.text(1.2, 0.010, 'Transition Friction\n(Two Heroes Syndrome)', fontsize=10, color='#333333')

# 2. Grafico della Vita del Sistema (m)
sns.lineplot(x=fasi, y=vita_m, marker="s", markersize=10, linewidth=3, color="#5CB85C", ax=ax2)
ax2.set_title("Cluster Survival Time (m)", fontweight='bold', pad=15)
ax2.set_ylabel("Simulation Steps (m)")
ax2.set_xlabel("Curriculum Learning Phases")

# 3. Grafico del Throughput (γ)
sns.lineplot(x=fasi, y=throughput, marker="^", markersize=10, linewidth=3, color="#0275D8", ax=ax3)
ax3.set_title("Global Throughput Evolution (γ)", fontweight='bold', pad=15)
ax3.set_ylabel("Throughput (%)")
ax3.set_xlabel("Curriculum Learning Phases")
ax3.set_ylim(97.0, 100.0)

plt.tight_layout()

# Salva l'immagine in alta qualità per la tesi
plt.savefig("curriculum_learning_plots_en.png", dpi=300, bbox_inches='tight')
print("Grafici salvati con successo in formato PNG alta risoluzione!")
plt.show()