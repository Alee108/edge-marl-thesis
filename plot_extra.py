import matplotlib.pyplot as plt
import seaborn as sns

# Impostazioni di stile eleganti per la tesi
sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.size': 12, 'font.family': 'sans-serif'})

# Dati estratti dalle 5 Fasi
fasi = ['Phase 1\n(1S, 4R)', 'Phase 2\n(2S, 3R)', 'Phase 3\n(3S, 2R)', 'Phase 4\n(4S, 1R)', 'Phase 5\n(5S, 0R)']

# Nuovi dati estratti dalle tue tabelle
delta = [0.18, 0.29, 0.22, 0.13, 0.07]
gamma_0 = [98.7, 98.7, 98.7, 99.5, 99.6]  # Task leggeri (Tipo 0)
gamma_1 = [100.0, 98.2, 99.1, 99.6, 99.5] # Task medi (Tipo 1)
gamma_2 = [96.5, 96.6, 98.7, 99.6, 99.2]  # Task pesanti (Tipo 2)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# --- 1. Grafico del Delta (δ) ---
sns.lineplot(x=fasi, y=delta, marker="D", markersize=10, linewidth=3, color="#E67E22", ax=ax1)
ax1.set_title("Maximum Battery Gap (Delta δ)", fontweight='bold', pad=15)
ax1.set_ylabel("Delta (Max - Min Battery %)")
ax1.set_xlabel("Curriculum Learning Phases")

# Freccia per evidenziare il picco di inefficienza nella Fase 2
ax1.annotate('Highest Imbalance\n(Agents competing)', 
             xy=(1, 0.29), xytext=(1.5, 0.27),
             arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
             fontsize=10)

# --- 2. Grafico del Throughput per Tipo di Task ---
sns.lineplot(x=fasi, y=gamma_0, marker="o", markersize=8, linewidth=2.5, label="Type 0 (Light)", color="#2ECC71", ax=ax2)
sns.lineplot(x=fasi, y=gamma_1, marker="s", markersize=8, linewidth=2.5, label="Type 1 (Medium)", color="#F1C40F", ax=ax2)
sns.lineplot(x=fasi, y=gamma_2, marker="^", markersize=8, linewidth=2.5, label="Type 2 (Heavy)", color="#E74C3C", ax=ax2)

ax2.set_title("Throughput Breakdown by Task Type", fontweight='bold', pad=15)
ax2.set_ylabel("Throughput (%)")
ax2.set_xlabel("Curriculum Learning Phases")
ax2.legend(loc="lower right", frameon=True, shadow=True)
ax2.set_ylim(95.0, 100.5)

plt.tight_layout()

# Salva l'immagine in alta qualità per la tesi
plt.savefig("curriculum_learning_extra_en.png", dpi=300, bbox_inches='tight')
print("Grafici extra salvati con successo come 'curriculum_learning_extra_en.png'!")
plt.show()