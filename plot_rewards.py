import os
import glob
import pandas as pd
import matplotlib.pyplot as plt

# Se la tua cartella ray_results è altrove, modifica questo path
RAY_RESULTS_DIR = os.path.expanduser("~/ray_results")

def smooth_curve(points, factor=0.85):
    """Applica uno smoothing per rendere la curva più leggibile"""
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous * factor + point * (1 - factor))
        else:
            smoothed_points.append(point)
    return smoothed_points

def plot_training_subplots():
    # Cerca tutti i file progress.csv
    csv_files = glob.glob(os.path.join(RAY_RESULTS_DIR, "**", "progress.csv"), recursive=True)
    
    if not csv_files:
        print("Nessun file progress.csv trovato! Controlla il percorso di ray_results.")
        return

    # Ordina i file per data di creazione e prendi gli ultimi 3
    csv_files.sort(key=os.path.getmtime)
    latest_csvs = csv_files[-3:] 
    
    # Etichette in ordine (assicurati che l'ordine temporale in cui li hai lanciati sia questo!)
    # Se li hai lanciati in un ordine diverso (es. 0.0, 1.0, 0.5), cambiale di conseguenza.
    labels_pool = ["Alpha = 0.0", "Alpha = 1.0", "Alpha = 0.5"]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c'] 

    fig, axs = plt.subplots(len(latest_csvs), 1, figsize=(10, 3 * len(latest_csvs)), sharex=True)
    if len(latest_csvs) == 1: axs = [axs]

    for idx, csv_file in enumerate(latest_csvs):
        try:
            df = pd.read_csv(csv_file)
            
            iter_col = 'training_iteration' if 'training_iteration' in df.columns else df.columns[0]
            reward_col = next((col for col in df.columns if 'episode_reward_mean' in col), None)
            
            if not reward_col: continue
            
            iterations = df[iter_col].values
            mean_rewards = df[reward_col].values
            smoothed_rewards = smooth_curve(mean_rewards)
            
            label = labels_pool[idx] if idx < len(labels_pool) else f"Training {idx}"
            color = colors[idx] if idx < len(colors) else 'black'
            
            axs[idx].plot(iterations, mean_rewards, color=color, alpha=0.3, linewidth=1)
            axs[idx].plot(iterations, smoothed_rewards, color=color, linewidth=2.5, label=label)
            
            axs[idx].set_title(label, fontsize=12, fontweight='bold')
            axs[idx].set_ylabel('Reward', fontsize=10)
            axs[idx].grid(True, linestyle='--', alpha=0.6)
            
        except Exception as e:
            print(f"Errore in {csv_file}: {e}")

    axs[-1].set_xlabel('Iterazioni di Addestramento', fontsize=12)
    fig.suptitle('Curve di Convergenza MAPPO', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()

    plt.savefig('learning_curves_final.png', dpi=300, bbox_inches='tight')
    print("Grafico salvato come 'learning_curves_final.png'!")
    plt.show()

if __name__ == "__main__":
    plot_training_subplots()