import os
import glob

paths = glob.glob('/var/folders/**/rllib_checkpoint.json', recursive=True)
paths.sort(key=os.path.getmtime, reverse=True)

print("\nUltimi 3 checkpoint salvati (dal più recente):")
print("="*60)
for i, p in enumerate(paths[:3]):
    print(f"{i+1}. -> {os.path.dirname(p)}")
print("="*60)