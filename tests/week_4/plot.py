import os

import matplotlib.pyplot as plt
import pandas as pd

# CSV einlesen (Header überspringen, Kommentarzeile ignorieren)
df = pd.read_csv(
    "/Users/tizianohumpert/Documents/Studium_local/RL/Repo/rl-week-4-rl_ma/logs/hydra/2025-05-15/14-56-50/train.monitor.csv",
    comment="#",
)

# Gleitenden Mittelwert berechnen (z.B. über 10 Episoden)
window = 10
df["mean_reward"] = df["r"].rolling(window, min_periods=1).mean()

# Ordner für Plots anlegen
os.makedirs("plots", exist_ok=True)

# Plot erstellen
plt.figure(figsize=(10, 5))
plt.plot(df.index, df["mean_reward"], label=f"Mean Reward (window={window})")
plt.xlabel("Episode")
plt.ylabel("Mean Reward")
plt.title("Training Curve: DQN (CartPole, Beispielarchitektur)")
plt.legend()
plt.grid(True)
plt.tight_layout()

# Plot speichern
plt.savefig("plots/training_curve.png")
plt.show()
