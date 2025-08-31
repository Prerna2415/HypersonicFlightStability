import pandas as pd
import matplotlib.pyplot as plt

# change this to whichever episode you want to plot
episode_num = 1
df = pd.read_csv(f"flight_logs_episode_{episode_num}.csv")

fig, axs = plt.subplots(4, 1, figsize=(10, 12), sharex=True)

axs[0].plot(df["step"], df["altitude_ft"])
axs[0].set_ylabel("Altitude (ft)")

axs[1].plot(df["step"], df["mach"])
axs[1].set_ylabel("Mach")

axs[2].plot(df["step"], df["roll_deg"])
axs[2].set_ylabel("Roll (deg)")

axs[3].plot(df["step"], df["reward"])
axs[3].set_ylabel("Reward")
axs[3].set_xlabel("Step")

plt.suptitle(f"Flight Log Episode {episode_num}")
plt.tight_layout()
plt.show()
