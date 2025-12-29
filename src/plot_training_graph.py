import os
import pickle
import matplotlib.pyplot as plt

# Get project root and path to saved losses
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
loss_file = os.path.join(BASE_DIR, "outputs", "training_losses.pkl")

# Load loss history
with open(loss_file, "rb") as f:
    data = pickle.load(f)

D_losses = data["D_losses"]
G_losses = data["G_losses"]

# Plot
plt.figure(figsize=(10, 6))
plt.plot(D_losses, label="Discriminator Loss")
plt.plot(G_losses, label="Generator Loss")
plt.title("DCGAN Training Loss Curve")
plt.xlabel("Iterations (batches)")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)

# Save and show
plot_path = os.path.join(BASE_DIR, "outputs", "loss_curve_plot.png")
plt.savefig(plot_path)
plt.show()

print("Saved training graph at:", plot_path)
