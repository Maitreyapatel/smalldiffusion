import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from smalldiffusion import (
    TimeInputMLP, ScheduleLogLinear, training_loop, samples,
    CombinedDistribution
)
from smalldiffusion.model import get_sigma_embeds
import seaborn as sns

def plot_batch(batch, name="batch"):
    # Convert tensor to numpy
    batch = batch.cpu().numpy()
    
    # Set Seaborn style for better aesthetics
    sns.set_style("whitegrid")
    plt.figure(figsize=(10, 8))
    
    # Create scatter plot with Seaborn
    sns.scatterplot(
        x=batch[:, 0], 
        y=batch[:, 1],
        size=10,           # Point size
        alpha=0.6,         # Transparency
        color='#1f77b4',   # Default Seaborn color
        legend=False       # Remove legend for cleaner look
    )
    
    # Customize the plot
    plt.title("Distribution of Combined Clusters", fontsize=16, pad=20)
    plt.xlabel("X Coordinate", fontsize=12)
    plt.ylabel("Y Coordinate", fontsize=12)
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    # Save with higher DPI for better quality
    plt.savefig(f'./logs/{name}.png', dpi=300, bbox_inches='tight')
    plt.close()

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

dataset = CombinedDistribution(
    [(0, 0), (1, 0), (0, 1), (3, 3), (3, 2), (2, 3)], 
    std=0.1, 
    points_per_center=1000
)

loader = DataLoader(dataset, batch_size=10000, shuffle=True)
plot_batch(next(iter(loader)))

schedule = ScheduleLogLinear(N=200, sigma_min=0.01, sigma_max=10)
sx, sy = get_sigma_embeds(len(schedule), schedule.sigmas).T

model = TimeInputMLP(hidden_dims=(16,128,128,128,128,16))

trainer = training_loop(loader, model, schedule, epochs=5000, lr=1e-3)
losses = [ns.loss.item() for ns in trainer]

plt.plot(moving_average(losses, 100))
plt.savefig('./logs/loss.png', dpi=300, bbox_inches='tight')
plt.close()

*xts, x0 = samples(model, schedule.sample_sigmas(20), batchsize=5000, gam=2, mu=0)
plot_batch(x0, name="generated")