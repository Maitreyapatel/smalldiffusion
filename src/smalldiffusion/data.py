import torch
import csv
from torch.utils.data import Dataset
from torchvision import transforms as tf

class Swissroll(Dataset):
    def __init__(self, tmin, tmax, N, center=(0,0), scale=1.0):
        t = tmin + torch.linspace(0, 1, N) * tmax
        center = torch.tensor(center).unsqueeze(0)
        self.vals = center + scale * torch.stack([t*torch.cos(t)/tmax, t*torch.sin(t)/tmax]).T

    def __len__(self):
        return len(self.vals)

    def __getitem__(self, i):
        return self.vals[i]

class DatasaurusDozen(Dataset):
    def __init__(self, csv_file, dataset, enlarge_factor=15, delimiter='\t', scale=50, offset=50):
        self.enlarge_factor = enlarge_factor
        self.points = []
        with open(csv_file, newline='') as f:
            for name, *rest in csv.reader(f, delimiter=delimiter):
                if name == dataset:
                    point = torch.tensor(list(map(float, rest)))
                    self.points.append((point - offset) / scale)

    def __len__(self):
        return len(self.points) * self.enlarge_factor

    def __getitem__(self, i):
        return self.points[i % len(self.points)]

# Mainly used to discard labels and only output data
class MappedDataset(Dataset):
    def __init__(self, dataset, fn):
        self.dataset = dataset
        self.fn = fn
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, i):
        return self.fn(self.dataset[i])

class GaussianDistribution(Dataset):
    def __init__(self, center, std=0.1, num_points=1000):
        """
        Create a 2D Gaussian distribution centered at (x,y) with specified standard deviation.
        
        Args:
            center: Tuple (x,y) representing the center of the distribution
            std: Standard deviation of the distribution (controls spread)
            num_points: Number of points to generate
        """
        self.center = torch.tensor(center, dtype=torch.float32)
        self.std = std
        self.num_points = num_points
        
        # Pre-generate all points to ensure deterministic behavior
        self.points = self.std * torch.randn(num_points, 2) + self.center
        
    def __len__(self):
        return self.num_points
        
    def __getitem__(self, idx):
        return self.points[idx]

class CombinedDistribution(Dataset):
    def __init__(self, centers, std=0.1, points_per_center=1000):
        """
        Create a combined distribution from multiple non-overlapping Gaussian distributions.
        
        Args:
            centers: List of (x,y) tuples representing centers of each Gaussian
            std: Standard deviation for all Gaussians
            points_per_center: Number of points to generate per center
        """
        # Create a Gaussian distribution for each center
        self.distributions = [
            GaussianDistribution(center, std, points_per_center) 
            for center in centers
        ]
        
        # Calculate cumulative sizes for indexing
        self.cum_sizes = [0]
        total = 0
        for d in self.distributions:
            total += len(d)
            self.cum_sizes.append(total)
    
    def __len__(self):
        return self.cum_sizes[-1]
    
    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self):
            raise IndexError("Index out of range")
        
        # Find which distribution this index belongs to
        dist_idx = 0
        while dist_idx < len(self.distributions) and idx >= self.cum_sizes[dist_idx + 1]:
            dist_idx += 1
        
        # Calculate local index within that distribution
        local_idx = idx - self.cum_sizes[dist_idx]
        
        # Return the corresponding point
        return self.distributions[dist_idx][local_idx]

img_train_transform = tf.Compose([
    tf.RandomHorizontalFlip(),
    tf.ToTensor(),
    tf.Lambda(lambda t: (t * 2) - 1)
])

img_normalize = lambda x: ((x + 1)/2).clamp(0, 1)