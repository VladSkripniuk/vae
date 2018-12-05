import numpy as np

from sklearn.datasets import make_moons, make_blobs

from torch.utils.data import Dataset, DataLoader



class Blobs(Dataset):
    def __init__(self, n_samples=1000, std=1.0, centers=None):
        
        self.x, self.y = make_blobs(n_samples=n_samples, centers=centers, shuffle=True, cluster_std=std, random_state=42)
        self.x = np.asarray(self.x, dtype=np.float32)
        
    
    def __len__(self):
        
        return len(self.x)
    
    
    def __getitem__(self, idx):
        
        return self.x[idx]
        

class Bananas(Dataset):
    def __init__(self, n_samples=1000, std=0.1):
        
        self.x, self.y = make_moons(n_samples=n_samples, shuffle=True, noise=std, random_state=42)
        self.x = np.asarray(self.x, dtype=np.float32)
        
    
    def __len__(self):
        
        return len(self.x)
    
    
    def __getitem__(self, idx):
        
        return self.x[idx]
        