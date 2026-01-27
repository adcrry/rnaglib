import torch

def rbf_expand(dists, num_bins: int = 64, min_distance=2.0, max_distance=22.0, gamma=None):    
    # Calculate centers
    # First bin
    centers = torch.zeros(num_bins)
    centers[0] = 1.0
    # Middle bins
    width = (max_distance-min_distance) / (num_bins-2)

    for i in range(1, num_bins-1):
        centers[i] = min_distance + width * (i - 0.5)

    # Last bin
    centers[-1] = max_distance
    centers = centers.view(1, -1)

    if gamma is None:
        gamma = 1.0 / (width ** 2)

    if dists.dim() == 1:
        dists = dists.unsqueeze(-1) # [E, 1]

    # Calculate standard Gaussian for all: [E, 64]
    diff = dists - centers
    rbf = torch.exp(-gamma * diff.pow(2))

    # Apply saturation to the tail
    # if dist > max_distance, force value to 1.0
    rbf[:, -1] = torch.where(dists.squeeze() > max_distance, 
                                torch.ones_like(rbf[:, -1]), 
                                rbf[:, -1])

    return rbf