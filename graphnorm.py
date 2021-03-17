import torch
import torch.nn as nn

class GraphNorm(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(latent_dim))
        self.bias = nn.Parameter(torch.zeros(latent_dim))
        self.mean_scale = nn.Parameter(torch.ones(latent_dim))

    def forward(self, x, batch):
        batch_size = batch[-1]+1
        batch_list = [0] * batch_size
        for i in range(len(batch)):
            batch_list[batch[i]] += 1
        batch_list = torch.Tensor(batch_list).long().to(x.device)

        batch_index = batch.view((-1,) + (1,) * (x.dim() - 1)).expand_as(x)

        mean = torch.zeros(batch_size, *x.shape[1:]).to(x.device)
        mean = mean.scatter_add(0, batch_index, x)
        mean = (mean.T / batch_list).T
        mean = mean.repeat_interleave(batch_list, dim=0)

        sub = x - mean * self.mean_scale

        std = torch.zeros(batch_size, *x.shape[1:]).to(x.device)
        std = std.scatter_add_(0, batch_index, sub.pow(2))
        std = ((std.T / batch_list).T + 1e-6).sqrt()
        std = std.repeat_interleave(batch_list, dim=0)
        return self.weight * sub / std + self.bias