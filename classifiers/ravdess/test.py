import torch

a = torch.zeros([2850, 1])
print(a.shape)

# want to get it to [2, 2, 1]
b = a.view(50, 1, 1)
print(b)