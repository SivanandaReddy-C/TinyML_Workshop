import torch

print("PyTorch Version :", torch.__version__)
print("CUDA Available :", torch.cuda.is_available())

a = torch.tensor([1,2,3])

print(a)

A=torch.tensor([[1.,2.],[3.,4.]])

B=torch.tensor([[5.,6.],[7.,8.]])

print(torch.matmul(A,B))