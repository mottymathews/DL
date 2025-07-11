import torch
import time

# Select device
cpu = torch.device("cpu")
mps = torch.device("mps") if torch.backends.mps.is_available() else None


# Check memory before
print(f"Memory allocated before: {torch.mps.current_allocated_memory() / 1024**2:.2f} MB")

x = torch.rand(2**20, requires_grad=True, device=mps)

print(f"Memory allocated after: {torch.mps.current_allocated_memory() / 1024**2:.2f} MB")

b = torch.relu(x)
print(f"Memory allocated after: {torch.mps.current_allocated_memory() / 1024**2:.2f} MB")

b = x
for _ in range(100):
    b = torch.relu(b)
print(f"Memory allocated after: {torch.mps.current_allocated_memory() / 1024**2:.2f} MB")

b.sum().backward()
print(f"Memory allocated after: {torch.mps.current_allocated_memory() / 1024**2:.2f} MB")

