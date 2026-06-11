import torch

assert torch.cuda.is_available(), "需要有 CUDA 才能跑这个 demo"

# 1. 用 dense 矩阵构造 A
A_dense = torch.tensor([[4.0, 1.0],
                        [2.0, 3.0]], device='cuda')

# 2. 转为 CSR 稀疏矩阵（注意是 to_sparse_csr）
A_csr = A_dense.to_sparse_csr()

# 3. 构造 b
b = torch.tensor([1.0, 2.0], device='cuda')

# 4. 调用 spsolve（注意：输入必须在 GPU 上，且 A 是 CSR 格式）
x = torch.sparse.spsolve(A_csr, b)

print("解 x =", x)
print("验证 Ax =", torch.matmul(A_dense, x))
