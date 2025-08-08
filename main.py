# main.py
import random, os
import numpy as np
import torch
from torch.utils.data import Subset, DataLoader, TensorDataset
from torchvision import datasets, transforms

from src.utils.models import TwoNN
from src.client.FedAvg import FedAvgClient
from src.server.FedAvg import FedAvgServer


# ----------------- 小工具 -----------------
def set_seed(seed: int = 0):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

@torch.no_grad()
def evaluate(model: torch.nn.Module, device="cpu") -> float:
    model.eval()
    tfm = transforms.ToTensor()
    test = datasets.MNIST("./data", train=False, download=True, transform=tfm)
    loader = DataLoader(test, batch_size=256, shuffle=False, num_workers=0)
    correct = total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += y.numel()
    return correct / total


# ----------------- 构建客户端 -----------------
def make_clients_mnist(K=5, device="cpu", lr=0.05, batch_size=64):
    tfm = transforms.ToTensor()
    train = datasets.MNIST(root="./data", train=True, download=True, transform=tfm)

    n = len(train)
    shard = n // K
    subsets = [Subset(train, list(range(i*shard, (i+1)*shard))) for i in range(K-1)]
    subsets.append(Subset(train, list(range((K-1)*shard, n))))

    global_model = TwoNN(in_dim=28*28, hidden=200, num_classes=10)
    clients = [
        FedAvgClient(i, global_model, subsets[i], lr=lr, batch_size=batch_size, device=device)
        for i in range(K)
    ]
    server = FedAvgServer(global_model, clients, device=device)
    return server

# ----------------- 训练主循环 -----------------
def main():
    set_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    # 训练参数
    K = 20               # 客户端数
    R = 50               # 联邦轮数
    fraction = 0.2      # 每轮采样比例
    local_epochs = 2    # 客户端本地训练 epoch
    lr = 0.05
    batch_size = 64

    # 选一个数据构造器（MNIST / 合成）
    server = make_clients_mnist(K=K, device=device, lr=lr, batch_size=batch_size)
    # server = make_clients_synth(K=K, device=device, lr=lr, batch_size=batch_size)

    for r in range(R):
        stats = server.run_round(fraction=fraction, local_epochs=local_epochs)
        acc = evaluate(server.global_model, device=device)
        print(f"[Round {r}] {stats}  acc={acc:.4f}")

if __name__ == "__main__":
    main()
