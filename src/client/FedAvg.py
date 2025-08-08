import torch
from torch.utils.data import DataLoader
from copy import deepcopy

class FedAvgClient:
    """
    最小客户端：
    - 接收全局参数
    - 本地训练 local_epochs
    - 回传本地模型参数 + 样本数
    """
    def __init__(self, cid, model, dataset, *, lr=0.05, batch_size=64, device="cpu"):
        self.cid = cid
        self.device = torch.device(device)
        # 用一个本地副本，避免直接改全局模型
        self.model = deepcopy(model).to(self.device)
        self.dataset = dataset
        self.batch_size = batch_size
        self.lr = lr

    # —— 广播时调用 —— #
    def set_global_weights(self, state_dict):
        self.model.load_state_dict(state_dict, strict=True)

    # —— 聚合时使用 —— #
    def get_state(self):
        # 参数+buffers 都要带上（BN等）
        return {k: v.detach().cpu() for k, v in self.model.state_dict().items()}

    def num_samples(self):
        return len(self.dataset)

    def _loader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True, drop_last=False)

    # —— 本地训练 —— #
    def train_one_round(self, *, local_epochs=1):
        self.model.train()
        opt = torch.optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.9)
        loss_fn = torch.nn.CrossEntropyLoss()
        loader = self._loader()

        for _ in range(local_epochs):
            for x, y in loader:
                x, y = x.to(self.device), y.to(self.device)
                opt.zero_grad()
                logits = self.model(x)
                loss = loss_fn(logits, y)
                loss.backward()
                opt.step()

        return {
            "cid": self.cid,
            "num_samples": self.num_samples(),
            "state": self.get_state(),
        }
