import random
import torch

class FedAvgServer:
    """
    最小服务端：
    - 选择客户端
    - 下发全局参数
    - 接收本地更新并按样本数加权聚合
    """
    def __init__(self, global_model, clients, *, device="cpu"):
        self.device = torch.device(device)
        self.global_model = global_model.to(self.device)
        self.clients = clients

    def _global_state(self):
        return {k: v.detach().cpu() for k, v in self.global_model.state_dict().items()}

    def select_clients(self, fraction=1.0):
        m = max(1, int(fraction * len(self.clients)))
        return random.sample(self.clients, m)

    def broadcast(self, selected_clients):
        g_state = self._global_state()
        for c in selected_clients:
            c.set_global_weights(g_state)

    def aggregate(self, payloads):
        total = sum(p["num_samples"] for p in payloads)
        keys = payloads[0]["state"].keys()
        new_state = {}
        for k in keys:
            acc = None
            for p in payloads:
                w = p["num_samples"] / total
                tensor = p["state"][k]
                acc = tensor * w if acc is None else acc + tensor * w
            new_state[k] = acc
        self.global_model.load_state_dict(new_state, strict=True)

    def run_round(self, *, fraction=1.0, local_epochs=1):
        selected = self.select_clients(fraction)
        self.broadcast(selected)
        payloads = [c.train_one_round(local_epochs=local_epochs) for c in selected]
        self.aggregate(payloads)
        return {
            "selected": [p["cid"] for p in payloads],
            "total_samples": sum(p["num_samples"] for p in payloads),
        }
