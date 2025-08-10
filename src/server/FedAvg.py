# src/server/FedAvg.py
import random
import torch
from typing import Dict, List

class FedAvgServer:
    def __init__(self, global_model: torch.nn.Module, clients: List, *, device):
        self.device = torch.device(device)
        self.global_model = global_model.to(self.device)
        self.clients = clients

    def _global_state(self) -> Dict[str, torch.Tensor]:
        return {k: v.detach().cpu() for k, v in self.global_model.state_dict().items()}

    def select_clients(self, fraction: float = 1.0) -> List:
        m = max(1, int(fraction * len(self.clients)))
        return random.sample(self.clients, m)

    def broadcast(self, selected_clients: List) -> None:
        g_state = self._global_state()
        for c in selected_clients:
            c.set_global_weights(g_state)

    def aggregate(self, payloads: List[Dict]) -> None:
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

    def run_round(self, *, fraction: float = 1.0, local_epochs: int = 1) -> Dict:
        selected = self.select_clients(fraction)
        self.broadcast(selected)
        payloads = [c.train_one_round(local_epochs=local_epochs) for c in selected]
        self.aggregate(payloads)
        return {
            "selected": [p["cid"] for p in payloads],
            "total_samples": sum(p["num_samples"] for p in payloads),
        }

    @torch.no_grad()
    def evaluate_global(self, dataset, *, batch_size: int, device, loss_fn) -> Dict[str, float]:
        model = self.global_model
        model.eval()
        device = torch.device(device)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

        total, correct = 0, 0
        total_loss = 0.0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = loss_fn(logits, y)
            total_loss += loss.item() * y.size(0)
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)

        return {
            "loss": (total_loss / total) if total > 0 else 0.0,
            "accuracy": (correct / total) if total > 0 else 0.0,
            "num_samples": total,
        }
