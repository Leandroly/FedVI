import torch
from torch.utils.data import DataLoader
from copy import deepcopy

class ScaffoldClient:
    def __init__(self, cid, model, dataset, *, lr, batch_size, device):
        self.cid = cid
        self.device = torch.device(device)
        self.model = deepcopy(model).to(self.device)
        self.dataset = dataset
        self.lr = lr
        self.batch_size = batch_size
        
        with torch.no_grad():
            self.c_local = {k: torch.zeros_like(v, device=self.device) for k, v in self.model.state_dict().items()}
            self.c_global = {k: torch.zeros_like(v, device=self.device) for k, v in self.model.state_dict().items()}

    def _loader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True, drop_last=False)

    def train_one_round(self, *, local_epochs=1):
        self.model.train()
        loss_fn = torch.nn.CrossEntropyLoss()
        opt =  torch.optim.SGD(self.model.parameters(), lr=self.lr)
        loader = self._loader()

        with torch.no_grad():
            w0 = {k: v.clone() for k, v in self.model.state_dict().items()}
        
        K = 0
        for _ in range(local_epochs):
            for x, y in loader:
                x, y = x.to(self.device), y.to(self.device)
                opt.zero_grad()
                logits = self.model(x)
                loss = loss_fn(logits, y)
                loss.backward()

                # fix grad with g = g - c_local + c_global
                with torch.no_grad():
                    for name, g in self.model.named_parameters():
                        if g.grad is None:
                            continue
                        g.grad.add_(self.c_global[name] - self.c_local[name])
                
                opt.step()
                K += 1