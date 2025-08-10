# main.py
import torch, random, numpy as np

from src.utils.config import DATASET, MODEL, TRAINING, OPTIMIZER, LOSS_FN
from src.utils.models import TwoNN
from src.client.FedAvg import FedAvgClient
from src.server.FedAvg import FedAvgServer
from generate_data import mnist_subsets

def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def main():
    # ---- seed & device ----
    set_seed(TRAINING["seed"])
    device = TRAINING["device"]
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    print(f"Device: {device}")

    # ---- data split (IID / Dirichlet non-IID) ----
    train_subsets, testset = mnist_subsets(
        n_clients=DATASET["num_clients"],
        scheme=DATASET["partition"],          # "iid" 或 "dirichlet"/"noniid"
        alpha=DATASET["dirichlet_alpha"],
        seed=TRAINING["seed"],
        root=DATASET["root"],
    )

    # ---- model ----
    global_model = TwoNN(
        in_dim=MODEL["in_dim"],
        hidden=MODEL["hidden"],
        num_classes=MODEL["num_classes"],
    )

    # ---- clients & server ----
    clients = [
        FedAvgClient(
            cid=i,
            model=global_model,
            dataset=train_subsets[i],
            lr=OPTIMIZER["lr"],
            batch_size=TRAINING["train_batch_size"],
            device=device,
        )
        for i in range(DATASET["num_clients"])
    ]
    server = FedAvgServer(global_model, clients, device=device)

    # ---- training loop ----
    for r in range(TRAINING["rounds"]):
        stats = server.run_round(
            fraction=TRAINING["fraction"],
            local_epochs=TRAINING["local_epochs"],
        )
        metrics = server.evaluate_global(
            dataset=testset,
            batch_size=TRAINING["eval_batch_size"],
            device=device,
            loss_fn=LOSS_FN,
        )
        print(f"[Round {r}] {stats}  acc={metrics['accuracy']:.4f}  loss={metrics['loss']:.4f}")

if __name__ == "__main__":
    main()
