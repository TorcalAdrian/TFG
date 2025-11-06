# ae_kmeans_pipeline.py
import os, time, random, argparse
import numpy as np
import pandas as pd
from collections import Counter
from tqdm import tqdm
import faiss

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score


# ===== Utilities =====
def set_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def load_transactions_csv(file_path):
    classes, transactions = [], []
    with open(file_path, mode='rt', encoding='utf-8') as f:
        next(f)
        for line in f:
            parts = line.strip().split(';')
            if len(parts) < 3:
                continue
            cls = parts[1].strip()
            evs = parts[2].strip().split()
            classes.append(cls)
            transactions.append(evs)
    return np.array(classes), transactions

def build_vocabulary(transactions, min_freq=1, max_vocab=None):
    cnt = Counter()
    for tr in transactions:
        cnt.update(tr)
    items = [it for it, c in cnt.items() if c >= min_freq]
    if max_vocab is not None and len(items) > max_vocab:
        items = [it for it, _ in cnt.most_common(max_vocab)]
    item2idx = {it: i for i, it in enumerate(sorted(items))}
    return item2idx

def transactions_to_multihot(transactions, item2idx):
    N, V = len(transactions), len(item2idx)
    X = np.zeros((N, V), dtype=np.float32)
    for i, tr in enumerate(transactions):
        for it in tr:
            j = item2idx.get(it, None)
            if j is not None:
                X[i, j] = 1.0
    return X


# ===== Autoencoder =====
class AE(nn.Module):
    def __init__(self, input_dim, latent_dim=64, hidden=[512, 256], dropout=0.0):
        super().__init__()
        enc_layers, prev = [], input_dim
        for h in hidden:
            enc_layers += [nn.Linear(prev, h), nn.ReLU()]
            if dropout > 0:
                enc_layers += [nn.Dropout(dropout)]
            prev = h
        self.encoder = nn.Sequential(*enc_layers, nn.Linear(prev, latent_dim))
        dec_layers, prev = [], latent_dim
        for h in reversed(hidden):
            dec_layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        dec_layers += [nn.Linear(prev, input_dim), nn.Sigmoid()]
        self.decoder = nn.Sequential(*dec_layers)

    def forward(self, x):
        z = self.encoder(x)
        xhat = self.decoder(z)
        return xhat, z


def train_autoencoder(X, latent_dim=64, hidden=[512, 256],
                      epochs=10, batch_size=256, lr=1e-3, device='cpu'):
    model = AE(input_dim=X.shape[1], latent_dim=latent_dim, hidden=hidden).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()
    ds = TensorDataset(torch.from_numpy(X))
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=False)

    start = time.time()
    model.train()
    for ep in range(epochs):
        ep_loss = 0.0
        for (xb,) in dl:
            xb = xb.to(device)
            xhat, _ = model(xb)
            loss = criterion(xhat, xb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            ep_loss += loss.item() * xb.size(0)
        avg = ep_loss / X.shape[0]
        print(f"[AE] epoch {ep+1}/{epochs} | loss={avg:.5f}")
    end = time.time()
    return model, end - start


def encode_latents(model, X, batch_size=512, device='cpu'):
    model.eval()
    ds = TensorDataset(torch.from_numpy(X))
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, drop_last=False)
    latents = []
    with torch.no_grad():
        for (xb,) in dl:
            xb = xb.to(device)
            _, z = model(xb)
            latents.append(z.cpu().numpy())
    return np.vstack(latents).astype(np.float32)


# ===== K-Means =====
def run_kmeans_faiss(embeddings, k, niter=20):
    embeddings = embeddings.astype(np.float32)
    dim = embeddings.shape[1]
    kmeans = faiss.Kmeans(d=dim, k=k, niter=niter, verbose=False)
    kmeans.train(embeddings)
    y_pred = kmeans.index.search(embeddings, 1)[1].flatten()
    return y_pred

def run_kmeans_sklearn(embeddings, k, n_init=10, random_state=42):
    km = KMeans(n_clusters=k, n_init=n_init, random_state=random_state)
    return km.fit_predict(embeddings)

def compute_nmi(y_true, y_pred):
    return normalized_mutual_info_score(y_true, y_pred)


# ===== Main =====
def main(args):
    set_seeds(42)
    device = 'cuda' if torch.cuda.is_available() and not args.cpu else 'cpu'
    print(f"Device: {device}")

    y_true, transactions = load_transactions_csv(args.data)
    print(f"Transactions: {len(transactions)}")

    item2idx = build_vocabulary(transactions, min_freq=args.min_freq, max_vocab=args.max_vocab)
    print(f"Vocab size: {len(item2idx)} (min_freq={args.min_freq}, max_vocab={args.max_vocab})")
    X = transactions_to_multihot(transactions, item2idx)

    model, t_train = train_autoencoder(
        X,
        latent_dim=args.latent_dim,
        hidden=[int(h) for h in args.hidden.split(',')],
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=device
    )
    print(f"AE training time: {t_train:.2f}s")

    t0 = time.time()
    Z = encode_latents(model, X, batch_size=max(512, args.batch_size), device=device)
    t_embed = time.time() - t0
    print(f"Encoding time: {t_embed:.2f}s | Z.shape={Z.shape}")

    for k in args.k_list:
        t1 = time.time()
        if args.use_faiss:
            y_pred = run_kmeans_faiss(Z, k=k, niter=20)
            method = "FAISS"
        else:
            y_pred = run_kmeans_sklearn(Z, k=k, n_init=args.kmeans_n_init, random_state=42)
            method = "Sklearn"
        t_kmeans = time.time() - t1
        nmi = compute_nmi(y_true, y_pred)
        print(f"[{method}] k={k} | NMI={nmi:.4f} | time={t_kmeans:.2f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--min_freq", type=int, default=1)
    parser.add_argument("--max_vocab", type=int, default=None)
    parser.add_argument("--latent_dim", type=int, default=64)
    parser.add_argument("--hidden", type=str, default="512,256")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--k_list", type=int, nargs="+", default=[4,8,16])
    parser.add_argument("--kmeans_n_init", type=int, default=10)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--use_faiss", type=bool, default=False, help="Use FAISS K-Means if True, otherwise sklearn")
    args = parser.parse_args()
    main(args)
