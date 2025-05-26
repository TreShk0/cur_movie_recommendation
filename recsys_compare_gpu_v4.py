
"""
recsys_compare_gpu_v4.py
— корректная привязка индексов после MaxVol в CUR
— поддержка времени на каждое разложение
"""

import os, csv, time
import numpy as np
import pandas as pd
import cupy as cp
import torch

MATRIX_CSV = "UI_data_3.csv"
DEVICE     = "cuda"
FRACTIONS  = [0.01, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
TOP_N = 10
MIN_SCORE = 1.0

USER_RATINGS = {
    "Friends with Benefits (2011)": 3,
    'Devil Wears Prada, The (2006)': 5,
    'Crazy, Stupid, Love. (2011)': 5,
    'Palm Springs (2020)': 3,
    'Hangover, The (2009)': 5,
    "The Devil's Advocate (1997)": 5,
    "Bad Boys (1995)": 5,
    "Bad Boys II (2003)": 4,
    'Avatar (2009)': 5,
    "Avatar: The Way of Water (2022)": 5,
    "Harry Potter and the Half-Blood Prince (2009)": 5,
    "Harry Potter and the Deathly Hallows: Part 2 (2011)": 5,
    "Harry Potter and the Deathly Hallows: Part 1 (2010": 5,
    "Harry Potter and the Prisoner of Azkaban (2004)": 5,
    'Hot Fuzz (2007)': 5,
    "In Bruges (2008)": 5,
    "The Nice Guys (2016)": 5,
    "Snatch (2000)": 4
}

def extract_sparse_columns(A_sparse, col_indices, n_rows, r):
    idx = A_sparse._indices(); val = A_sparse._values()
    mask = torch.zeros(idx.shape[1], dtype=torch.bool, device=A_sparse.device)
    for ci in col_indices.cpu(): mask |= (idx[1] == ci.item())
    new_idx = idx[:, mask].clone(); new_val = val[mask].clone()
    mapping = -torch.ones(A_sparse.shape[1], dtype=torch.long, device=A_sparse.device)
    mapping[col_indices] = torch.arange(r, device=A_sparse.device)
    new_idx[1] = mapping[new_idx[1]]
    return torch.sparse_coo_tensor(new_idx, new_val, size=(n_rows, r)).coalesce()

def maxvol_gpu(A, e=1.05, k=100):
    n, r = A.shape
    LU, piv = torch.lu(A)
    P, L, U_mat = torch.lu_unpack(LU, piv)
    I = torch.argmax(P[:, :r], dim=0).clone()
    Q = torch.linalg.solve_triangular(U_mat.T, A.T, upper=False, left=True)
    Y = torch.linalg.solve_triangular(L[:r, :].T, Q, upper=True, left=True, unitriangular=True)
    B = Y.T
    for _ in range(k):
        absB = torch.abs(B); max_idx = torch.argmax(absB)
        i, j = divmod(max_idx.item(), r)
        if absB[i, j] <= e: break
        I[j] = i
        bj, bi = B[:, j].clone(), B[i, :].clone(); bi[j] -= 1
        B -= torch.outer(bj, bi) / B[i, j]
    return I, B

def maxvol_sparse_gpu(A_sparse, r, e=1.05, k=100):
    n, m = A_sparse.shape
    idx = A_sparse._indices(); val = A_sparse._values()
    col_norms = torch.zeros(m, device=A_sparse.device, dtype=A_sparse.dtype)
    col_norms = col_norms.scatter_add(0, idx[1], val**2).sqrt()
    _, J = torch.topk(col_norms, r); J, _ = torch.sort(J)
    A_cols = extract_sparse_columns(A_sparse, J, n, r).to_dense()
    I, _ = maxvol_gpu(A_cols, e=e, k=k)
    return A_cols[I, :], J.cpu().numpy(), I.cpu().numpy()

def cur_predict(user_vec, cols_idx, UR, film_to_id, id_to_film, known, mu, top_n=10, min_score=1.0):
    c = user_vec[cols_idx]  # правильное соответствие
    pred = c @ UR + mu
    for t in known:
        if t in film_to_id:
            pred[film_to_id[t]] = -1e9
    top_idx = np.argsort(pred)[::-1]
    recs = [(id_to_film[i], float(pred[i])) for i in top_idx if pred[i] >= min_score and not np.isnan(pred[i])]
    return recs[:top_n]

def svd_predict(user_vec, Vt_k, S_k, mu, titles, known, top_n=10, min_score=1.0):
    r_c = user_vec.copy(); m = r_c != 0; r_c[m] -= mu
    s = np.diag(S_k); s_inv = np.where(s > 1e-8, 1/s, 0)
    u = (r_c @ Vt_k.T) * s_inv
    pred = u @ S_k @ Vt_k + mu
    recs = [(t, float(p)) for t, p in zip(titles, pred) if t not in known and p >= min_score and not np.isnan(p)]
    recs.sort(key=lambda x: x[1], reverse=True)
    return recs[:top_n]

def main():
    t0 = time.time()
    df = pd.read_csv(MATRIX_CSV, index_col=0).fillna(0.0).astype(np.float32)
    print(f"[INFO] CSV loaded in {time.time()-t0:.2f}s")
    df = df.loc[:, (df != 0).any()]
    titles = df.columns.tolist()
    film2id = {t: i for i, t in enumerate(titles)}
    id2film = {i: t for t, i in film2id.items()}
    values = df.values
    fro = np.linalg.norm(values)
    n_users, n_movies = values.shape
    max_rank = min(n_users, n_movies)

    mu = values[values > 0].mean()
    centered = values.copy(); centered[values > 0] -= mu
    U_cp, s_cp, Vt_cp = cp.linalg.svd(cp.asarray(centered), full_matrices=False)

    rows, cols = np.nonzero(values); vals = values[rows, cols]
    A_sparse = torch.sparse_coo_tensor(torch.tensor([rows, cols], device=DEVICE),
                                       torch.tensor(vals, dtype=torch.float32, device=DEVICE),
                                       size=(n_users, n_movies)).coalesce()

    cur_w = csv.writer(open("cur_results.csv", "w", newline="", encoding="utf-8"))
    svd_w = csv.writer(open("svd_results.csv", "w", newline="", encoding="utf-8"))
    cur_w.writerow(["rank", "error", "time_sec", "recommendations"])
    svd_w.writerow(["factors", "error", "time_sec", "recommendations"])

    user_vec = np.zeros(n_movies, np.float32)
    for t, s in USER_RATINGS.items():
        if t in film2id: user_vec[film2id[t]] = s
    known = list(USER_RATINGS.keys())

    for frac in FRACTIONS:
        r = max(1, int(round(frac * max_rank)))

        t1 = time.time()
        sub, cols_idx, rows_idx = maxvol_sparse_gpu(A_sparse, r)
        #C = values[:, cols_idx]; R = values[rows_idx, :]
        C = centered[:, cols_idx]; R = centered[rows_idx, :]
        U = np.linalg.pinv(sub.cpu().numpy())
        UR = U @ R
        approx = C @ UR
        approx = approx + mu
        err = np.linalg.norm(approx - values) / fro
        recs = cur_predict(user_vec, cols_idx, UR, film2id, id2film, known, mu, TOP_N, MIN_SCORE)
        cur_w.writerow([r, f"{err:.6f}", f"{time.time()-t1:.2f}", "|".join(f"{t}:{s:.2f}" for t,s in recs)])

        t2 = time.time()
        k = min(r, len(s_cp))
        U_k = U_cp[:, :k]; s_k = s_cp[:k]; Vt_k = Vt_cp[:k, :]
        approx_svd = (U_k * s_k) @ Vt_k
        approx_svd = approx_svd.get() + mu
        err_svd = np.linalg.norm(approx_svd - values) / fro
        recs_svd = svd_predict(user_vec, Vt_k.get(), np.diag(s_k.get()), mu, titles, known, TOP_N, MIN_SCORE)
        svd_w.writerow([k, f"{err_svd:.6f}", f"{time.time()-t2:.2f}", "|".join(f"{t}:{s:.2f}" for t,s in recs_svd)])

        print(f"[+] Rank={r} done")

if __name__ == "__main__":
    main()
