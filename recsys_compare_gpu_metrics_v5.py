"""
recsys_compare_gpu_metrics_v5.py
  – добавлены метрики MAP@K, MAR@K, Coverage и Personalization
  – оцениваются случайные пользователи ( >50 оценок ), 20 % оценок скрываются
  – результаты CUR и SVD пишутся в cur_results.csv / svd_results.csv
"""

import csv, time, random, os
import numpy as np
import pandas as pd
import cupy as cp
import torch

# -------------------- настройки --------------------
MATRIX_CSV   = "UI_data_1.csv"
DEVICE       = "cuda"
FRACTIONS    = [0.01,0.1,0.15,0.2,0.25,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
TOP_K        = 10
MIN_SCORE    = 1.0
N_EVAL_USERS = 100
TEST_RATIO   = 0.20
SEED         = 42
# ---------------------------------------------------

rng = np.random.default_rng(SEED)
random.seed(SEED)

# ---------- служебные функции (MaxVol, CUR и т.д.) ----------
# ...  ➜  оставлены без изменений из вашего файла
# (extract_sparse_columns, maxvol_gpu, maxvol_sparse_gpu,
#  cur_predict, svd_predict — полностью скопированы)

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


# ---------- метрики ----------
def apk(gt, pred, k):                 # AP@K
    if len(pred) > k: pred = pred[:k]
    num_hits = score = 0.0
    for i,p in enumerate(pred,1):
        if p in gt:
            num_hits += 1.0
            score += num_hits / i
    return score / max(1, min(len(gt), k))

def recall_k(gt, pred, k):
    if len(pred) > k: pred = pred[:k]
    return sum(p in gt for p in pred) / max(1, len(gt))

def calc_metrics(recs, gts, catalog_size):
    users = list(recs.keys())
    MAP = np.mean([apk(gts[u], recs[u], TOP_K) for u in users])
    MAR = np.mean([recall_k(gts[u], recs[u], TOP_K) for u in users])
    covered = {item for u in users for item in recs[u][:TOP_K]}
    coverage = len(covered) / catalog_size
    if len(users)>1:
        sims = []
        for i,u in enumerate(users):
            Lu = set(recs[u])
            for v in users[i+1:]:
                sims.append(len(Lu & set(recs[v]))/TOP_K)
        personalization = 1 - np.mean(sims)
    else:
        personalization = 0.0
    return MAP, MAR, coverage, personalization

# ---------- основной скрипт ----------
def main():
    print('start reading .csv-file')
    df = pd.read_csv(MATRIX_CSV, index_col=0).fillna(0).astype(np.float32)
    print('file readed')
    df = df.loc[:,(df!=0).any()]
    values  = df.values
    titles  = df.columns.tolist()
    film2id = {t:i for i,t in enumerate(titles)}
    id2film = {i:t for t,i in film2id.items()}
    mu  = values[values>0].mean()
    fro = np.linalg.norm(values)
    print('Норма посчитана')

    centered = values.copy()
    centered[values>0] -= mu
    U_cp,s_cp,Vt_cp = cp.linalg.svd(cp.asarray(centered), full_matrices=False)

    rows,cols = np.nonzero(values)
    A_sparse = torch.sparse_coo_tensor(
        torch.tensor([rows,cols],device=DEVICE),
        torch.tensor(values[rows,cols], dtype=torch.float32, device=DEVICE),
        size=values.shape).coalesce()

    # ➜ выбираем пользователей для off-line оценки
    eligible = np.where((df!=0).sum(axis=1).values > 50)[0]
    sample_users = rng.choice(eligible,
                              size=min(N_EVAL_USERS,len(eligible)),
                              replace=False)

    cur_w = csv.writer(open("cur_results.csv","w",newline="",encoding="utf-8"))
    svd_w = csv.writer(open("svd_results.csv","w",newline="",encoding="utf-8"))
    hdr = ["rank","error","time_sec",f"MAP@{TOP_K}",f"MAR@{TOP_K}","Coverage","Personalization"]
    cur_w.writerow(hdr); svd_w.writerow(hdr)

    n_users,n_movies = values.shape
    max_rank = min(n_users,n_movies)

    for frac in FRACTIONS:
        rank = max(1, round(frac*max_rank))

        # -------- CUR --------
        t0 = time.time()
        sub,cols_idx,rows_idx = maxvol_sparse_gpu(A_sparse, rank)
        C   = centered[:,cols_idx]
        R = centered[rows_idx, :]
        Ucur = np.linalg.pinv(sub.cpu().numpy())
        UR   = Ucur @ R          # R-матрица в CUR
        err_cur = np.linalg.norm(C@UR+mu - values)/fro
        t_cur  = time.time()-t0

        # -------- SVD (k=rank) --------
        t0 = time.time()
        k = min(rank, len(s_cp))
        approx_svd = (U_cp[:,:k]*(s_cp[:k])) @ Vt_cp[:k,:] + mu
        err_svd = np.linalg.norm(approx_svd.get() - values) / fro
        t_svd  = time.time()-t0

        # --------- off-line метрики --------
        recs_cur, recs_svd, gts = {},{},{}
        for u in sample_users:
            rated = np.where(values[u]>0)[0]
            n_test = max(1, int(TEST_RATIO*len(rated)))
            test   = rng.choice(rated, size=n_test, replace=False)
            train  = np.setdiff1d(rated, test, assume_unique=True)

            gt_titles = {id2film[i] for i in test}
            gts[u] = gt_titles

            vec = np.zeros(n_movies, np.float32)
            vec[train] = values[u,train]
            known = {id2film[i] for i in train}

            recs_cur[u] = [t for t,_ in cur_predict(
                              vec, cols_idx, UR, film2id, id2film, known, mu,
                              top_n=TOP_K, min_score=MIN_SCORE)]
            recs_svd[u] = [t for t,_ in svd_predict(
                              vec, Vt_cp[:k,:].get(), np.diag(s_cp[:k].get()),
                              mu, titles, known, top_n=TOP_K, min_score=MIN_SCORE)]

        map_c,mar_c,cov_c,pers_c = calc_metrics(recs_cur,gts,n_movies)
        map_s,mar_s,cov_s,pers_s = calc_metrics(recs_svd,gts,n_movies)

        cur_w.writerow([rank,f"{err_cur:.6f}",f"{t_cur:.2f}",
                        f"{map_c:.4f}",f"{mar_c:.4f}",
                        f"{cov_c:.4f}",f"{pers_c:.4f}"])
        svd_w.writerow([rank,f"{err_svd:.6f}",f"{t_svd:.2f}",
                        f"{map_s:.4f}",f"{mar_s:.4f}",
                        f"{cov_s:.4f}",f"{pers_s:.4f}"])

        print(f"[+] rank={rank}  CUR MAP={map_c:.3f}  SVD MAP={map_s:.3f}")

if __name__ == "__main__":
    main()
