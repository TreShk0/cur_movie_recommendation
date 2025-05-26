
"""
recsys_compare_gpu_v2.py
Сравнение CUR и SVD (GPU) по ошибке и рекомендациям.

* Загружает матрицу всего один раз.
* Перебирает ранги/факторы = FRACTIONS * min(n,m).
* Для CUR использует ту же логику MaxvolCURModel:
      C,U,R → UR, c@UR (учёт индексов).
* Для SVD использует формулу u = r V Σ^{-1}.

Сохраняет результаты в cur_results.csv и svd_results.csv
"""

import os, csv, time, math, difflib
import numpy as np
import pandas as pd
import cupy as cp
import torch
from typing import List, Tuple, Dict

# ---------------- Настройки ----------------
MATRIX_CSV = "UI_data_2.csv"
DEVICE     = "cuda"
FRACTIONS  = [0.01, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5,0.6,0.7,0.8,0.9,0.95,0.99]
TOP_N      = 10
MIN_SCORE  = 1.0

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
    "Harry Potter and the Deathly Hallows: Part 1 (2010)": 5,
    "Harry Potter and the Prisoner of Azkaban (2004)": 5,
    'Hot Fuzz (2007)': 5,
    "In Bruges (2008)": 5,
    "The Nice Guys (2016)": 5,
    "Snatch (2000)": 4
}
# -------------------------------------------

# ---------- CUR helpers (из оригинала) -----
def extract_sparse_columns(A_sparse, col_indices, num_rows, r):
    orig_idx = A_sparse._indices()
    orig_val = A_sparse._values()
    mask = torch.zeros(orig_idx.shape[1], dtype=torch.bool, device=A_sparse.device)
    for ci in col_indices.cpu():
        mask |= (orig_idx[1] == ci.item())
    new_idx = orig_idx[:, mask].clone()
    new_val = orig_val[mask].clone()
    mapping = -torch.ones(A_sparse.shape[1], dtype=torch.long, device=A_sparse.device)
    mapping[col_indices] = torch.arange(r, device=A_sparse.device)
    new_idx[1] = mapping[new_idx[1]]
    A_cols = torch.sparse_coo_tensor(new_idx, new_val, size=(num_rows, r)).coalesce()
    return A_cols

def maxvol_gpu(A, e=1.05, k=100):
    n, r = A.shape
    LU, piv = torch.lu(A)
    P,L,U_mat = torch.lu_unpack(LU, piv)
    I = torch.argmax(P[:, :r], dim=0).clone()
    Q = torch.linalg.solve_triangular(U_mat.T, A.T, upper=False, left=True)
    Y = torch.linalg.solve_triangular(L[:r, :].T, Q, upper=True, left=True, unitriangular=True)
    B = Y.T
    for _ in range(k):
        absB = torch.abs(B)
        max_idx = torch.argmax(absB)
        i = max_idx // r
        j = max_idx % r
        if absB[i, j] <= e:
            break
        I[j] = i
        bj = B[:, j].clone()
        bi = B[i, :].clone()
        bi[j] -= 1.0
        B = B - torch.outer(bj, bi) / B[i, j]
    return I, B

def maxvol_sparse_gpu(A_sparse, r, e=1.05, k=100):
    n,m = A_sparse.shape
    idx = A_sparse._indices(); val = A_sparse._values()
    col_norms = torch.zeros(m, device=A_sparse.device, dtype=A_sparse.dtype)
    col_norms = col_norms.scatter_add(0, idx[1], val**2).sqrt()
    _, J = torch.topk(col_norms, r)
    J, _ = torch.sort(J)
    A_cols = extract_sparse_columns(A_sparse, J, n, r).to_dense()
    I,_ = maxvol_gpu(A_cols, e=e, k=k)
    submatrix = A_cols[I,:]
    return submatrix, J, I

def build_c_vector(cols_idx: np.ndarray,
                   film_to_id: Dict[str,int],
                   user_ratings: Dict[str,int]) -> np.ndarray:
    r = len(cols_idx)
    c = np.zeros(r, dtype=np.float32)
    for title, rating in user_ratings.items():
        if title in film_to_id:
            fid = film_to_id[title]
            # в каких столбцах оригинала этот fid? если он выбран в C, выставим
            if fid in cols_idx:
                pos = np.where(cols_idx == fid)[0][0]
                c[pos] = rating
    return c

def cur_predict(UR: np.ndarray,
                cols_idx: np.ndarray,
                film_to_id: Dict[str,int],
                id_to_film: Dict[int,str],
                user_ratings: Dict[str,int],
                top_n:int=10,
                min_score:float=1.0) -> List[Tuple[str,float]]:
    c = build_c_vector(cols_idx, film_to_id, user_ratings)
    pred = c @ UR
    # hide known
    for title in user_ratings:
        if title in film_to_id:
            fid = film_to_id[title]
            pred[fid] = -1e9
    idx_sorted = np.argsort(pred)[::-1]
    recs=[]
    for idx in idx_sorted:
        score = pred[idx]
        if score<min_score or len(recs)>=top_n: break
        recs.append((id_to_film.get(idx,f"Film {idx}"), float(score)))
    return recs

# ---------- SVD recommend -----------------
def svd_latent_predict(ratings_vec: np.ndarray,
                       Vt_k: np.ndarray,
                       S_k: np.ndarray,
                       global_mean: float,
                       titles: List[str],
                       known_titles: List[str],
                       top_n:int=10,
                       min_score:float=1.0)->List[Tuple[str,float]]:
    r_centered = ratings_vec.copy()
    mask = r_centered !=0
    r_centered[mask] -= global_mean
    s_vals = np.diag(S_k)
    s_inv = np.where(s_vals>1e-8,1.0/s_vals,0.0)
    u = (r_centered @ Vt_k.T)*s_inv
    pred = u @ S_k @ Vt_k + global_mean
    recs=[(t,float(s)) for t,s in zip(titles,pred)
          if t not in known_titles and not np.isnan(s) and s>=min_score]
    recs.sort(key=lambda x:x[1], reverse=True)
    return recs[:top_n]

# ---------------- MAIN --------------------
def main():
    print("Loading rating matrix …")
    df = pd.read_csv(MATRIX_CSV, index_col=0).astype(np.float32)
    df = df.loc[:, (df!=0).any()]          # remove empty films
    titles = df.columns.tolist()
    film_to_id = {t:i for i,t in enumerate(titles)}
    id_to_film = {i:t for t,i in film_to_id.items()}
    values = df.values
    fro = np.linalg.norm(values)
    n_users, n_movies = values.shape
    max_rank = min(n_users, n_movies)
    # ------------- SVD full --------------
    mask = values>0
    global_mean = values[mask].mean()
    centered = values.copy()
    centered[mask] -= global_mean
    U_cp, s_cp, Vt_cp = cp.linalg.svd(cp.asarray(centered), full_matrices=False)
    # ------------- prepare sparse torch --
    rows_np, cols_np = np.nonzero(values)
    vals_np = values[rows_np, cols_np]
    A_sparse = torch.sparse_coo_tensor(
        torch.tensor([rows_np, cols_np], device=DEVICE),
        torch.tensor(vals_np, dtype=torch.float32, device=DEVICE),
        size=(n_users, n_movies)).coalesce()
    # ------------- output files ----------
    cur_f = open("cur_results.csv","w",newline='',encoding='utf-8'); cur_w=csv.writer(cur_f)
    svd_f = open("svd_results.csv","w",newline='',encoding='utf-8'); svd_w=csv.writer(svd_f)
    cur_w.writerow(["rank","error","recommendations"])
    svd_w.writerow(["factors","error","recommendations"])
    # ----------- iterate -----------------
    for frac in FRACTIONS:
        r = max(1, int(round(frac*max_rank)))
        # ---- CUR ----
        sub, cols_idx_t, rows_idx_t = maxvol_sparse_gpu(A_sparse, r)
        cols_idx = cols_idx_t.cpu().numpy()
        rows_idx = rows_idx_t.cpu().numpy()
        C = values[:, cols_idx]
        R = values[rows_idx, :]
        U = np.linalg.inv(sub.cpu().numpy())
        UR = U @ R                       # shape r×m
        approx_cur = C @ UR
        err_cur = np.linalg.norm(approx_cur - values)/fro
        recs_cur = cur_predict(UR, cols_idx, film_to_id, id_to_film,
                               USER_RATINGS, top_n=TOP_N, min_score=MIN_SCORE)
        rec_str_cur = "|".join([f"{t}:{s:.2f}" for t,s in recs_cur])
        cur_w.writerow([r,f"{err_cur:.5f}",rec_str_cur])
        cur_f.flush()
        # ---- SVD ----
        k = min(r, len(s_cp))
        U_k = U_cp[:, :k]; s_k = s_cp[:k]; Vt_k = Vt_cp[:k,:]
        approx_svd = (U_k*s_k) @ Vt_k
        approx_svd = approx_svd.get() + global_mean
        err_svd = np.linalg.norm(approx_svd - values)/fro
        # build user vector
        user_vec = np.zeros(n_movies, dtype=np.float32)
        for t,score in USER_RATINGS.items():
            if t in film_to_id:
                user_vec[film_to_id[t]] = score
        recs_svd = svd_latent_predict(user_vec, Vt_k.get(), np.diag(s_k.get()), global_mean,
                                      titles, list(USER_RATINGS.keys()),
                                      top_n=TOP_N, min_score=MIN_SCORE)
        rec_str_svd = "|".join([f"{t}:{s:.2f}" for t,s in recs_svd])
        svd_w.writerow([k, f"{err_svd:.5f}", rec_str_svd])
        svd_f.flush()
        print(f"Done rank={r}")
    cur_f.close(); svd_f.close()
    print("Complete. Results saved.")

if __name__ == "__main__":
    main()
