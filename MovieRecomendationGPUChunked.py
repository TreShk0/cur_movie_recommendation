import torch, numpy as np, pandas as pd, os, pickle, difflib, gc

# -------------------------------------------------
# 0. Вспомогалка для явного освобождения GPU‑RAM
# -------------------------------------------------
def free_memory(*vars_):
    for v in vars_:
        del v
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# -------------------------------------------------
# 1. Классические функции из прошлой версии
# -------------------------------------------------
def extract_sparse_columns(A_sparse, col_indices, num_rows, r):
    orig_idx, orig_val = A_sparse._indices(), A_sparse._values()
    mask = torch.zeros(orig_idx.shape[1], dtype=torch.bool, device=A_sparse.device)
    for ci in col_indices.cpu():
        mask |= (orig_idx[1] == ci.item())
    new_idx, new_val = orig_idx[:, mask].clone(), orig_val[mask].clone()
    mapping = -torch.ones(A_sparse.shape[1], dtype=torch.long, device=A_sparse.device)
    mapping[col_indices] = torch.arange(r, device=A_sparse.device)
    new_idx[1] = mapping[new_idx[1]]
    return torch.sparse_coo_tensor(new_idx, new_val, size=(num_rows, r)).coalesce()

def maxvol_gpu(A, e=1.05, k=100):
    n, r = A.shape
    LU, piv = torch.lu(A)
    P, L, U = torch.lu_unpack(LU, piv)
    I = torch.argmax(P[:, :r], dim=0).clone()
    Q = torch.linalg.solve_triangular(U.T, A.T, upper=False)
    Y = torch.linalg.solve_triangular(L[:r, :].T, Q, upper=True, unitriangular=True)
    B = Y.T
    for _ in range(k):
        amax = torch.abs(B).max()
        if amax <= e: break
        idx = torch.argmax(torch.abs(B))
        i, j = idx // r, idx % r
        I[j] = i
        bj, bi = B[:, j].clone(), B[i, :].clone()
        bi[j] -= 1.0
        B -= torch.outer(bj, bi) / B[i, j]
    return I, B

def maxvol_sparse_gpu(A_sparse, r, e=1.05, k=100):
    n, m = A_sparse.shape
    idx, val = A_sparse._indices(), A_sparse._values()
    col_norms = torch.zeros(m, device=A_sparse.device, dtype=A_sparse.dtype)
    col_norms = col_norms.scatter_add(0, idx[1], val**2).sqrt()
    _, J = torch.topk(col_norms, r); J, _ = torch.sort(J)
    A_cols = extract_sparse_columns(A_sparse, J, n, r).to_dense()
    I, _ = maxvol_gpu(A_cols, e, k)
    return A_cols[I, :], J, I

# -------------------------------------------------
# 2. Блочные (chunked) версии
# -------------------------------------------------
def compute_col_norms_sparse(A_sparse, chunk_size=10_000):
    n, m = A_sparse.shape
    col_norms_sq = torch.zeros(m, device=A_sparse.device, dtype=A_sparse.dtype)
    idx, val = A_sparse._indices(), A_sparse._values()
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        mask = (idx[0] >= start) & (idx[0] < end)
        if mask.any():
            col_norms_sq = col_norms_sq.scatter_add(0, idx[1, mask], val[mask] ** 2)
    return col_norms_sq.sqrt()

def sparse_to_dense_chunked(A_sparse, chunk_size=10_000):
    n, r = A_sparse.shape
    idx, val = A_sparse._indices(), A_sparse._values()
    dense_parts = []
    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        mask = (idx[0] >= start) & (idx[0] < end)
        part = torch.zeros((end - start, r), device=A_sparse.device, dtype=A_sparse.dtype)
        if mask.any():
            sub_idx = idx[:, mask].clone(); sub_idx[0] -= start
            part.index_put_((sub_idx[0], sub_idx[1]), val[mask], accumulate=True)
        dense_parts.append(part)
    return torch.cat(dense_parts, 0)

def block_maxvol_gpu(A, e=1.05, k=100, chunk_size=10_000):
    n, r = A.shape
    I = torch.arange(r, device=A.device)
    sub = A[I, :]; inv_sub = torch.pinverse(sub)
    for _ in range(k):
        max_val, max_row, max_col = 0.0, None, None
        for start in range(0, n, chunk_size):
            end = min(start + chunk_size, n)
            block = A[start:end, :]
            B_blk = block @ inv_sub
            absB = torch.abs(B_blk)
            loc_max_val = absB.max()
            if loc_max_val > max_val:
                max_val = loc_max_val.item()
                flat_idx = torch.argmax(absB)
                local_row, local_col = flat_idx // r, flat_idx % r
                max_row, max_col = start + local_row.item(), local_col.item()
        if max_val <= e: break
        I[max_col] = max_row
        sub = A[I, :]
        inv_sub = torch.inverse(sub)
    return I, inv_sub

# -------------------------------------------------
# 3. Модель
# -------------------------------------------------
class MaxvolCURModel:
    """
    CUR‑разложение с выбором столбцов/строк методом maxvol.
    fit(..., chunked=True) — использовать блочный алгоритм.
    """
    def __init__(self, r=None, device="cuda", e=1.05, k=100):
        self.r, self.device, self.e, self.k = r, device, e, k
        self.UR_ = self.cols_ = self.rows_ = None
        self.n_ = self.m_ = 0
        self.film_to_id_, self.id_to_film_ = {}, {}

    # ---------- FIT ----------
    def fit(
        self,
        ratings_csv: str,
        film_titles_csv: str | None = None,
        film_col_name: str = "title",
        save_dir: str = "model_data",
        *,
        chunked: bool = False,
        chunk_size: int = 10_000,
    ):
        """Обучение модели.  chunked=True → батчевый (RAM‑friendly) режим."""
        # --- 1. читаем CSV ---
        df = pd.read_csv(ratings_csv)
        if "userId" in df.columns: df.drop(columns=["userId"], inplace=True)
        vals_np = df.values.astype(np.float32)
        self.n_, self.m_ = vals_np.shape

        # --- 2. словари названий ---
        if film_titles_csv:
            titles_df = pd.read_csv(film_titles_csv)
            for row in titles_df.itertuples(index=False):
                fid, title = int(getattr(row, "id")), getattr(row, film_col_name)
                self.film_to_id_[title] = fid; self.id_to_film_[fid] = title
        else:
            for col_id, col_name in enumerate(df.columns):
                self.film_to_id_[col_name] = col_id; self.id_to_film_[col_id] = col_name

        # --- 3. sparse‑матрица на GPU ---
        rows_np, cols_np = np.nonzero(vals_np)
        v_nonzero = vals_np[rows_np, cols_np]
        A_sparse = torch.sparse_coo_tensor(
            torch.tensor([rows_np, cols_np], dtype=torch.long, device=self.device),
            torch.tensor(v_nonzero, dtype=torch.float32, device=self.device),
            size=(self.n_, self.m_),
        ).coalesce()
        free_memory(rows_np, cols_np, v_nonzero)

        if self.r is None: self.r = self.m_ // 2

        # ==========================================
        # 4. ВЕТВЛЕНИЕ: обычный vs. блочный вариант
        # ==========================================
        if not chunked:
            print(">> maxvol_sparse_gpu (full‑memory) ...")
            sub, cols, rows = maxvol_sparse_gpu(A_sparse, self.r, self.e, self.k)
            A_cols_dense = extract_sparse_columns(A_sparse, cols, self.n_, self.r).to_dense()
        else:
            print(">> chunked mode: compute_col_norms_sparse ...")
            col_norms = compute_col_norms_sparse(A_sparse, chunk_size)
            _, cols = torch.topk(col_norms, self.r); cols, _ = torch.sort(cols)
            print(">>   extracting selected columns ...")
            A_cols_sp = extract_sparse_columns(A_sparse, cols, self.n_, self.r)
            A_cols_dense = sparse_to_dense_chunked(A_cols_sp, chunk_size)
            print(">>   block_maxvol_gpu ...")
            rows, inv_sub = block_maxvol_gpu(A_cols_dense, self.e, self.k, chunk_size)
            sub = A_cols_dense[rows, :]
        # ----------

        self.cols_, self.rows_ = cols.cpu().numpy(), rows.cpu().numpy()
        C = vals_np[:, self.cols_]                     # (n × r)
        R = vals_np[self.rows_, :]                     # (r × m)
        U = np.linalg.inv(sub.cpu().numpy()) if not chunked else inv_sub.cpu().numpy()
        self.UR_ = U @ R                               # (r × m)

        # --- 5. сохраняем ---
        os.makedirs(save_dir, exist_ok=True)
        np.save(os.path.join(save_dir, "UR.npy"), self.UR_)
        np.save(os.path.join(save_dir, "cols.npy"), self.cols_)
        np.save(os.path.join(save_dir, "rows.npy"), self.rows_)
        with open(os.path.join(save_dir, "metadata.pkl"), "wb") as f:
            pickle.dump(
                dict(
                    n=self.n_, m=self.m_, r=self.r,
                    film_to_id_=self.film_to_id_, id_to_film_=self.id_to_film_,
                ),
                f,
            )
        print(f"Model trained (chunked={chunked}).  UR shape={self.UR_.shape}")
        free_memory(A_sparse, A_cols_dense, sub, U, C, R)

# -------------------------------------------------
# 4. Инференс (не менялся)
# -------------------------------------------------
def load_model_and_recommend(
    model_dir: str,
    user_ratings_title_based: dict[str, float],
    top_n: int = 5,
    min_score: float = 1.0,
):
    """
    Возвращает:
      recs    — топ‑N рекомендаций  [(title, score), …]
      matched — фильмы, которые модель нашла во вводе пользователя
                и действительно использовала в векторе c
                [(matched_title, score_in_model), …]
    """
    UR   = np.load(os.path.join(model_dir, "UR.npy"))
    cols = np.load(os.path.join(model_dir, "cols.npy"))
    with open(os.path.join(model_dir, "metadata.pkl"), "rb") as f:
        meta = pickle.load(f)

    film_to_id, id_to_film = meta["film_to_id_"], meta["id_to_film_"]
    m, r = meta["m"], meta["r"]

    # id фильма → позиция столбца в C
    film_id_to_pos = {int(fid): int(pos) for pos, fid in enumerate(cols)}

    # ---------- 1. формируем вектор c ----------
    c              = np.zeros(r, dtype=np.float32)
    matched        = []          # что именно «узнали» во вводе

    all_titles = list(film_to_id.keys())
    for raw_title, rating in user_ratings_title_based.items():
        match = difflib.get_close_matches(raw_title, all_titles, n=1, cutoff=0.0)
        if not match:
            continue
        fid = film_to_id[match[0]]
        if fid in film_id_to_pos:                     # фильм действительно в C
            pos = film_id_to_pos[fid]
            c[pos] = rating

    # ---------- 2. предсказание ----------
    pred_full = c @ UR                               # до зануления
    # Заполняем список matched
    for raw_title in user_ratings_title_based:
        match = difflib.get_close_matches(raw_title, all_titles, n=1, cutoff=0.0)
        if not match:
            continue
        fid = film_to_id[match[0]]
        if fid in film_id_to_pos:
            matched.append((id_to_film.get(fid, f"Film {fid}"), float(pred_full[fid])))

    # ---------- 3. маскируем уже оценённые фильмы ----------
    pred = pred_full.copy()
    for raw_title in user_ratings_title_based:
        match = difflib.get_close_matches(raw_title, all_titles, n=1, cutoff=0.0)
        if match:
            fid = film_to_id[match[0]]
            if 0 <= fid < m:
                pred[fid] = -1e9

    # ---------- 4. top‑N ----------
    idx_sorted = np.argsort(pred)[::-1]
    recs = []
    for idx in idx_sorted:
        if len(recs) >= top_n or pred[idx] < min_score:
            break
        recs.append((id_to_film.get(idx, f"Film {idx}"), float(pred[idx])))

    return recs, matched

# -------------------------------------------------
# 5. Пример запуска
# -------------------------------------------------
if __name__ == "__main__":
    model = MaxvolCURModel(r=3792, device="cuda")
    model.fit(
        ratings_csv="UI_data_2.csv",
        save_dir="cur_model_chunked",
        chunked=True,
        chunk_size=50_000,
    )

    user_ratings = {
        "Harry Potter and the Half-Blood Prince (2009)": 5,
        "Harry Potter and the Deathly Hallows: Part 2 (2011)": 5,
        "Harry Potter and the Deathly Hallows: Part 1 (2010": 5,
        "Harry Potter and the Prisoner of Azkaban (2004)": 5,
    }
    recs, matched = load_model_and_recommend(
        model_dir="my_cur_model",
        user_ratings_title_based=user_ratings,
        top_n=10,
        min_score=1.0,
    )

    print("Рекомендации:")
    for t, s in recs:
        print(f" • {t}  (pred={s:.2f})")

    print("\nСовпавшие вводы:")
    for t, s in matched:
        print(f" • {t}  (pred={s:.2f})")
