import torch
import numpy as np
import pandas as pd
import difflib
import os
import pickle

############################################
# Вспомогательные функции (без изменений)
############################################
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
    if n <= r:
        raise ValueError("A must be tall (n > r)")

    LU, pivots = torch.lu(A)
    P, L, U_mat = torch.lu_unpack(LU, pivots)

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
        bi[j] = bi[j] - 1.0
        B = B - torch.outer(bj, bi) / B[i, j]

    return I, B

def maxvol_sparse_gpu(A_sparse, r, e=1.05, k=100):
    n, m = A_sparse.shape
    idx = A_sparse._indices()
    val = A_sparse._values()

    col_norms = torch.zeros(m, device=A_sparse.device, dtype=A_sparse.dtype)
    col_norms = col_norms.scatter_add(0, idx[1], val ** 2).sqrt()

    _, J = torch.topk(col_norms, r)
    J, _ = torch.sort(J)

    A_cols = extract_sparse_columns(A_sparse, J, n, r).to_dense()
    I, _ = maxvol_gpu(A_cols, e=e, k=k)
    submatrix = A_cols[I, :]

    return submatrix, J, I

###################################################
# Модель
###################################################
class MaxvolCURModel:
    """
    Обучается на матрице (пользователи × фильмы), считает CUR‑разложение
    и хранит все нужные соответствия.
    """
    def __init__(self, r=None, device="cuda", e=1.05, k=100):
        self.r = r
        self.device = device
        self.e = e
        self.k = k

        self.UR_ = None
        self.cols_ = None
        self.rows_ = None
        self.n_ = None
        self.m_ = None

        self.film_to_id_ = {}
        self.id_to_film_ = {}

    # ---------- FIT ----------
    def fit(
        self,
        ratings_csv: str,
        film_titles_csv: str | None = None,
        film_col_name: str = "title",
        save_dir: str = "model_data",
    ):
        # 1. Загружаем матрицу рейтингов
        df = pd.read_csv(ratings_csv, nrows=100_000)
        if "userId" in df.columns:
            df.drop(columns=["userId"], inplace=True)
        values = df.values.astype(np.float32)
        self.n_, self.m_ = values.shape

        # 2. Строим словари названий
        if film_titles_csv is not None:
            titles_df = pd.read_csv(film_titles_csv)
            for row in titles_df.itertuples(index=False):
                fid = getattr(row, "id")
                title = getattr(row, film_col_name)
                self.film_to_id_[title] = int(fid)
                self.id_to_film_[int(fid)] = title
        else:
            # !!! ИСПРАВЛЕНО: используем названия столбцов как они есть
            for col_id, col_name in enumerate(df.columns):
                self.film_to_id_[col_name] = col_id
                self.id_to_film_[col_id] = col_name

        # 3. Готовим sparse A
        if self.r is None:
            self.r = self.m_ // 2

        rows_np, cols_np = np.nonzero(values)
        vals_np = values[rows_np, cols_np]

        indices = torch.tensor([rows_np, cols_np], dtype=torch.long, device=self.device)
        vals = torch.tensor(vals_np, dtype=torch.float32, device=self.device)
        A_sparse = torch.sparse_coo_tensor(indices, vals, size=(self.n_, self.m_)).coalesce()

        # 4. Maxvol
        submatrix, cols, rows = maxvol_sparse_gpu(A_sparse, self.r, self.e, self.k)
        self.cols_ = cols.cpu().numpy()      # (r,)
        self.rows_ = rows.cpu().numpy()      # (r,)

        C = values[:, self.cols_]            # (n × r)
        R = values[self.rows_, :]            # (r × m)

        U = np.linalg.inv(submatrix.cpu().numpy())
        self.UR_ = U @ R                     # (r × m)
        error = np.linalg.norm(C@U@R - values)/np.linalg.norm(values)

        # 5. Сохраняем
        os.makedirs(save_dir, exist_ok=True)
        np.save(os.path.join(save_dir, "UR.npy"), self.UR_)
        np.save(os.path.join(save_dir, "cols.npy"), self.cols_)
        np.save(os.path.join(save_dir, "rows.npy"), self.rows_)
        with open(os.path.join(save_dir, "metadata.pkl"), "wb") as f:
            pickle.dump(
                {
                    "n": self.n_,
                    "m": self.m_,
                    "r": self.r,
                    "film_to_id_": self.film_to_id_,
                    "id_to_film_": self.id_to_film_,
                },
                f,
            )

        print(f"Model trained. UR shape={self.UR_.shape}. Saved to {save_dir}")
        print(f'Error: {error:.4f}')

###################################################
# Функция inference
###################################################
def load_model_and_recommend(
    model_dir: str,
    user_ratings_title_based: dict[str, float],
    top_n: int = 5,
    min_score: float = 1.0,
):
    """
    model_dir — папка, где лежит модель.
    user_ratings_title_based — {'название фильма': рейтинг}.
    """
    UR = np.load(os.path.join(model_dir, "UR.npy"))      # (r × m)
    cols = np.load(os.path.join(model_dir, "cols.npy"))  # (r,)

    with open(os.path.join(model_dir, "metadata.pkl"), "rb") as f:
        meta = pickle.load(f)

    film_to_id = meta["film_to_id_"]
    id_to_film = meta["id_to_film_"]
    m = meta["m"]
    r = meta["r"]

    # ⇢ Быстрый маппинг: id фильма → позиция в C
    film_id_to_pos = {int(fid): int(pos) for pos, fid in enumerate(cols)}

    # 1. Вектор пользовательских рейтингов (1 × r)
    c = np.zeros(r, dtype=np.float32)
    all_titles = list(film_to_id.keys())

    for raw_title, rating in user_ratings_title_based.items():
        match = difflib.get_close_matches(raw_title, all_titles, n=1, cutoff=0.0)
        if not match:
            print(f"'{raw_title}' — не нашёл похожий фильм, пропускаю.")
            continue
        best_title = match[0]
        fid = film_to_id[best_title]
        if fid in film_id_to_pos:
            c[film_id_to_pos[fid]] = rating  # корректная позиция!
        # иначе фильм не вошёл в C — игнорируем

    # 2. Предсказание: c × UR
    pred = c @ UR                           # (m,)

    # 3. Скрываем уже оценённые фильмы
    for raw_title in user_ratings_title_based:
        match = difflib.get_close_matches(raw_title, all_titles, n=1, cutoff=0.0)
        if match:
            fid = film_to_id[match[0]]
            if 0 <= fid < m:
                pred[fid] = -1e9

    # 4. Top‑N
    idx_sorted = np.argsort(pred)[::-1]
    recs = []
    for idx in idx_sorted:
        score = pred[idx]
        if score < min_score or len(recs) >= top_n:
            break
        recs.append((id_to_film.get(idx, f"Film {idx}"), float(score)))

    return recs

###################################################
# Пример запуска
###################################################

import time
if __name__ == "__main__":
    start = time.time()
    model = MaxvolCURModel(r=5000, device="cuda")
    model.fit(
        ratings_csv="UI_data_2.csv",
        film_titles_csv=None,          # названия берутся из заголовков
        save_dir="my_cur_model_1",
    )

    
    user_ratings = {
        "Friends with Benefits (2011)": 3,
        'Devil Wears Prada, The (2006)': 5,
        'Crazy, Stupid, Love. (2011)':5,
        'Palm Springs (2020)':3,
        'Hangover, The (2009)':5,
        "The Devil's Advocate (1997)": 5,
        "Bad Boys (1995)":5,
        "Bad Boys II (2003)":4,
        'Avatar (2009)':5,
        "Avatar: The Way of Water (2022)":5,
        "Harry Potter and the Half-Blood Prince (2009)": 5,
        "Harry Potter and the Deathly Hallows: Part 2 (2011)": 5,
        "Harry Potter and the Deathly Hallows: Part 1 (2010)": 5,
        "Harry Potter and the Prisoner of Azkaban (2004)": 5,
        'Hot Fuzz (2007)': 5,
        "In Bruges (2008)":5,
        "The Nice Guys (2016)":5,
        "Snatch (2000)": 4
    }

    recs = load_model_and_recommend(
        "my_cur_model_1",
        user_ratings,
        top_n=10,
        min_score=1.0,
    )
    end = time.time()
    print("Рекомендации:")
    for t, s in recs:
        print(f" • {t}  (pred={s:.2f})")

    print(f'Время затраченное на CUR: {end-start}')