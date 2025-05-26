
"""svd_numpy_recommender.py
Рекомендательная система на чистом NumPy/Pandas без Surprise.

Входной CSV (`MATRIX_CSV`) — широкая матрица:
    • Заголовки столбцов — названия фильмов.
    • Первый столбец (index) — идентификатор пользователя.
    • Значения 1–5 — рейтинг, 0 или пусто — фильм не просмотрен.

Алгоритм:
    1. Загружаем матрицу рейтингов.
    2. Вычитаем глобальное среднее, заполняем пропуски нулями.
    3. Строим усечённое SVD (NumPy).
    4. Для нового пользователя решаем u = r V Σ^{-1} (линейная регрессия).
    5. Предсказываем оценки и выводим top‑N фильмов, которых пользователь ещё не видел.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple

# --- Конфигурация -----------------------------------------------------------

MATRIX_CSV = 'UI_data_3.csv'   # широкая матрица пользователей × фильмов
N_FACTORS   = 50                    # число латентных факторов
TOP_N       = 10                    # сколько фильмов рекомендовать

# Оценки нового пользователя:
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

# ---------------------------------------------------------------------------


def load_matrix(path: str) -> pd.DataFrame:
    """Читает CSV → DataFrame (index = userId)."""
    df = pd.read_csv(path, index_col=0).astype(np.float64)
    df.index.name = 'userId'
    return df


def svd_decompose(ratings: pd.DataFrame, k: int):
    """Строит усечённое SVD матрицы с вычитанием глобального среднего."""
    mask = ratings != 0
    global_mean = ratings[ratings >0].stack().mean()

    # центрируем наблюдаемые рейтинги
    centered = ratings.astype(float).copy()
    centered[mask] = centered[mask] - global_mean
    centered = centered.fillna(0.0)  # если были NaN

    A = centered.values
    # вычисляем SVD
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    U_k = U[:, :k]
    S_k = np.diag(s[:k])
    Vt_k = Vt[:k, :]
    print(f"[SVD] Глобальное среднее = {global_mean:.3f}, сохранены {k} факторов.")
    return global_mean, U_k, S_k, Vt_k


def latent_vector_for_user(r_vector: np.ndarray,
                           Vt_k: np.ndarray,
                           S_k: np.ndarray,
                           global_mean: float) -> np.ndarray:
    """Оценивает латентный вектор пользователя u = r V Σ^{-1}."""
    r_centered = r_vector.copy()
    rated_mask = r_centered != 0
    r_centered[rated_mask] = r_centered[rated_mask] - global_mean

    with np.errstate(divide='ignore', invalid='ignore'):
        s_vals = np.diag(S_k)
        s_inv_vals = np.where(s_vals > 1e-8, 1.0 / s_vals, 0.0)
        S_inv = np.diag(s_inv_vals)

    u_vec = r_centered @ Vt_k.T @ S_inv
    return u_vec


def predict_ratings(u_vec: np.ndarray,
                    S_k: np.ndarray,
                    Vt_k: np.ndarray,
                    global_mean: float) -> np.ndarray:
    """Возвращает предсказанный вектор оценок."""
    centered_pred = u_vec @ S_k @ Vt_k
    return centered_pred + global_mean


def build_user_vector(columns: List[str],
                      user_ratings: Dict[str, int]) -> np.ndarray:
    """Формирует numpy‑вектор длиной = #фильмов для нового пользователя."""
    vec = np.zeros(len(columns), dtype=float)
    for title, score in user_ratings.items():
        if title in columns:
            idx = columns.index(title)
            vec[idx] = score
        else:
            print(f"[WARN] Фильм '{title}' отсутствует в матрице и будет пропущен.")
    return vec


def top_n_recommendations(pred_vector: np.ndarray,
                          columns: List[str],
                          known_titles: List[str],
                          n: int = 10) -> List[Tuple[str, float]]:
    """Возвращает n лучших (title, score) не просмотренных фильмов."""
    known = set(known_titles)
    candidates = [
        (title, score) for title, score in zip(columns, pred_vector)
        if title not in known
    ]
    candidates.sort(key=lambda x: x[1], reverse=True)
    return candidates[:n]


def main():
    print('[+] Чтение матрицы рейтингов…')
    ratings_df = load_matrix(MATRIX_CSV)

    start = time.time()
    print('[+] SVD‑разложение…')
    global_mean, U_k, S_k, Vt_k = svd_decompose(ratings_df, k=N_FACTORS)

    print('[+] Формируем вектор нового пользователя…')
    titles = ratings_df.columns.tolist()
    r_vec = build_user_vector(titles, USER_RATINGS)

    print('[+] Оцениваем латентный вектор…')
    u_vec = latent_vector_for_user(r_vec, Vt_k, S_k, global_mean)
    end = time.time()

    print('[+] Предсказываем оценки…')
    pred = predict_ratings(u_vec, S_k, Vt_k, global_mean)

    print('[+] Top‑{} рекомендаций:'.format(TOP_N))
    recs = top_n_recommendations(pred, titles, list(USER_RATINGS.keys()), n=TOP_N)
    for title, score in recs:
        print(f"{title:60} → {score:.2f}")

    print(f'Затраченное время на SVD: {end-start} сек.')


import time

if __name__ == '__main__':
    main()
