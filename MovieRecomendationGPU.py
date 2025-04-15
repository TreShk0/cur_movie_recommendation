import torch
import numpy as np
import pandas as pd
import difflib
import os
import pickle

############################################
# Вспомогательные функции из вашего кода 
# (extract_sparse_columns, maxvol_gpu, maxvol_sparse_gpu и т.д.)
############################################

def extract_sparse_columns(A_sparse, col_indices, num_rows, r):
    orig_indices = A_sparse._indices()  # [2, nnz]
    orig_values = A_sparse._values()
    
    # Фильтруем ненулевые элементы, принадлежащие выбранным столбцам
    mask = torch.zeros(orig_indices.shape[1], dtype=torch.bool, device=A_sparse.device)
    for ci in col_indices.cpu():
        mask |= (orig_indices[1] == ci.item())
    
    new_indices = orig_indices[:, mask].clone()
    new_values = orig_values[mask].clone()
    
    # Создаём маппинг: оригинальный номер столбца -> новый номер (0...r-1)
    mapping = -torch.ones(A_sparse.shape[1], dtype=torch.long, device=A_sparse.device)
    mapping[col_indices] = torch.arange(r, device=A_sparse.device)
    
    new_indices[1] = mapping[new_indices[1]]
    
    A_cols = torch.sparse_coo_tensor(new_indices, new_values, size=(num_rows, r)).coalesce()
    return A_cols

def maxvol_gpu(A, e=1.05, k=100):
    n, r = A.shape
    if n <= r:
        raise ValueError("Матрица A должна быть 'tall': n > r")
    
    LU, pivots = torch.lu(A)
    P, L, U_mat = torch.lu_unpack(LU, pivots)
    # Инициализация
    I = torch.argmax(P[:, :r], dim=0).clone()  # (r,)
    
    # Решаем U^T x = A^T
    Q = torch.linalg.solve_triangular(U_mat.T, A.T, upper=False, left=True, unitriangular=False)
    # Решаем (L[:r, :])^T y = Q
    Y = torch.linalg.solve_triangular(L[:r, :].T, Q, upper=True, left=True, unitriangular=True)
    B = Y.T  # тогда A = B @ A[I, :]
    
    for _iter in range(k):
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
    # 1) L2-нормы столбцов с учётом разреженности
    indices = A_sparse._indices()  # [2, nnz]
    values = A_sparse._values()
    col_norms = torch.zeros(m, device=A_sparse.device, dtype=A_sparse.dtype)
    col_norms = col_norms.scatter_add(0, indices[1], values**2)
    col_norms = torch.sqrt(col_norms)
    
    # 2) Выбираем r столбцов с наибольшей нормой
    _, J = torch.topk(col_norms, r)
    J, _ = torch.sort(J)
    
    # Извлекаем подматрицу (n x r) в dense
    A_cols_sparse = extract_sparse_columns(A_sparse, J, n, r)
    A_cols = A_cols_sparse.to_dense()
    
    # 3) Применяем точный maxvol для "tall" (n > r)
    I, _ = maxvol_gpu(A_cols, e=e, k=k)
    
    submatrix = A_cols[I, :]  # (r x r)
    return submatrix, J, I


###################################################
# Класс для обучения и сохранения модели
###################################################
class MaxvolCURModel:
    """
    Класс, который обучается на матрице (пользователи x фильмы) из CSV,
    вычисляет CUR, а также хранит маппинг "название фильма" -> "ID столбца".
    """
    def __init__(self, r=None, device="cuda", e=1.05, k=100):
        """
        r: размер искомой подматрицы (r x r). Если None, возьмём m//2.
        device: "cuda" или "cpu"
        e, k : параметры для maxvol
        """
        self.r = r
        self.device = device
        self.e = e
        self.k = k
        
        # После fit():
        self.UR_ = None     # (r x m) numpy
        self.cols_ = None   # (r, ) torch.Tensor
        self.rows_ = None   # (r, ) torch.Tensor
        self.n_ = None
        self.m_ = None
        
        # Маппинг названия фильма -> ID столбца
        self.film_to_id_ = {}
        # Обратный словарь (ID -> название) для удобства
        self.id_to_film_ = {}
        
    def fit(self, ratings_csv, film_titles_csv=None, film_col_name="title", save_dir="model_data"):
        """
        Обучение модели:
         1) Читаем ratings_csv: в каждой строке - пользователь, в каждом столбце - рейтинг фильма.
            Если есть столбец 'userId', убираем его.
         2) Если есть film_titles_csv (например, {id, title}), подгружаем названия фильмов и связываем
            их с порядковым номером столбца.
            - Предполагаем, что там есть столбцы ['id', 'title'] (или любой другой film_col_name).
         3) Выполняем sparse + maxvol, получаем C, U, R, считаем UR = U*R.
         4) Сохраняем UR, cols_, rows_, film_to_id_ на диск.
        """
        # 1) Читаем CSV с рейтингами
        df = pd.read_csv(ratings_csv)
        if 'userId' in df.columns:
            df.drop(columns=['userId'], inplace=True)
        values = df.values.astype(np.int8)  # (n, m)
        self.n_, self.m_ = values.shape
        
        # 2) Если нужно, читаем таблицу с (id, title)
        #    Предполагаем, что ID фильма совпадает с индексом столбца:
        #    - Если в film_titles_csv хранится информация вида:
        #         filmId, title
        #      и filmId = 0..m-1, мы читаем и складываем в словари.
        if film_titles_csv is not None:
            titles_df = pd.read_csv(film_titles_csv)
            # Ожидаем, что в titles_df['id'] лежит столбец с 0..m-1
            # и titles_df[film_col_name] - название.
            for row in titles_df.itertuples(index=False):
                fid = getattr(row, 'id')
                ftitle = getattr(row, film_col_name)
                self.film_to_id_[ftitle] = fid
                self.id_to_film_[fid] = ftitle
        else:
            # Если у нас нет внешнего csv, то пусть просто title = str колонки
            # Т.е. film_i -> "film i"
            for col_id in range(self.m_):
                title_str = f"Film {col_id}"
                self.film_to_id_[title_str] = col_id
                self.id_to_film_[col_id] = title_str
        
        # 3) Готовим sparse-модель
        if self.r is None:
            self.r = self.m_ // 2
        
        rows_np, cols_np = np.nonzero(values)
        vals_np = values[rows_np, cols_np]
        
        # Тензор на GPU
        indices_torch = torch.tensor([rows_np, cols_np], dtype=torch.long, device=self.device)
        values_torch = torch.tensor(vals_np, dtype=torch.float32, device=self.device)
        A_sparse = torch.sparse_coo_tensor(indices_torch, values_torch, size=(self.n_, self.m_)).coalesce()
        
        # maxvol
        submatrix, cols, rows = maxvol_sparse_gpu(A_sparse, self.r, e=self.e, k=self.k)
        self.cols_ = cols  # (r,)
        self.rows_ = rows  # (r,)
        
        # Формируем матрицы C, R
        C = values[:, self.cols_.cpu().numpy()]  # (n x r)
        R = values[self.rows_.cpu().numpy(), :]  # (r x m)
        
        # U = inv(submatrix), т.к. submatrix = A_cols[I, :]
        submatrix_cpu = submatrix.cpu().numpy()
        U = np.linalg.inv(submatrix_cpu)  # (r x r)
        
        # UR
        UR = U @ R  # (r x m)
        self.UR_ = UR  # numpy
        
        # 4) Сохраняем всё на диск
        os.makedirs(save_dir, exist_ok=True)
        np.save(os.path.join(save_dir, "UR.npy"), UR)
        np.save(os.path.join(save_dir, "cols.npy"), self.cols_.cpu().numpy())
        np.save(os.path.join(save_dir, "rows.npy"), self.rows_.cpu().numpy())
        metadata = {
            "n": self.n_,
            "m": self.m_,
            "r": self.r,
            "film_to_id_": self.film_to_id_,
            "id_to_film_": self.id_to_film_
        }
        with open(os.path.join(save_dir, "metadata.pkl"), "wb") as f:
            pickle.dump(metadata, f)
        
        print(f"Модель обучена. UR shape={UR.shape}, сохранено в папку: {save_dir}")

###################################################
# 2. Функция predict, где мы загружаем модель с диска
#    и формируем рекомендации
###################################################
def load_model_and_recommend(
    model_dir, 
    user_ratings_title_based, 
    top_n=5, 
    min_score=1.0
):
    """
    Аргументы:
      model_dir (str): путь к папке, где лежат UR.npy, cols.npy, rows.npy, metadata.pkl
      user_ratings_title_based (dict): { "название фильма": рейтинг }, рейтинг от 1..5
      top_n (int): сколько рекомендаций хотим
      min_score (float): минимальный балл, чтобы рекомендация считалась приемлемой
    
    Возвращает: список (film_title, pred_score) top_n
    """
    import difflib
    
    # 1) Загружаем матрицу UR, cols, метаданные
    UR = np.load(os.path.join(model_dir, "UR.npy"))  # (r x m)
    cols = np.load(os.path.join(model_dir, "cols.npy"))  # (r,)
    with open(os.path.join(model_dir, "metadata.pkl"), "rb") as f:
        metadata = pickle.load(f)
    
    film_to_id_ = metadata["film_to_id_"]
    id_to_film_ = metadata["id_to_film_"]
    m = metadata["m"]
    r = metadata["r"]
    
    # 2) Формируем вектор c (1 x r), где c_j = рейтинг пользователя,
    #    если j-й столбец выбран, иначе 0.
    c = np.zeros(r, dtype=np.float32)
    
    # 2.1) Для каждого "названия фильма", которое ввёл пользователь,
    #      найдём ближайшее совпадение в film_to_id_:
    all_titles = list(film_to_id_.keys())
    
    for title_input, rating in user_ratings_title_based.items():
        # Ищем самое близкое по названию
        matches = difflib.get_close_matches(title_input, all_titles, n=1, cutoff=0.0)
        if not matches:
            print(f"Не нашли похожих фильмов для '{title_input}' — пропускаем.")
            continue
        
        best_title = matches[0]
        film_id = film_to_id_[best_title]
        # Теперь проверяем, попал ли film_id в cols (выбранные столбцы):
        pos_arr = np.where(cols == film_id)[0]
        if len(pos_arr) > 0:
            pos = pos_arr[0]
            c[pos] = rating
        else:
            # Фильм не в cols, тогда просто пропускаем (его не учтём в векторе c)
            pass
    
    # 3) Умножаем c (1 x r) на UR (r x m) => pred (1 x m)
    pred = c @ UR  # shape (m,)
    
    # 4) Уберём из рекомендаций фильмы, которые пользователь уже оценил (снова по названиям)
    #    то есть поставим им -9999
    for title_input in user_ratings_title_based.keys():
        matches = difflib.get_close_matches(title_input, all_titles, n=1, cutoff=0.0)
        if matches:
            best_title = matches[0]
            film_id = film_to_id_[best_title]
            if 0 <= film_id < m:
                pred[film_id] = -9999
    
    # 5) Выбираем top_n
    indices_sorted = np.argsort(pred)[::-1]
    results = []
    for idx in indices_sorted:
        score = pred[idx]
        if score < min_score:
            # Если хотим отсечь слабые рекомендации:
            break
        if len(results) >= top_n:
            break
        # формируем (title, score)
        film_title = id_to_film_.get(idx, f"Film {idx}")
        results.append((film_title, float(score)))
    
    return results


###################################################
# Пример использования всего этого
###################################################
if __name__ == "__main__":
    # 1) Обучаем модель (делается один раз)
    model = MaxvolCURModel(r=None, device="cuda", e=1.05, k=100)
    model.fit(
        ratings_csv="UI_data_2.csv",         # Ваш файл с рейтингами
        film_titles_csv=None,               # Или csv с (id, title)
        film_col_name="title", 
        save_dir="my_cur_model"
    )
    
    # 2) В реальном приложении, когда хотим выдать рекомендации:
    user_ratings_title_based = {
        "film 10": 5,   # Пользователь ввёл "film 10" - ищем ближайшее название
        "film 25": 3,
    }
    recommended = load_model_and_recommend(
        model_dir="my_cur_model",
        user_ratings_title_based=user_ratings_title_based,
        top_n=5,
        min_score=1.0
    )
    
    print("Рекомендации:")
    for film_title, score in recommended:
        print(f" - {film_title} (прогноз={score:.2f})")
