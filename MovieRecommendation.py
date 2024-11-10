import numpy as np
import pandas as pd
from scipy.linalg import pinv
from scipy.linalg import lu
from scipy.linalg import solve_triangular

class MovieRecommendation:
    def __init__(self, rank=0, test=False, n_recommendations=5):
        """
        Инициализация класса MovieRecommendation.
        
        Параметры:
        rank (int): Ранг разложения CUR.
        test (bool): Параметр для тестового режима (по умолчанию False).
        n_recommendations (int): Количество рекомендованных фильмов.
        """
        self.rank = rank
        self.test = test
        self.n_recommendations = n_recommendations
        self.C = None
        self.U = None
        self.R = None
        self.movies = None  # Для хранения названий фильмов
        self.users = None   # Для хранения пользователей
        self.unseen = None
        self.matrix = None

    def maxvol(self, A, e=1.05, k=100):
        n, r = A.shape
        if n <= r:
            raise ValueError('Input matrix should be "tall"')

        P, L, U = lu(A, check_finite=False)
        I = P[:, :r].argmax(axis=0)

        if not self.test:
            Q = solve_triangular(U, A.T, trans=1, check_finite=False)
            B = solve_triangular(L[:r, :], Q, trans=1, check_finite=False, unit_diagonal=True, lower=True).T
        else:
            Q = np.dot(np.linalg.pinv(U), A.T)
            B = np.dot(np.linalg.pinv(L[:r, :]), Q).T

        for _ in range(k):
            i, j = np.divmod(np.abs(B).argmax(), r)
            if np.abs(B[i, j]) <= e:
                break

            I[j] = i
            bj = B[:, j]
            bi = B[i, :].copy()
            bi[j] -= 1.
            B -= np.outer(bj, bi / B[i, j])

        return I, B

    def maxsumcolumns(self, A, r):
        columns_sums = A.sum(axis=0)
        columns_index = np.argsort(columns_sums)[-r:][::-1]
        return columns_index

    def cur_decomposition(self, A):
        if self.rank == 0:
            self.rank = np.linalg.matrix_rank(A)

        I, B = self.maxvol(A)
        I = I[:self.rank]
        self.R = A[I, :]

        column_indices = self.maxsumcolumns(A, self.rank)
        self.C = A[:, column_indices]

        W = A[np.ix_(I, column_indices)]
        self.U = pinv(W)

    def fit(self, UI_matrix):
        """
        Обучает рекомендательную систему с использованием CUR-разложения.
        
        Параметры:
        UI_matrix (pd.DataFrame): Матрица оценок пользователей и фильмов.
        """
        self.users = UI_matrix.index.values
        self.movies = UI_matrix.columns.values
        self.matrix = UI_matrix.values
        self.cur_decomposition(self.matrix)

    def recommend(self, user):
        """
        Рекомендует фильмы для указанного пользователя.
        
        Параметры:
        user (int): Индекс пользователя.
        
        Возвращает:
        рекомендации (list): Список индексов рекомендованных фильмов.
        """
        self.unseen = np.where(self.matrix[user] == 0)[0]
        unseen_movie_scores = (self.C@self.U@self.R)[user, self.unseen]
        recommendations = self.unseen[np.argsort(unseen_movie_scores)[::-1][:self.n_recommendations]]

        return self.movies[recommendations].tolist()