import numpy as np
import math
from numpy. linalg import norm
import pandas as pd

def ED_distance(ts1, ts2):
    """
    Calculate the Euclidean distance.

    Parameters
    ----------
    ts1 : numpy.ndarray
        The first time series.

    ts2 : numpy.ndarray
        The second time series.

    Returns
    -------
    ed_dist : float
        Euclidean distance between ts1 and ts2.
    """

    ED = []

    for i in range(len(ts1)):
        ED.append((ts1[i] - ts2[i]) ** 2)

    return math.sqrt(sum(ED))


def norm_ED_distance(ts1, ts2):
    """
    Calculate the normalized Euclidean distance.

    Parameters
    ----------
    ts1 : numpy.ndarray
        The first time series.

    ts2 : numpy.ndarray
        The second time series.

    Returns
    -------
    norm_ed_dist : float
        The normalized Euclidean distance between ts1 and ts2.
    """

    norm_ed_dist = 0

    # INSERT YOUR CODE 

    return norm_ed_dist


def DTW_distance(ts1, ts2, r=None):
    """
    Calculate DTW distance.

    Parameters
    ----------
    ts1 : numpy.ndarray
        The first time series.

    ts2 : numpy.ndarray
        The second time series.

    r : float
        Warping window size.
    
    Returns
    -------
    dtw_dist : float
        DTW distance between ts1 and ts2.
    """

    n = len(ts1)
    m = len(ts2)

    # Создаем матрицу для хранения вычисленных расстояний
    dp = np.zeros((n, m))

    # Заполняем первую строку и первый столбец матрицы
    for i in range(n):
        for j in range(m):
            d = abs(ts1[i] - ts2[j]) ** 2
            if i == 0 and j == 0:
                dp[i, j] = d
            elif i == 0:
                dp[i, j] = d + dp[i, j - 1]
            elif j == 0:
                dp[i, j] = d + dp[i - 1, j]
            else:
                dp[i, j] = d + min(dp[i - 1, j], dp[i, j - 1], dp[i - 1, j - 1])

    # Возвращаем значение DTW расстояния между последними элементами
    return dp[n - 1, m - 1]