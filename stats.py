import numpy as np
from scipy.stats import f_oneway
import statsmodels.stats.multicomp as mc


def groups_ANOVA_Tukey(data, significance_ANOVA=0.05):
    """
    Data: list of 1D numpy arrays.
    """

    f_value, p_value = f_oneway(*data)
    print(f'ANOVA results: F = {f_value}, p = {p_value}')
    print('')
    if p_value < significance_ANOVA:
        data_flat = np.concatenate(data)
        groups = np.array([])
        for i in range(len(data)):
            groups = np.append(groups, [f'Group {i+1}'] * len(data[i]))
        tukey = mc.MultiComparison(data_flat, groups)
        result = tukey.tukeyhsd()
        print(result.summary())


def find_elbow_point(x, y):
    x = np.array(x)
    y = np.array(y)
    x_norm = (x - np.min(x)) / (np.max(x) - np.min(x))
    y_norm = (y - np.min(y)) / (np.max(y) - np.min(y))
    line_vec = np.array([x_norm[-1] - x_norm[0], y_norm[-1] - y_norm[0]])
    line_vec_norm = line_vec / np.sqrt(np.sum(line_vec**2))
    distances = np.empty(x_norm.shape)
    for i in range(len(x_norm)):
        point_vec = np.array([x_norm[i] - x_norm[0], y_norm[i] - y_norm[0]])
        proj_length = np.dot(point_vec, line_vec_norm)
        proj_vec = proj_length * line_vec_norm
        distances[i] = np.linalg.norm(point_vec - proj_vec)
    elbow_index = np.argmax(distances)
    elbow_point = (x[elbow_index], y[elbow_index])
    return elbow_point