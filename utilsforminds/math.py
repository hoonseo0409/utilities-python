import numpy as np

def p_exponent_matrix(arr, p):
    """
    
    Examples
    --------
    self.diagonals['D_7'] = (self.p / 2) * helpers.p_exponent_matrix(W_p1p @ W_p1p.transpose() + self.delta * np.eye(self.d), (self.p / 2 - 1))
    """
    assert(len(arr.shape) == 2)

    #%% Using SVD
    u, s, v = np.linalg.svd(arr, full_matrices=False)
    return u @ np.diag(s ** p) @ v

    #$$ Using Igenvalue decomposition
    # w, v = np.linalg.eig(arr)
    # return v @ np.diag(w ** p) @ v.transpose()

def get_norm_from_matrix(arr, under_p_1, under_p_2):
    """
    
    Examples
    --------
    testArr_1 = np.array(list(range(6))).reshape(2,3)\n
    print(helpers.get_norm_from_matrix(testArr_1, 2, 2))
        => 7.416198487095663
    """

    summed = 0
    for i in range(arr.shape[0]):
        summed += np.sum(arr[i, :] ** under_p_1) ** (under_p_2 / under_p_1)
    return summed ** (1 / under_p_2)