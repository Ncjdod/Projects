import numpy as np
import scipy.linalg as la


def lu_decomposition(original_matrix):
    n = len(original_matrix)
    lower_matrix = np.eye(n)
    upper_matrix = np.zeros((n, n))
    
    for k in range(n):
        upper_matrix[k, k:] = original_matrix[k, k:] - (lower_matrix[k, :k] @ upper_matrix[:k, k:])
        if k < n - 1: 
            lower_matrix[k+1:, k] = (1.0 / upper_matrix[k, k]) * (
                original_matrix[k+1:, k] - (lower_matrix[k+1:, :k] @ upper_matrix[:k, k])
            )
    
    return lower_matrix, upper_matrix 


A = np.array([[4, 3, -2],
              [-1, -1, 3],
              [2, -1, 5],
              ])
b = np.array([9, -4, 6])

L, U = lu_decomposition(A)

x = la.inv(A).dot(b) #1
y = la.solve(A, b) #2

a = la.solve(L, b) #3
b = la.solve(U, a)
