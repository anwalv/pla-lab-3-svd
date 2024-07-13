import numpy as np
#Алгоритм:
#1. Знайти правовизначену і лівовизначену матриці(A^T*A, A*A^T) 
#2. Знайти власні значення і корінь з них(сингулярні числа)/ M =сингулярні числа по діаг.
#3. Знайти власні векроти ліво(u) і правовизначених(v) матриць. Пронормувати.
#4. Посортувати в порядку спадання
#5. ф*v1 = A^T*u1
#6. A = U*M*V^T 
import numpy as np

def svd(A):
    m, n = A.shape
    print("A= ", A)
    print("-----------------")
    print("m= ", m)
    print("n= ", n)
    print("-----------------")
    transpose_matrix = np.transpose(A)
    U = np.zeros((m, m))
    V = np.zeros((n, n))

    if m>=n:
        our_matrix = np.dot(transpose_matrix, A)  # n x n
        print("right defined matrix= ", our_matrix)
        print("-----------------")
    elif m<n:
        our_matrix = np.dot(A, transpose_matrix)  # m x m
        print("left defined matrix= ", our_matrix)
        print("-----------------")

    eigenvalues, eigenvectors = np.linalg.eig(our_matrix)
    print("eigenvalues = ", eigenvalues)
    print("eigenvectors = ", eigenvectors)
    print("-----------------")
    idx_l = eigenvalues.argsort()[::-1]  
    eigenvalues = eigenvalues[idx_l]
    eigenvectors = eigenvectors[:, idx_l]

    singular_values = np.sqrt(eigenvalues)
    print("singular values = ", singular_values)
    M = np.zeros((m, n))
    M[:min(m, n), :min(m, n)] = np.diag(singular_values)
    print("m:", M)

    normalized_eigenvectors= eigenvectors/ np.linalg.norm(eigenvectors, axis=0)
    print("normalized_eigenvectors = ", normalized_eigenvectors)
    if m < n:
        U = normalized_eigenvectors
        for i in range(len(singular_values)):
            u_i = U[:, i]
            sigma_i = singular_values[i]
            A_T_u_i = np.dot(A.T, u_i)
            v_i = A_T_u_i / sigma_i
            V[:, i] = v_i

    elif m >= n:
        V = normalized_eigenvectors
        for i in range(len(singular_values)):
            v_i = V[:, i]
            sigma_i = singular_values[i]
            A_v_i = np.dot(A, v_i)
            u_i = A_v_i / sigma_i
            U[:, i] = u_i

    new_A = np.dot(U, np.dot(M, V))

    return U, M, V, new_A

A = np.array([[3, 2],
              [1, 9],
              [4, 1]])

U, M, V, new_A = svd(A)
print("A = ", A)
print("----------------------------------")
print("U =\n", U)
print("----------------------------------")
print("M =\n", M)
print("----------------------------------")
print("V tranposed =\n", V)
print("----------------------------------")
print("new A =\n", new_A)
