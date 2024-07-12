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

    left_defined_matrix = np.dot(A, transpose_matrix)  # m x m
    right_defined_matrix = np.dot(transpose_matrix, A)  # n x n
    print("left defined matrix= ", left_defined_matrix)
    print("-----------------")
    print("right defined matrix= ", right_defined_matrix)
    print("-----------------")

    eigenvalues_l, eigenvectors_l = np.linalg.eig(left_defined_matrix)
    eigenvalues_r, eigenvectors_r = np.linalg.eig(right_defined_matrix)
    print("eigenvalues l= ", eigenvalues_l)
    print("eigenvectors l= ", eigenvectors_l)
    print("-----------------")
    
    print("eigenvalues r= ", eigenvalues_r)
    print("eigenvectors r= ", eigenvectors_r)
    print(eigenvalues_r, eigenvectors_r)
    idx_l = eigenvalues_l.argsort()[::-1]  
    eigenvalues_l = eigenvalues_l[idx_l]
    eigenvectors_l = eigenvectors_l[:, idx_l]
    
    idx_r = eigenvalues_r.argsort()[::-1] 
    eigenvalues_r = eigenvalues_r[idx_r]
    eigenvectors_r = eigenvectors_r[:, idx_r]

    if n >= m:
        singular_values = np.sqrt(eigenvalues_l)
        print("singular values l= ", singular_values)
    else:
        singular_values = np.sqrt(eigenvalues_r)
        print("singular values r= ", singular_values)

    normalized_eigenvectors_l = np.abs(eigenvectors_l) / np.linalg.norm(eigenvectors_l, axis=0)
    normalized_eigenvectors_r = np.abs(eigenvectors_r) / np.linalg.norm(eigenvectors_r, axis=0)
    print("normalized_eigenvectors_l= ", normalized_eigenvectors_l)
    print("normalized_eigenvectors_r= ", normalized_eigenvectors_r)
    U = normalized_eigenvectors_l
    V = normalized_eigenvectors_r
    M = np.zeros((m, n))
    M[:min(m, n), :min(m, n)] = np.diag(singular_values)
    print("m:", M)
    new_A = np.dot(U, np.dot(M, V.T))

    return U, M, V.T, new_A

A = np.array([[4, 2],
              [6, 3],
              [9, 4]])

U, M, V_T, new_A = svd(A)
print("A = ", A)
print("----------------------------------")
print("U =\n", U)
print("----------------------------------")
print("M =\n", M)
print("----------------------------------")
print("V tranposed =\n", V_T)
print("----------------------------------")
print("new A =\n", new_A)

