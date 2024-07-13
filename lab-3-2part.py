import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.sparse.linalg import svds
file_path = 'C:/Users/User/PycharmProjects/lab-3-svd/ratings.csv'
df = pd.read_csv(file_path)
ratings_matrix = df.pivot(index='userId', columns='movieId', values='rating')

ratings_matrix = ratings_matrix.dropna(thresh=160, axis=0)
ratings_matrix = ratings_matrix.dropna(thresh=90, axis=1)

mean_rating = df['rating'].mean()
ratings_matrix = ratings_matrix.fillna(mean_rating)

ratings_matrix_filled = ratings_matrix.fillna(2.5)
R = ratings_matrix_filled.values
user_ratings_mean = np.mean(R, axis=1)
R_demeaned = R - user_ratings_mean.reshape(-1, 1)

U, sigma, Vt = svds(R_demeaned, k=3)

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.set_title('Visualization U-matrix')
colors = np.linspace(0, 1, len(U))
ax.scatter(U[:, 0], U[:, 1], U[:, 2], c=colors, cmap='viridis', marker='o')
ax.legend()
plt.show()

print(U)
print("------------------------------")
print(Vt.T)

num_movies_to_visualize = 10

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

x = Vt[0, :num_movies_to_visualize]
y = Vt[1, :num_movies_to_visualize]
z = Vt[2, :num_movies_to_visualize]

colors = np.linspace(0, 1, num_movies_to_visualize)
ax.scatter(x, y, z, c=colors, cmap='viridis', marker='o')
ax.set_title('Visualization V-matrix ')
plt.show()