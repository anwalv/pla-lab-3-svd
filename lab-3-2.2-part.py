import pandas as pd
import numpy as np
from scipy.sparse.linalg import svds

file_path = 'C:/Users/User/PycharmProjects/lab-3-svd/ratings.csv'
df = pd.read_csv(file_path)

ratings_matrix = df.pivot(index='userId', columns='movieId', values='rating')

ratings_matrix = ratings_matrix.dropna(thresh=43, axis=0)
ratings_matrix = ratings_matrix.dropna(thresh=31, axis=1)
print(ratings_matrix)
ratings_matrix_filled = ratings_matrix.fillna(2.5)

R = ratings_matrix_filled.values
user_ratings_mean = np.mean(R, axis=1)
R_demeaned = R - user_ratings_mean.reshape(-1, 1)

U, sigma, Vt = svds(R_demeaned, k=3)
sigma = np.diag(sigma)

all_user_predicted_ratings = np.dot(np.dot(U, sigma), Vt) + user_ratings_mean.reshape(-1, 1)
preds_df = pd.DataFrame(all_user_predicted_ratings, columns=ratings_matrix.columns, index=ratings_matrix.index)

predicted_ratings_only = preds_df.copy()

for row in ratings_matrix.index:
    for column in ratings_matrix.columns:
        if not np.isnan(ratings_matrix.loc[row, column]):
            predicted_ratings_only.loc[row, column] = np.nan

print("\nPredicted Ratings Only Matrix:")
print(predicted_ratings_only)
movies_file_path = 'C:/Users/User/PycharmProjects/lab-3-svd/movies.csv'
movies_df = pd.read_csv(movies_file_path)

def top_10_movies(user_id):
    user_predicted_ratings = predicted_ratings_only.loc[user_id].sort_values(ascending=False)
    top_ten_movies = user_predicted_ratings.head(10)
    top_n_movies_info = movies_df[movies_df['movieId'].isin(top_ten_movies.index)][['title', 'genres']]
    return top_n_movies_info

user_id = 56
recomendation = top_10_movies(user_id)
print(f"Top 10 recommended movies for user {user_id}:")
print(recomendation)