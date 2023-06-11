from fastapi import FastAPI
from models import MovieRecommend, MovieRecommendRequest
import numpy as np
import pandas as pd
import pickle
from starlette import status


app = FastAPI()

movies_pivot = pd.read_csv(r'C:\Users\Sahil\fastapi\movie_rec_knn_fasta\movies_pivot.csv')
movies_pivot.set_index('title', drop = True, inplace = True)

model = pickle.load(open(r'C:\Users\Sahil\fastapi\movie_rec_knn_fasta\model_knn.pkl', 'rb'))

movie_names = movies_pivot.index

@app.get('/', status_code=status.HTTP_200_OK)
def index():
    return {'movies': 'euclidean_dist'}


@app.get('/all_movies', status_code=status.HTTP_200_OK)
def get_all_movies():
    d = {}
    for i in range(len(movie_names)):
        key = f'movie_{i}'
        d[key] = movie_names[i]
    return d



@app.post('/recommend', status_code=status.HTTP_201_CREATED)
def movies_recommend(data_request: MovieRecommendRequest):
    data = MovieRecommend(**data_request.dict())
    movie_name = data.movie_name
    number = data.num_of_movies_to_be_recommended
   
    dist, ind = model.kneighbors(movies_pivot.loc[movie_name].values.reshape(1, -1), n_neighbors = number, return_distance = True)
    movies = []
    eucliean_distance = []

    for a,b in zip(dist.flatten(), ind.flatten()):
        movies.append(movies_pivot.index[b])
        eucliean_distance.append(round(a,2))

    res = {movies[i]: eucliean_distance[i] for i in range(len(movies))}
    return  res


