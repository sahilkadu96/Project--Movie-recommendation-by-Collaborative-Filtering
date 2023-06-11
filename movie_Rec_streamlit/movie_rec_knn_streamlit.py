import streamlit as st
import numpy as np
import pandas as pd
import pickle

st.title('Movie recommendation by KNN')
st.header('Movie recommendation')
st.subheader('Closer the Euclidean distance better the recommendation!')

movies_pivot = pd.read_csv(r'C:\Users\Sahil\.spyder-py3\movies_pivot.csv')
movies_pivot.set_index('title', drop = True, inplace = True)

model = pickle.load(open(r'C:\Users\Sahil\.spyder-py3\model_knn.pkl', 'rb'))

movie_names = movies_pivot.index

movie_name = st.selectbox('Enter the movie name you want recommendation for', movie_names )
num_movies_to_be_recommended = st.number_input('Enter the number of movies to be recommended', min_value=0, max_value=10)

def recommend_movies(movie_name, num_movies_to_be_recommended):
    dist, ind = model.kneighbors(movies_pivot.loc[movie_name].values.reshape(1, -1), n_neighbors = num_movies_to_be_recommended, return_distance = True)
    movies = []
    eucliean_distance = []
    for a,b in zip(dist.flatten(), ind.flatten()):
        movies.append(movies_pivot.index[b])
        eucliean_distance.append(round(a,2))
    df = pd.DataFrame({'Movies_recommended':movies, 'Euclidean_distance':eucliean_distance})
    st.write(df)
        #st.write(f'Recommended movie is {movies_pivot.index[b]}, euclidean dist is {round(a,2)}')

if st.button('Recommend'):
    recommend_movies(movie_name, num_movies_to_be_recommended)