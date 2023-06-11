from flask import Flask, render_template, redirect, session
from flask_wtf import FlaskForm
from wtforms import FloatField, SubmitField, SelectField, IntegerField, StringField
from wtforms.validators import DataRequired
import numpy as np
import pandas as pd
import pickle

app = Flask(__name__)
app.config['SECRET_KEY'] = 'my_secret_key'

movies_pivot = pd.read_csv(r'C:\Users\Sahil\Flask_bootcamp\flask_tut\Movie_rec_KNN\movies_pivot.csv')
movies_pivot.set_index('title', drop = True, inplace = True)
movie_names = movies_pivot.index

model = pickle.load(open(r'C:\Users\Sahil\.spyder-py3\model_knn.pkl', 'rb'))

class MovieRecommender(FlaskForm):
    movie_name = SelectField('Select the name of movie', choices= movie_names)
    number = IntegerField('Select the number of movies to be recommended', validators= [DataRequired()])
    submit = SubmitField('Recommend')

def recommend_movies(movie_name, num_movies_to_be_recommended):
    dist, ind = model.kneighbors(movies_pivot.loc[movie_name].values.reshape(1, -1), n_neighbors = num_movies_to_be_recommended, return_distance = True)
    movies = []
    eucliean_distance = []

    for a,b in zip(dist.flatten(), ind.flatten()):
        movies.append(movies_pivot.index[b])
        eucliean_distance.append(round(a,2))

    res = {movies[i]: eucliean_distance[i] for i in range(len(movies))}
    return  res

#home page
@app.route('/', methods = ['GET', 'POST'])
def index():
    form = MovieRecommender()

    if form.validate_on_submit():
        session['movie_name'] = form.movie_name.data
        session['number'] = form.number.data
        return  redirect('result')
    
    return render_template('home.html', form = form)

#result page
@app.route('/result', methods = ['GET', 'POST'])
def result():
    res = recommend_movies(session['movie_name'], session['number'])
    return render_template('result.html', res = res, movie_name = session['movie_name'])

if __name__ == '__main__':
    app.run()


