from posixpath import split
import numpy as np
import pandas as pd
import ast
import nltk
pd.options.mode.chained_assignment = None  # default='warn'
movies = pd.read_csv('C:/Users/CW/OneDrive/Desktop/Movie Recommendation System/tmdb_5000_movies.csv')
credits = pd.read_csv('C:/Users/CW/OneDrive/Desktop/Movie Recommendation System/tmdb_5000_credits.csv')
#instead of dealing with two seperate databases we are meging them
movies = movies.merge(credits, on='title')
#taking only 6 columns from the available dataset which will help us tagging
movies = movies[['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew']]
movies.dropna(inplace = True)
#print(movies.isnull().sum())
#print(movies.duplicated().sum())
#print(movies.iloc[0].genres)
def convert(obj):
    L = []
    for i in ast.literal_eval(obj):
        L.append(i['name'])
    return L

#converting genres and keyword to a more better format having only names
movies['genres'] = movies['genres'].apply(convert)
movies['keywords'] = movies['keywords'].apply(convert)

#function for fetching just the top 3 crew members
def convert3(obj):
    L = []
    counter = 0
    for i in ast.literal_eval(obj):
        if counter != 3 :
            L.append(i['name'])
            counter += 1
        else :
            break
    return L

#fetching just top 3 crew members
movies['cast'] =  movies['cast'].apply(convert3)
#print( movies['cast'] )
#function for fetching just the director name from the whole crew
def fetch_director(obj):
    L = []
    for i in ast.literal_eval(obj):
        if i['job'] == 'Director':
            L.append(i['name'])
            break
    return L
#fetching just the director name from the whole crew
movies['crew'] = movies['crew'].apply(fetch_director)
#changing the overview from string to list
movies['overview'] = movies['overview'].apply(lambda x: x.split())
#removing spaces
movies['genres'] = movies['genres'].apply(lambda x: [i.replace(" ","") for i in x])
movies['keywords'] = movies['keywords'].apply(lambda x: [i.replace(" ","") for i in x])
movies['cast'] = movies['cast'].apply(lambda x: [i.replace(" ","") for i in x])
movies['crew'] = movies['crew'].apply(lambda x: [i.replace(" ","") for i in x])

#forming a new column in dataset of movie by the name tag which have the values of overview, genres, cast, keywords and crew all in one column
movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']
#print(movies.head())

#using building class for stemming
#stemming basically converts [lover, loving, loved] to [love, love, love]
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()

#helper function for stemming
def stem(text):
    y = []
    
    for i in text.split():
        y.append(ps.stem(i))
    
    return " ".join(y)

#now that we have a tag column we don't need seperate columns of genres, overview etc as it is just the duplicate record
#So we are making a new dataframe which just have the id, title and the tags
new_df = movies[['movie_id', 'title', 'tags']]
#converting the tag values from a list to a string
new_df['tags'] = new_df['tags'].apply(lambda x: " ".join(x))
#converting the tag data into lower format
new_df['tags'] = new_df['tags'].apply(lambda x: x.lower() if type(x) is str else 'empty')

new_df['tags'] = new_df['tags'].apply(stem)
#print(new_df)

#build in class for vectorizer
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 5000, stop_words = 'english')

vectors = cv.fit_transform(new_df['tags']).toarray()

from sklearn.metrics.pairwise import cosine_similarity
similarity = cosine_similarity(vectors)

def recommend(movie):
    movie_index = new_df[new_df['title'] == movie].index[0]
    distances = similarity[movie_index]
    movie_list = sorted(list(enumerate(distances)), reverse = True, key=lambda x: x[1])[1:6]
    for i in movie_list:
        print(new_df.iloc[i[0]].title)

print(recommend('Avatar'))


import pickle
pickle.dump(new_df.to_dict(),open('movie_dict.pkl','wb'))
pickle.dump(similarity, open('similarity.pkl', 'wb'))