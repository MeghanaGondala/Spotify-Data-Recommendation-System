import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
import streamlit as st

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.manifold import TSNE

import warnings
warnings.filterwarnings('ignore')

# Load the data
df = pd.read_parquet(r"C:\Users\tejas\Downloads\0000 (1).parquet")

# Remove rows with missing values
df.dropna(inplace=True)

# Sort the DataFrame by popularity and keep only the top 10000 most popular songs
df = df.sort_values(by=['popularity'], ascending=False).head(10000)

# Initialize CountVectorizer
song_vectorizer = CountVectorizer()
song_vectorizer.fit(df['track_genre'])

# Function to calculate similarities
def get_similarities(song_name, data):
    text_array1 = song_vectorizer.transform(data[data['track_name']==song_name]['track_genre']).toarray()
    num_array1 = data[data['track_name']==song_name].select_dtypes(include=np.number).to_numpy()

    sim = []
    for idx, row in data.iterrows():
        name = row['track_name']
        text_array2 = song_vectorizer.transform(data[data['track_name']==name]['track_genre']).toarray()
        num_array2 = data[data['track_name']==name].select_dtypes(include=np.number).to_numpy()

        text_sim = cosine_similarity(text_array1, text_array2)[0][0]
        num_sim = cosine_similarity(num_array1, num_array2)[0][0]
        sim.append(text_sim + num_sim)
    
    return sim

# Function to recommend songs
def recommend_songs(song_name, data=df):
    if df[df['track_name'] == song_name].shape[0] == 0:
        st.write('This song is either not so popular or you have entered an invalid name.')
        st.write('Some songs you may like:')
        for song in data.sample(n=5)['track_name'].values:
            st.write(song)
        return

    data['similarity_factor'] = get_similarities(song_name, data)

    data.sort_values(by=['similarity_factor', 'popularity'], ascending=[False, False], inplace=True)

    st.write(data[['track_name', 'artists']].iloc[2:7])

# Streamlit UI
st.title('Song Recommendation System')

# Input field for song name
song_name = st.text_input('Enter the name of a song:', '')

# Button to trigger recommendation
if st.button('Recommend'):
    recommend_songs(song_name)
