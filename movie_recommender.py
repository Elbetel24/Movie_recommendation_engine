import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Helper functions

def get_title_from_index(index):
    return df[df['index'] == index]['title'].values[0]


def get_index_from_title(title):
    return df[df.title == title]['index'].values[0]


def combine_features(row):
    return ' '.join(str(row[feature]) for feature in features)


def get_similar_movies(watched_titles, top_n=20):
    watched_titles = [title.strip() for title in watched_titles if title.strip()]
    if not watched_titles:
        return []

    watched_indices = []
    for title in watched_titles:
        try:
            watched_indices.append(get_index_from_title(title))
        except IndexError:
            print(f"Warning: Movie '{title}' not found in the dataset.")

    if not watched_indices:
        return []

    # Combine similarity scores from each watched movie
    combined_scores = np.zeros(cosine_sim.shape[0])
    for movie_index in watched_indices:
        combined_scores += cosine_sim[movie_index]

    # Make sure watched movies are not recommended back
    for movie_index in watched_indices:
        combined_scores[movie_index] = -1

    sorted_indices = np.argsort(combined_scores)[::-1]
    recommendations = []

    for index in sorted_indices:
        if len(recommendations) >= top_n:
            break
        title = get_title_from_index(index)
        if title not in watched_titles:
            recommendations.append(title)

    return recommendations


# Step 1: Read CSV file

df = pd.read_csv('movie_dataset.csv')
features = ['keywords', 'cast', 'genres', 'director']
df[features] = df[features].fillna('')

df['combined_features'] = df.apply(combine_features, axis=1)

# Step 2: Build feature matrix and similarity scores
cv = CountVectorizer()
count_matrix = cv.fit_transform(df['combined_features'])
cosine_sim = cosine_similarity(count_matrix)

# Step 3: Ask the user for movies they have watched

input_text = input("Enter movies you've watched (comma-separated): ")
watched_list = [title.strip() for title in input_text.split(',') if title.strip()]

if not watched_list:
    print('No watched movies entered. Please enter at least one movie title.')
else:
    recommendations = get_similar_movies(watched_list, top_n=20)
    if recommendations:
        print(f"Movies similar to {', '.join(watched_list)}:")
        for title in recommendations:
            print(title)
    else:
        print('No recommendations could be generated. Check the movie titles and try again.')