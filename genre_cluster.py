from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report
from collections import Counter

# Parameters
K = 5
DISTANCE_METRIC = 'hamming' # hamming distance should work for binary vectors I think

# File paths
GENRES_PATH = Path('data/u.genre')
MOVIES_PATH = Path('data/u.item')

def main():

    # Read in all genres
    genres = []
    with open(GENRES_PATH, 'r') as file:
        for line in file.readlines():
            if line.strip():
                genre_name = line.split('|')[0]
                genres.append(genre_name)
    print(f"Read in {len(genres)} genres: {genres}")

    # Read in movies
    movie_names = list() # names
    movie_genre_vectors = list()
    with open(MOVIES_PATH, 'r', encoding='latin-1') as file:
        for line in file:
            split_line = line.split('|')
            split_line = [v.replace('\n', '') for v in split_line]
            movie_names.append(split_line[1])
            movie_genre_vectors.append(split_line[5:])

    # Create genre vector
    movie_vector_array = np.array(movie_genre_vectors, dtype=int)
    # Create labeled movies vector
    labeled_movies = np.array([np.argmax(row) if row.sum() > 0 else -1 for row in movie_vector_array])
    counts = Counter(labeled_movies)

    # Only use labels that have more than 1 movie
    valid_labels = {label for label, count in counts.items() if count >= 2}

    # Apply mask of valid labels
    mask = np.array([label in valid_labels for label in labeled_movies])
    movie_vector_array = movie_vector_array[mask]
    labeled_movies = labeled_movies[mask]

    # Create training/testing split with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        movie_vector_array, labeled_movies, test_size=0.2, random_state=67, stratify=labeled_movies
    )

    # Fit KNN
    knn = KNeighborsClassifier(n_neighbors=K, metric=DISTANCE_METRIC)
    knn.fit(X_train, y_train)

    # Make a prediction
    y_pred = knn.predict(X_test)
    unique_labels = sorted(set(y_test))
    target_names = [genres[i] for i in unique_labels]

    # Print report
    print("Classification Report")
    print(classification_report(y_test, y_pred, target_names=target_names, labels=unique_labels))

if __name__ == '__main__':
    main()