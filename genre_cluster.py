from pathlib import Path
import numpy as np
from sklearn.cluster import KMeans

GENRES_PATH = Path('data/u.genre')
MOVIES_PATH = Path('data/u.item')


def main():

    # Read in all genres
    genres = list()
    with open(GENRES_PATH, 'r') as file:
        for line in file.readlines():
            if line != '\n':
                genre_name = line.split('|')[0]
                genres.append(genre_name)
    print(f"Read in {len(genres)} genres: {genres}")

    movie_genre_clusters = list()

    # Read in all movies and their genres
    with open(MOVIES_PATH, 'r', encoding='latin-1') as file:
        for line in file:
            split_line = line.split('|')
            split_line = [value.replace('\n', '') for value in split_line]
            movie_genre_clusters.append(split_line[5:])

    kmeans = KMeans(n_clusters=2, random_state=0, n_init="auto").fit(np.array(movie_genre_clusters))
    print(kmeans)



if __name__ == '__main__':
    main()