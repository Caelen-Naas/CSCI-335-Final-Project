from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

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

    movie_names = []
    movie_genre_vectors = []

    # Read in all movies and their genres
    with open(MOVIES_PATH, 'r', encoding='latin-1') as file:
        for line in file:
            split_line = line.split('|')
            split_line = [v.replace('\n', '') for v in split_line]
            movie_names.append(split_line[1])
            movie_genre_vectors.append(split_line[5:])

    X = np.array(movie_genre_vectors, dtype=int)

    # --- Cluster with K=19 (one per genre) ---
    n_clusters = 19
    kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init="auto")
    labels = kmeans.fit_predict(X)

    # --- Reduce to 2D for plotting ---
    # Option 1: PCA (fast, linear)
    pca = PCA(n_components=2)
    X_2d = pca.fit_transform(X)

    # Option 2: t-SNE (slower, better separation — swap in if you prefer)
    # tsne = TSNE(n_components=2, random_state=0, perplexity=40)
    # X_2d = tsne.fit_transform(X)

    # --- Plot ---
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(
        X_2d[:, 0], X_2d[:, 1],
        c=labels,
        cmap='tab20',      # 20-color palette — great for 19 clusters
        alpha=0.6,
        s=15
    )
    plt.colorbar(scatter, label='Cluster')
    plt.title(f'Movie Genre Clusters (KMeans k={n_clusters}, PCA projection)')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.tight_layout()
    plt.savefig('genre_clusters.png', dpi=150)
    plt.show()

    # --- Print dominant genre per cluster ---
    print("\nDominant genre per cluster:")
    for i in range(n_clusters):
        cluster_movies = X[labels == i]
        genre_sums = cluster_movies.sum(axis=0)
        top_genre = genres[np.argmax(genre_sums)]
        print(f"  Cluster {i:2d} ({len(cluster_movies):4d} movies): {top_genre}")


if __name__ == '__main__':
    main()