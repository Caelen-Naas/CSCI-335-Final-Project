import numpy as np
import pandas as pd


VERBOSE = True

def load_data():
    #Load ratings from data set
    ratings = pd.read_csv(
        'data/u.data',
        sep='\t',
        names=['user_id','item_id','rating','timestamp']
    )

    #Load movie titles from data set
    movies = pd.read_csv(
        'data/u.item',
        sep='|',
        encoding='latin-1',
        usecols=[0,1],
        names=['item_id','title']
    )

    #Print ratings information
    if(VERBOSE == True):
        print(f'Ratings loaded:  {len(ratings):,} rows')
        print(f'Unique users:    {ratings.user_id.nunique()}')
        print(f'Unique movies:   {ratings.item_id.nunique()}')
        print(f'Rating range:    {ratings.rating.min()} - {ratings.rating.max()}')
        print(ratings.head(10))

    return ratings, movies
        

#Build the matrix
def build_matrix():
    """
    Create a ratings matrix using loaded data
    Matrix will have columns of movies/titles and rows of users
    very sparse matrix with user reviews (1-5) as values 
    """
    ratings, movies = load_data()

    m_df = ratings.pivot_table(index='user_id',columns='item_id',values='rating')

    #Determine how sparse the matrix is 
    n_users, n_movies = m_df.shape
    n_ratings = ratings.shape[0]
    total_entries = n_users * n_movies
    sparsity = 1 -(n_ratings/total_entries)

    #Create matrix with default values
    M = m_df.fillna(0).values

    if(VERBOSE == True):
        print(f'Matrix M dimensions: {n_users} users x {n_movies} movies')
        print(f'Total possible entries: {total_entries:,}')
        print(f'Known ratings:          {n_ratings:,}')
        print(f'Missing entries:        {total_entries - n_ratings:,}')
        print(f'Sparsity:               {sparsity:.1%}')
        print(f'\nMatrix M shape: {M.shape}')

    return M


if __name__ == '__main__':
    build_matrix()