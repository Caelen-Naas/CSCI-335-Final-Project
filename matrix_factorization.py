import numpy as np
import pandas as pd


class MatrixModel():
    """
    Common self variables used:
    VERBOSE --> print statementss 
    ratings --> pd frame of ratings
    movies --> pd frame of movies
    M --> matrix (pivot table)
    """
    VERBOSE = False

    #Variables
    n_users = None
    n_movies = None
    total_entries = None

    #Data frames
    M = None
    ratings = None
    movies = None

    #Initialize and load data
    def __init__(self, verbose):
        self.VERBOSE = verbose
        self.load_data()

    #Populate data frames for reviews and movies
    def load_data(self):
        #Load ratings from data set
        self.ratings = pd.read_csv(
            'data/u.data',
            sep='\t',
            names=['user_id','item_id','rating','timestamp']
        )

        #Load movie titles from data set
        self.movies = pd.read_csv(
            'data/u.item',
            sep='|',
            encoding='latin-1',
            usecols=[0,1],
            names=['item_id','title']
        )

        #Print ratings information
        if(self.VERBOSE == True):
            print(f'Ratings loaded:  {len(self.ratings):,} rows')
            print(f'Unique users:    {self.ratings.user_id.nunique()}')
            print(f'Unique movies:   {self.ratings.item_id.nunique()}')
            print(f'Rating range:    {self.ratings.rating.min()} - {self.ratings.rating.max()}')
            print(self.ratings.head(10))


    #Build the matrix
    def build_matrix(self):
        """
        Create a ratings matrix using loaded data
        Matrix will have columns of movies/titles and rows of users
        very sparse matrix with user reviews (1-5) as values 
        """

        m_df = self.ratings.pivot_table(index='user_id',columns='item_id',values='rating')

        #Determine how sparse the matrix is 
        n_users, n_movies = m_df.shape
        n_ratings = self.ratings.shape[0]
        total_entries = n_users * n_movies
        sparsity = 1 -(n_ratings/total_entries)

        #Create matrix with default values
        M = m_df.fillna(0).values

        if(self.VERBOSE == True):
            print(f'Matrix M dimensions: {n_users} users x {n_movies} movies')
            print(f'Total possible entries: {total_entries:,}')
            print(f'Known ratings:          {n_ratings:,}')
            print(f'Missing entries:        {total_entries - n_ratings:,}')
            print(f'Sparsity:               {sparsity:.1%}')
            print(f'\nMatrix M shape: {M.shape}')

        self.M = M
        self.n_users = n_users
        self.n_movies = n_movies
        self.total_entries = total_entries

    #Create split training and test set and training model
    def random_train_test(self, seed_val=42, split_val=0.8):
        np.random.seed(seed_val)
        all_ratings = self.ratings[['user_id','item_id','rating']].copy() #copy ratings, don't need time stamps
        all_ratings['user_idx'] = all_ratings['user_id']-1 #0-indexed for matrix
        all_ratings['item_idx'] = all_ratings['item_id'] -1

        shuffle = all_ratings.sample(frac=1, random_state=seed_val).reset_index(drop=True)
        split = int(split_val * len(shuffle)) #split data set based on split val (defaults to 80/20)

        train_df = shuffle.iloc[:split] #Assuming default... 80% training
        test_df = shuffle.iloc[split:] #Assuming default... 20% test

        #Build training matrix
        M_train = np.zeros((self.n_users, self.n_movies))
        for x, row in train_df.iterrows():
            M_train[int(row['user_idx']), int(row['item_idx'])] = row['rating']

        if self.VERBOSE:
            print(f'Training ratings: {len(train_df):,}')
            print(f'Test ratings:     {len(test_df):,}')
            print(f'Train sparsity:   {1 - len(train_df)/self.total_entries:.1%}')

        self.M_train = M_train


if __name__ == '__main__':
    matrix = MatrixModel(True)
    matrix.build_matrix()
    matrix.random_train_test()