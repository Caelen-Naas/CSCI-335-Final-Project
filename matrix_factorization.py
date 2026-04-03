import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error


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
    M_train = None
    test_df = None

    #Matrix factorization factors
    U = None  # user latent matrix  (n_users  x k)
    V = None  # item latent matrix  (n_movies x k)

    #Matrix factorization hyperparameters
    k = None
    alpha = None
    lambda_reg = None
    n_epochs = None

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
        self.test_df = test_df


    def init_factors(self, k=20, alpha=0.005, lambda_reg=0.02, n_epochs=10):
        """
        Initialise the latent factor matrices U and V with small random values.
        Must be called after build_matrix() so that n_users and n_movies are known.

        Parameters
        ----------
        k          : number of latent dimensions
        alpha      : SGD learning rate
        lambda_reg : L2 regularisation coefficient
        n_epochs   : number of full passes through the training data
        """
        self.k = k
        self.alpha = alpha
        self.lambda_reg = lambda_reg
        self.n_epochs = n_epochs

        self.U = np.random.normal(scale=0.1, size=(self.n_users, k))
        self.V = np.random.normal(scale=0.1, size=(self.n_movies, k))

        if self.VERBOSE:
            print(f'Latent factors initialised — U: {self.U.shape}, V: {self.V.shape}')
            print(f'Hyperparameters: k={k}, alpha={alpha}, lambda_reg={lambda_reg}, n_epochs={n_epochs}')

    def predict(self, u, i):
        """
        Predict the rating for user u and item i.
        U[u] · V[i]  (dot product of latent vectors)
        """
        return np.dot(self.U[u], self.V[i])

    def predict_all(self):
        """
        Reconstruct the full ratings matrix: M̂ = U @ Vᵀ
        Every cell is filled — including previously missing ratings.
        """
        return np.dot(self.U, self.V.T)

    def train(self):
        """
        Train the model via Stochastic Gradient Descent.
        Uses self.M_train built by random_train_test() and evaluates against
        self.test_df if available.

        For each epoch:
          1. Shuffle all known (user, item, rating) triples from M_train
          2. Compute prediction error for each triple
          3. Update U[u] and V[i] with a regularised gradient step

        Returns
        -------
        train_losses : list of per-epoch average regularised squared error
        test_rmses   : list of per-epoch RMSE on the test set (empty if no test_df)
        """
        rows, cols = self.M_train.nonzero()
        train_samples = list(zip(rows, cols, self.M_train[rows, cols]))

        train_losses = []
        test_rmses = []

        for epoch in range(self.n_epochs):
            np.random.shuffle(train_samples)
            epoch_loss = 0.0

            for u, i, r in train_samples:
                r_hat = self.predict(u, i)
                e = r - r_hat

                u_old = self.U[u].copy()
                self.U[u] += self.alpha * (e * self.V[i] - self.lambda_reg * self.U[u])
                self.V[i] += self.alpha * (e * u_old    - self.lambda_reg * self.V[i])

                epoch_loss += e**2 + self.lambda_reg * (
                    np.sum(self.U[u]**2) + np.sum(self.V[i]**2)
                )

            train_losses.append(epoch_loss / len(train_samples))

            if self.test_df is not None:
                test_preds = [
                    np.clip(self.predict(int(row['user_idx']), int(row['item_idx'])), 1, 5)
                    for _, row in self.test_df.iterrows()
                ]
                rmse = np.sqrt(mean_squared_error(self.test_df['rating'], test_preds))
                test_rmses.append(rmse)

            if (epoch + 1) % 10 == 0:
                rmse_str = f'{test_rmses[-1]:.4f}' if test_rmses else 'N/A'
                print(f'Epoch {epoch+1:>3}/{self.n_epochs}  '
                      f'Train loss: {train_losses[-1]:.4f}  '
                      f'Test RMSE:  {rmse_str}')

        return train_losses, test_rmses


if __name__ == '__main__':
    matrix = MatrixModel(True)
    matrix.build_matrix()
    matrix.random_train_test()

    np.random.seed(42)
    matrix.init_factors(k=20, alpha=0.005, lambda_reg=0.02, n_epochs=50)

    print('\n--- Model Attributes ---')
    print(f'U shape: {matrix.U.shape}  (one {matrix.k}-d vector per user)')
    print(f'V shape: {matrix.V.shape}  (one {matrix.k}-d vector per movie)')
    print(f'Total parameters: {matrix.U.size + matrix.V.size:,} '
          f'vs {matrix.n_users * matrix.n_movies:,} in the full matrix')

    print('\n--- Training ---')
    train_losses, test_rmses = matrix.train()

    print(f'\nFinal Test RMSE: {test_rmses[-1]:.4f}')
    print('(RMSE of 1.0 means predictions are off by ~1 star on average)')