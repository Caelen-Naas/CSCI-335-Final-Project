import numpy as np
import pandas as pd

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
print(f'Ratings loaded:  {len(ratings):,} rows')
print(f'Unique users:    {ratings.user_id.nunique()}')
print(f'Unique movies:   {ratings.item_id.nunique()}')
print(f'Rating range:    {ratings.rating.min()} – {ratings.rating.max()}')
print()
ratings.head(10)