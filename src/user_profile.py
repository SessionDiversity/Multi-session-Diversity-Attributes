import pandas as pd
import numpy as np
import pickle
from utils import *
import scipy.sparse as sparse
import sklearn.preprocessing as pp

data_path = '../Data/data_files/movie_lens/10M/'

df_2 = pd.read_csv(data_path+'new_ratings.csv')
df_2 = df_2.rename(columns={'userId':'user_id','movieId':'song_id'})

items_ = set(df_2.song_id.unique())
df_2 = df_2[df_2.user_id.isin(users_data.keys())]
items_2 = set(df_2.song_id.unique())
remain_items = items_ - items_2

one_user = list(users_data.keys())[0]
max_idx = max(list(df_2.index))

df_2 = df_2.groupby(['user_id','song_id']).max().reset_index()

for i,it in enumerate(remain_items):
    df_2.loc[max_idx+i+1] = [one_user, it, 50, 0]

df = df_2

#del df_2

items = list(np.sort(df.song_id.unique()))  # all unique items
users = list(np.sort(df.user_id.unique()))    # all unique users
rating = list(np.sort(df.Rating))

rating = [i if i < 10 else 0 for i in rating]

rows = df.user_id.astype(pd.api.types.CategoricalDtype(categories = users)).cat.codes    # Get the associated row indices
cols = df.song_id.astype(pd.api.types.CategoricalDtype(categories = items)).cat.codes    # Get the associated row indices

index_list = list(rows.values)
users_ = list(df.user_id)
user2idx = dict()

j=0

for i in users_:
    if i not in user2idx:
        user2idx[i] = index_list[j]
        
    j=j+1

index_list = list(cols.values)
items_ = list(df.song_id)
items__ = dict()

j=0

for i in items_:
    if i not in items__:
        items__[i] = index_list[j]
        
    j=j+1

matrix = sparse.csc_matrix( (rating, (rows, cols)), shape = (len(users), len(items))  )

for enu, (key,value) in enumerate(users_data.items()):
    value = value['data_test']
    values_ = []

    for i,j in song2idx.items():
        if j in value:
            values_.append(items__[i])

    matrix[user2idx[key], values_] = 0

    print( enu/len(users_data.keys()) )


similarity_matrix = pp.normalize(matrix.tocsc(), axis=0)
similarity_matrix = similarity_matrix.T * similarity_matrix

user_profiles = dict()

for i,u in enumerate(users):
    idx = user2idx[u]
    purchased = matrix[idx,:].nonzero()[1]

    liste_sim = []
    for p in purchased:
        liste_sim.append(similarity_matrix[p,:].toarray()[0])

    reco_vector = liste_sim[0]
    for l in liste_sim[1:]:
        reco_vector= reco_vector + l
    
    reco_vector = reco_vector/len(liste_sim)
    #reco_vector = np.argsort(reco_vector)[::-1]

    user_profiles[ u ] = reco_vector
    print(i/len(users))

with open(data_path+'user_profiles_utility.pkl', 'wb') as f:
    pickle.dump(user_profiles, f)