import numpy as np
import pandas as pd
import torch
from utils import *
import sklearn.preprocessing as pp
import setproctitle
import argparse
import pickle
import os

setproctitle.setproctitle('Reply buffer')

parser = argparse.ArgumentParser()
parser.add_argument("-num_sess", type=int, default=3)
parser.add_argument("-num_items", type=int, default=5)
parser.add_argument("-num_add_attr", type=int, default=0)
args = parser.parse_args()

num_add_attr = args.num_add_attr
num_sessions = args.num_sess
num_items_in_session = args.num_items

embedding_n_results_path = '../Data/no_attr/'

with open(embedding_n_results_path+f'attributes_{num_add_attr}', "rb") as f:
    attributes = pickle.load(f)

data_path = '../Data/data_files/movie_lens/10M/'
embedding_n_results_path = f'../Data/no_attr/'

df = pd.read_csv(data_path+'new_ratings.csv')
df = df.rename(columns={'userId':'user_id','movieId':'song_id'})

users = df.user_id.unique()
items = df.song_id.unique()

embeddings_dict = dict()

for i,attr in enumerate(attributes):
    embeddings = torch.load(embedding_n_results_path+f'embeddings_movies_{attr}_{num_add_attr}.pt')
    embeddings = embeddings[1:,:]
    similarity_matrix = pp.normalize(embeddings, axis=1)
    similarity_matrix = similarity_matrix @ similarity_matrix.T
    similarity_matrix = np.around(similarity_matrix, decimals=3)
    similarity_matrix = 1/(1+similarity_matrix)

    embeddings_dict[i] = similarity_matrix

if os.path.isfile(data_path+'song2idx_mini.pkl') == False:
    cols = df.song_id.astype(pd.api.types.CategoricalDtype(categories = items)).cat.codes 

    index_list = list(cols.values)
    items_ = list(df.song_id)

    song2idx = dict()
    j=0

    for i in items_:
        if i not in song2idx:
            song2idx[i] = index_list[j]
        j=j+1

    with open(data_path+'song2idx_mini.pkl', 'wb') as f:
        pickle.dump(song2idx, f)

else:
    with open(data_path+'song2idx_mini.pkl', 'rb') as f:
        song2idx = pickle.load(f)


df['play_count'] = np.ones(len(df))
df = df[['user_id','song_id','play_count','timestamp']]

supp_users = df.groupby('user_id')['play_count'].sum().apply(lambda x: x >= 5*num_sessions*num_items_in_session).reset_index()
supp_users = set(supp_users[supp_users.play_count == True].user_id.unique())

df = df[df.user_id.isin(supp_users)]
df = df.sort_values(by=['user_id','timestamp'],ascending=True)

df_2 = df.copy()
df = df.groupby('user_id')['song_id'].apply(np.array).to_dict()

user_data = dict()

for idx , (us, local) in enumerate(df.items()):
    
    print(us,' ',idx/len(df))
    
    mod = len(local)%num_items_in_session

    if mod != 0:
        local = local[:-mod]

    test_local = local[-num_sessions*num_items_in_session:] #Get the ground Truth test set
    local = local[:-num_sessions*num_items_in_session] #Get the ground Truth train set

    test_local = [song2idx[i] for i in test_local]
    local = [song2idx[i] for i in local]

    user_data[us] = {'data_train': local, 'data_test': test_local, 'nb_actions': len(local)-1}


df_2.to_csv(data_path+f'train_set_movies_{num_sessions}_{num_items_in_session}', index=False)

file_name = f'buffer_dict_mini_{num_sessions}_{num_items_in_session}.pkl'

with open(data_path+file_name, 'wb') as f:
    pickle.dump(user_data, f)

states, len_st, action, sec_action, next_states, len_next, user, done = [], [], [], [], [], [], [], []

for idx , (us, local) in enumerate(user_data.items()):
    
    print(' --------- ',idx/len(user_data.keys()))

    local = local['data_train']
    state = local[:num_items_in_session]
    lengs = np.array([num_items_in_session])

    prev = num_items_in_session

    for items_idx in range(num_items_in_session*2,len(local)+num_items_in_session,num_items_in_session):
        next_state = local[prev:items_idx]

        div = [intra_div(next_state, lengs, embeddings_dict[i]) for i in range(len(attributes))]
        attr_best = np.argmax(div)
        
        states.append(np.array(state))
        len_st.append(lengs)
        action.append(np.array(next_state))
        sec_action.append(np.array([attr_best]))
        user.append(us)
        done.append(False)

        state = next_state
        prev = items_idx
    
    done[-1] = True

dic={'state':list(states),'len_state':list(len_st),'action':list(action),'sec_action':list(sec_action),
'next_state':list(action),'len_next_state':list(len_st),'done':list(done)}

reply_buffer=pd.DataFrame(data=dic)

file_name = f'buffer_csv_mini_{num_sessions}_{num_items_in_session}_{num_add_attr}.csv'

with open(data_path+file_name, 'wb') as f:
    pickle.dump(reply_buffer, f)
