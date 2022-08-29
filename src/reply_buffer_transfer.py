import numpy as np
import pandas as pd
import torch
from utils import *
import sklearn.preprocessing as pp
import setproctitle
import argparse
import pickle
from sklearn.cluster import KMeans
import os

setproctitle.setproctitle('Reply buffer Transfer')

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
    cols = df.song_id.astype(pd.api.types.CategoricalDtype(categories = items)).cat.codes    # Get the associated row indices

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

#Classes
df_2 = df.copy()
df = df.groupby('user_id')['song_id'].apply(np.array).to_dict()

df_4 = df_2.copy()
df_4.song_id = df_4.song_id.apply(lambda x: song2idx[x])
df_4 = df_4.groupby('user_id')['song_id'].apply(np.array).reset_index()

for i in range(len(attributes)):
    df_4[f'div_{i}'] = df_4.song_id.apply(lambda x: intra_div( x[-2*num_sessions*num_items_in_session:-num_sessions*num_items_in_session], 0, embeddings_dict[i]) )

df_4['div'] = df_4[f'div_0']

for i in range(1,len(attributes)):
    df_4['div'] = df_4['div']+df_4[f'div_{i}']

df_4['div'] = df_4['div']/len(attributes)

kmeans = KMeans(n_clusters=3)
y = kmeans.fit_predict(df_4['div'].to_numpy().reshape((-1,1)))

df_4 = df_4[['user_id']]
df_4['classe'] = y

df_3 = df_4.copy()

nb_classes = df_3.groupby('classe').size().reset_index().set_index('classe').T.to_dict('list')
df_3 = df_3[['user_id','classe']].set_index('user_id').T.to_dict('list')

#End Classes

user_data = dict()
user_data_transfer = dict()

counte = [0,0,0]

for idx , (us, local) in enumerate(df.items()):
    
    classe = df_3[us][0]
    print(us,' ',idx/len(df),f' classe = {classe} - {counte[classe]}',)
    
    mod = len(local)%num_items_in_session

    if mod != 0:
        local = local[:-mod]

    if counte[classe] < 0.75*nb_classes[classe][0]:
        
        counte[classe] += 1

        test_local = local[-num_sessions*num_items_in_session:] #Get the ground Truth test set
        local = local[:-num_sessions*num_items_in_session] #Get the ground Truth train set

        test_local = [song2idx[i] for i in test_local]
        local = [song2idx[i] for i in local]

        user_data[us] = {'data_train': local, 'data_test': test_local, 'nb_actions': len(local)-1, 'classe':classe}
    else:
        test_local = [song2idx[i] for i in local]
        user_data_transfer[us] = {'data_test': test_local, 'nb_actions': len(local)-1, 'classe':classe}

file_name = f'buffer_dict_mini_transfer_div_{num_sessions}_{num_items_in_session}.pkl'

with open(data_path+file_name, 'wb') as f:
    pickle.dump(user_data, f)

file_name = f'test_dict_mini_transfer_div_{num_sessions}_{num_items_in_session}.pkl'

with open(data_path+file_name, 'wb') as f:
    pickle.dump(user_data_transfer, f)

states_transfer = [[],[],[]]
len_st_transfer = [[],[],[]]
action_transfer = [[],[],[]]
sec_action_transfer = [[],[],[]]
next_states_transfer = [[],[],[]]
len_next_transfer = [[],[],[]]
user_transfer = [[],[],[]]
done_transfer= [[],[],[]]

for idx , (us, local) in enumerate(user_data.items()):
    
    print(' --------- ',idx/len(user_data.keys()))
    
    classe = local['classe']
    local = local['data_train']

    state = local[:num_items_in_session]
    lengs = np.array([num_items_in_session])

    prev = num_items_in_session

    for items_idx in range(num_items_in_session*2,len(local)+num_items_in_session,num_items_in_session):
        next_state = local[prev:items_idx]

        div = [intra_div(next_state, lengs, embeddings_dict[i]) for i in range(len(attributes))]
        attr_best = np.argmax(div)

        states_transfer[classe].append(np.array(state))
        len_st_transfer[classe].append(lengs)
        action_transfer[classe].append(np.array(next_state))
        sec_action_transfer[classe].append(np.array([attr_best]))
        user_transfer[classe].append(us)
        done_transfer[classe].append(False)

        state = next_state
        prev = items_idx
    
    done_transfer[classe][-1] = True


for classe in [0,1,2]:

    dic={'state':list(states_transfer[classe]),'len_state':list(len_st_transfer[classe]),'action':list(action_transfer[classe]),'sec_action':list(sec_action_transfer[classe]),
    'next_state':list(action_transfer[classe]),'len_next_state':list(len_st_transfer[classe]),'done':list(done_transfer[classe])}

    reply_buffer=pd.DataFrame(data=dic)

    print(f'classe = {classe} - {len(reply_buffer)}')

    file_name = f'buffer_csv_mini_transfer_div_classe_{classe}_{num_sessions}_{num_items_in_session}_{num_add_attr}.csv'

    with open(data_path+file_name, 'wb') as f:
        pickle.dump(reply_buffer, f)
