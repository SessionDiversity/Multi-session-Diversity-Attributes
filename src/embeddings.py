import pandas as pd
import numpy as np
import scipy.sparse as sparse
import sklearn.preprocessing as pp
import torch.nn as nn
from keras.utils import np_utils
import itertools
import torch
import random
import setproctitle
import pickle
import os

setproctitle.setproctitle('Movie Embeddings') 

data_path = '../Data/data_files/movie_lens/10M/'
embedding_n_results_path = '../Data/no_attr/'

num_add_attr = 10 #Number of attributes to add

df_data = pd.read_csv(data_path+'ratings.dat',delimiter='::',
                      names=['userId','movieId','Rating','timestamp'])

imdb_movies = pd.read_csv(data_path+'title.akas.tsv',delimiter='\t')

names = pd.read_csv(data_path+'movies.dat',delimiter='::',
                    names=['id','name','genre'])

names.name = names.name.apply(lambda x: x.split(',')[1].strip().split('(')[0].strip()+' '
                              +x.split('(')[0].strip().split(',')[0].strip() 
                     if ',' in x else x.split('(')[0].strip())

names.name = names.name.apply(lambda x: x.lower())

names_groupby = names.groupby('name')['id'].apply(list).reset_index()
names_groupby['len'] = names_groupby['id'].apply(lambda x : len(x))
names_groupby = names_groupby[names_groupby.len > 1]

replace_dict = dict()

names_groupby = names_groupby['id'].tolist()

for rep in names_groupby:
    j = rep[0]
    for i in rep[1:]:
        replace_dict[i] = j

names = names.replace({'id':replace_dict})
names = names[~names.duplicated(['name','id'])]

imdb_movies['mask'] = imdb_movies.title.apply(lambda x: isinstance(x, str))
imdb_movies = imdb_movies[imdb_movies['mask'] == True]
imdb_movies.title = imdb_movies.title.apply(lambda x : x.lower())

possible_ones = set(imdb_movies.title.unique())
real_ones = set(names.name.unique())
real_ones = real_ones.intersection(possible_ones)

imdb_movies = imdb_movies[imdb_movies.title.isin(real_ones)]
imdb_movies = imdb_movies.drop_duplicates(subset=['title'])
imdb_movies = imdb_movies[['titleId','title']].rename(columns={'title':'name','titleId':'imdbId'})
names = names.merge(imdb_movies)

basics_movies = pd.read_csv(data_path+f'title.basics.tsv',delimiter='\t')
basics_movies = basics_movies[basics_movies.tconst.isin(names.imdbId.unique())]
basics_movies = basics_movies[['tconst','startYear','runtimeMinutes','titleType']].\
rename(columns={'tconst':'imdbId'})
names = names.merge(basics_movies)

df_data = df_data.replace({'movieId':replace_dict})
df_data = df_data[df_data.movieId.isin(names.id.unique())]

df_data.to_csv(data_path+'new_ratings.csv',index=False)

rt_movies = df_data.groupby('movieId')['Rating'].mean().reset_index().rename(columns={'movieId':'id'})

names = names.merge(rt_movies)
names.Rating = names.Rating.apply(lambda x: np.round(x))

names.genre = names.genre.apply(lambda x: x.split('|'))
genres = names.genre.tolist()
genres = list(itertools.chain(*genres))
genres = set(genres)
genres.remove('(no genres listed)')

for i in genres:
    if '(' not in i:
        names[i] = names.genre.apply(lambda x: 1 if i in x else 0)

names = names.replace({'runtimeMinutes':{"\\N":'1'}, 'startYear':{"\\N":'0'}})
names.runtimeMinutes = pd.to_numeric(names.runtimeMinutes)
names.startYear = pd.to_numeric(names.startYear)


#Create new attributes
dist = ['normal','uniform','zipf','expo','gamma']
len_ = len(names)

df_root = df_data.copy()
items = list(np.sort(df_data.movieId.unique()))  # all unique items
users = list(np.sort(df_data.userId.unique()))    # all unique users

attributes = names.drop(columns=['id', 'name', 'imdbId']+list(genres)).columns
attributes = list(attributes)
attributes.reverse()

add_attr = []

for d in range(num_add_attr):
    loi = np.random.choice(dist,p=[1/len(dist) for i in range(len(dist))])
    
    if loi == 'normal':
        centre = np.random.randn()
        std = np.random.rand()
        
        s = np.random.normal(centre, std, size=len_)
    
    elif loi =='uniform':
        a = np.random.randint(0,20)
        b = np.random.randint(0,20)
        
        if b > a:
            maxi = b
            mini = a
        else:
            maxi = a
            mini = b
        
        s = np.random.uniform(mini,maxi,size=len_)
    
    elif loi == 'zipf':
        param = 1+np.random.rand()
        s = np.random.zipf(param, len_)
    
    elif loi == 'expo':
        param = np.random.rand()+np.random.randint(0,6)
        
        s = np.random.exponential(param, len_)
        
    elif loi == 'gamma':
        k = np.random.randint(1,6)+np.random.rand()
        teta = np.random.randint(0,3)+np.random.rand()
        
        s = np.random.gamma(k,teta,len_)
    
    names[f'attr_{d}'] = s
    add_attr.append(f'attr_{d}')

attributes = attributes+add_attr

with open(embedding_n_results_path+f'attributes_{num_add_attr}', "wb") as f:
    pickle.dump(attributes, f)

for attr in attributes:
    if attr == 'Rating':
        df_data = df_root[['userId','movieId','Rating']]
        df_data = df_data.rename(columns={'userId':'user_id','movieId':'song_id','Rating':'play_count'})

        rating = list(np.sort(df_data.play_count))

        rows = df_data.user_id.astype(pd.api.types.CategoricalDtype(categories = users)).cat.codes    # Get the associated row indices
        cols = df_data.song_id.astype(pd.api.types.CategoricalDtype(categories = items)).cat.codes    # Get the associated row indices

        index_list = list(cols.values)
        items_ = list(df_data.song_id)

        song2idx = dict()
        idx2song = dict()

        j=0

        for i in items_:
            if i not in song2idx:
                song2idx[i] = index_list[j]+1
                idx2song[ index_list[j]+1 ] = i
                
            j=j+1

        if os.path.isfile(data_path+'song2idx_mini.pkl') == False:
            with open(data_path+'song2idx_mini.pkl', 'wb') as f:
                pickle.dump(song2idx, f)

        matrix = sparse.csc_matrix( (rating, (rows, cols)), shape = (len(users), len(items))  )
        similarity_matrix = pp.normalize(matrix.tocsc(), axis=0)
        similarity_matrix = similarity_matrix.T * similarity_matrix

    elif attr == 'genre':
        df_data = names[['id']+list(genres)]
        df_data['mask'] = df_data.id.apply(lambda x: song2idx[x]-1)
        df_data = df_data.sort_values(by='mask').drop(columns=['mask','id'])
        
        similarity_matrix = df_data.to_numpy()
        similarity_matrix = similarity_matrix @ similarity_matrix.T

    else:
        similarity_matrix = names[['id',attr]]

    window_size = 3
    context_length = window_size*2

    def get_sim_item(similarity_matrix, song_id, song2idx, num_similars, attr):
        if attr in ['genre','Rating']:
            idx = song2idx[song_id]-1
            song_sim = similarity_matrix[idx,:]
            if attr == 'Rating':
                song_sim = song_sim.toarray().reshape(-1)
            song_sim = np.argsort(song_sim)[::-1]
            song_sim = song_sim[:num_similars]
            return [i+1 for i in song_sim]

        elif 'attr_' in attr:
            attr_val = similarity_matrix[similarity_matrix.id == song_id][attr].tolist()[0]
            local = similarity_matrix[similarity_matrix.id != song_id]
            local[attr] = local[attr] - attr_val
            local = local.sort_values(by=attr)['id'].tolist()[:num_similars]

            return [song2idx[i] for i in local]

        else:
            attr_val = similarity_matrix[similarity_matrix.id == song_id][attr].tolist()[0]
            local = similarity_matrix[similarity_matrix[attr] == attr_val]['id'].tolist()
            local.remove(song_id)
            local = local[:num_similars]

            if len(local) < context_length:
                return [song2idx[i] for i in local]+[0 for i in range(context_length-len(local))]

            return [song2idx[i] for i in local]

    song2sim = dict()

    for i in items:
        song2sim[i] = get_sim_item(similarity_matrix, i , song2idx, 50, attr)

    corpus = []

    for i in items:
        for j in range(10):
            liste = list( np.random.choice(song2sim[i],context_length) )
            corpus.append( (song2idx[i],liste) )

    class CBOW(nn.Module):
        def __init__(self, vocab_size, embedding_dim):
            super(CBOW,self).__init__()
            self.embeddings = nn.Embedding(vocab_size, embedding_dim)
            self.linear_1 = nn.Linear(embedding_dim, 128)
            self.activation_1 = nn.ReLU()
            self.linear_2 = nn.Linear(128,vocab_size)
            self.activation_2 = nn.LogSoftmax(dim=-1)
        
        def forward(self, inputs, context_size, batch_size):
            inputs = torch.reshape( torch.Tensor(inputs).long(), (batch_size, context_size,-1) )
            a = self.embeddings(inputs).view(batch_size, context_size,-1)  
            
            embedds = torch.mean(a, 1)
            
            out = self.linear_1(embedds)
            out = self.activation_1(out)
            out = self.linear_2(out)
            out = self.activation_2(out)
            
            return out

    vocab_len = len(items)+1
    embed_dim = 256
    batch_size = 128

    model = CBOW(vocab_len,embed_dim)

    loss_f = nn.NLLLoss(reduction='mean')
    optimizer = torch.optim.SGD(model.parameters(),lr=0.001)

    print(f'Attr : {attr}')
    for epoch in range(5):
        model.train()
        loss = 0.
        
        random.shuffle(corpus)
        
        start = 0
        end = batch_size
        
        while start <= len(corpus):
            batch = corpus[start:end]
            
            X = []
            y = []
            
            for row in batch:
                X.append(row[1])
                y.append( np_utils.to_categorical(row[0],vocab_len ) )
            
            y = torch.Tensor(np.array(y)).long()
                
            optimizer.zero_grad()
            
            logs = model(X, context_length, len(batch))
            loss = loss_f(logs, torch.max(y, 1)[1])
            
            loss.backward()
            optimizer.step()
            loss += loss.data
            
            start += batch_size
            end += batch_size
            
        print('Epoch:', epoch, '\tLoss:', loss)

    weights = list(model.parameters())[0].detach().numpy()
    torch.save(weights, embedding_n_results_path+f'embeddings_movies_{attr}_{num_add_attr}.pt')

