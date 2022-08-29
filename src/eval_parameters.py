import pandas as pd
import numpy as np
import torch
import pickle
import argparse
from utils import *
import torch.nn as nn
import sklearn.preprocessing as pp
import time

#SMORL
class SMORL(nn.Module):
    def __init__(self, state_size, hidden_size, input_size, batch_size, num_items, num_items_in_session = 5):
        super(SMORL, self).__init__()

        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.input_size = input_size
        self.num_items = num_items
        self.num_items_in_session = num_items_in_session

        self.emmb = nn.Embedding(num_items+1, input_size)

        self.gru_sessions = nn.GRU(input_size, hidden_size, batch_first=True)
        self.fc_1 = nn.Linear(hidden_size, state_size) 
        self.relu = nn.ReLU()

        self.fc_2 = nn.Linear(state_size, state_size)
        self.softmax = nn.Softmax(dim=1)

        self.s_fc = nn.Linear(state_size, num_items)

        self.fc_5 = nn.Linear(state_size, int(num_items/2))
        self.fc_6 = nn.Linear(int(num_items/2), num_items)

    def forward(self, inputs, lenghts):
        emb = self.emmb(inputs.clone()) #[Batch_size, num_track_in_session, Embedding_tracks]

        out,_ = self.gru_sessions(emb)

        out = self.fc_1(out)
        out = self.relu(out)

        out_2 = self.fc_2(out)
        out_2 = self.softmax(out_2)
        
        out = out * out_2
        state = out.sum(1)

        out = self.fc_5(state)
        out = self.fc_6(out)

        out_2 = self.s_fc(state)

        return out, out_2, state

class Attr_choice_Smorl(nn.Module):
    def __init__(self, state_size, hidden_size, input_size, batch_size, num_items, num_attributes, num_items_in_session = 5):
        super(Attr_choice_Smorl, self).__init__()

        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.input_size = input_size
        self.num_items = num_items
        self.num_items_in_session = num_items_in_session
        self.num_sessions = num_sessions
        self.num_attributes = num_attributes

        self.fc_1 = nn.Linear(state_size, 128)
        self.relu = nn.ReLU()
        self.fc_2 = nn.Linear(128, num_attributes)
    
    def forward(self, state):

        out = self.fc_1(state)
        out = self.relu(out)
        
        out = self.fc_2(out)

        return out

def attr_same(attr1, attr2):
    if int(attr1) == int(attr2):
        return 1     
    return 0

parser = argparse.ArgumentParser()
parser.add_argument("-num_sess", type=int, default=3)
parser.add_argument("-num_items", type=int, default=5)
args = parser.parse_args()

data_path = '../Data/data_files/movie_lens/10M/'
attr = 'no'
embedding_n_results_path = f'../Data/{attr}_attr/'

hidden_size = 256
input_size = 256

num_sessions = args.num_sess
num_items_in_session = args.num_items


with open(data_path+'song2idx_mini.pkl', 'rb') as f:
    song2idx = pickle.load(f)

with open(data_path+f'buffer_dict_mini_{num_sessions}_{num_items_in_session}.pkl', 'rb') as f:
    users_data = pickle.load(f)

with open(data_path+'user_profiles_utility.pkl', 'rb') as f:
    user_profiles = pickle.load(f)

all_results = pd.DataFrame(columns=['method','batch_size','state_size','decay/worker','l_r','precision','recall','diversity','alpha_ndcg','attr_precision'])

nb_exec = 2
num_add_attr = 0

batch_sizes = [256, 64]
l_r = [0.0001, 0.0003, 0.001, 0.003]
state_sizes = [100, 200, 500]
target_updates = [5, 10]

for execution in range(nb_exec):

    with open(embedding_n_results_path+f'attributes_{num_add_attr}', "rb") as f:
        attributes = pickle.load(f)

    embeddings_dict = dict()

    for i,attr in enumerate(attributes):
        embeddings = torch.load(embedding_n_results_path+f'embeddings_movies_{attr}_{num_add_attr}.pt')
        embeddings = embeddings[1:,:]
        similarity_matrix = pp.normalize(embeddings, axis=1)
        similarity_matrix = similarity_matrix @ similarity_matrix.T
        similarity_matrix = np.around(similarity_matrix, decimals=3)
        similarity_matrix = 1/(1+similarity_matrix)

        embeddings_dict[i] = similarity_matrix

    items = list(song2idx.keys())
    users_episode = list(users_data.keys())

    for batch_size in batch_sizes:
        for state_size in state_sizes:
            for target_update in target_updates:
                for learning_rate in l_r:

                    print(f'SMORL - {learning_rate} - {target_update} - {state_size} - {batch_size}')

                    #Get DQN Model
                    smorl = SMORL(state_size, hidden_size, input_size, batch_size, len(items), num_items_in_session)
                    smorl_choice = Attr_choice_Smorl(state_size, hidden_size, input_size, batch_size, len(items), len(attributes), num_items_in_session)

                    smorl.load_state_dict(torch.load(embedding_n_results_path+
                    f'models/parameters_study/smorl_qlearning_1_movies_{learning_rate}_{target_update}_{state_size}_{batch_size}.pth'))
                    
                    smorl_choice.load_state_dict(torch.load(embedding_n_results_path+
                    f'models/parameters_study/smorl_attr_1_movies_{learning_rate}_{target_update}_{state_size}_{batch_size}.pth'))

                    smorl.eval()
                    smorl_choice.eval()

                    num_tot = 0
                    num = 1

                    attribute_prec_smorl = 0
                    intras_smorl = 0
                    precs_smorl = 0
                    recs_smorl = 0
                    alpha_ndcg_smorl = 0

                    for epoch in range(len(users_episode)):
                        if epoch % 1000 == 0:
                            print(f'epoch : {epoch}')

                        user_id = epoch % len(users_episode)

                        user = users_episode[user_id]
                        us = user

                        user_profile_utility = user_profiles[user]
                        user_profile = list(np.argsort(user_profile_utility)[::-1])

                        user = users_data[user]

                        user_items = user['data_train']
                        user_items = user_items[-num_items_in_session:]
                        user_items = np.array(user_items).reshape((1,num_items_in_session))

                        static_dict = dict()

                        test_items = user['data_test']
                        test_sessions = [set(test_items[i:i+num_items_in_session]) for i in range(0,num_items_in_session*num_sessions,num_items_in_session)]

                        test_items_ = test_items
                        test_items = set(test_items)

                        if len(test_items) < 1:
                            continue

                        num_tot += 1
 
                        len_state = np.array([num_items_in_session])
                        len_state = torch.Tensor(len_state).long()

                        state_2 = torch.Tensor(user_items).long()
                        state_smorl = state_2.detach().clone()

                        attributes_smorl = []
                        states_smorl = []

                        time_smorl = 0
                        #Model
                        for j in range(num):
                            time_beg = time.time() #time 

                            _, pred_smorl, embed = smorl(state_smorl, len_state)

                            pred_smorl = torch.topk(pred_smorl,num_items_in_session,1)[1].detach().clone().view(-1,num_items_in_session)
                            attr = smorl_choice(embed).max(1)[1].view(-1,1)
                            
                            attr_pos = int(attr[0][0])

                            state_smorl = pred_smorl.detach().clone()

                            time_smorl += time.time() - time_beg #time

                            attributes_smorl.append(attr_pos)
                            states_smorl.append(state_smorl.detach().clone())
    
                        best_attr_smorl = 0
                        div_best_smorl = intra_div(state_smorl[0].clone(), num_items_in_session, embeddings_dict[0])

                        for i in range(1,len(attributes)):
                            div_smorl = intra_div(state_smorl[0].clone(), num_items_in_session, embeddings_dict[i])

                            if div_smorl > div_best_smorl:
                                div_best_smorl = div_smorl
                                best_attr_smorl = i

                        attribute_prec_smorl += attr_same(best_attr_smorl, attributes_smorl[-1])

                        items_pred_smorl = set(states_smorl[0][0].numpy())

                        prec_smorl = len( test_items.intersection(items_pred_smorl) ) / len(items_pred_smorl)
                        rec_smorl = len( test_items.intersection(items_pred_smorl) ) / len(test_items)

                        alpha_smorl = alpha_ndcg(states_smorl[0][0].clone().numpy(), test_items, num_items_in_session)
                        div_smorl = intra_div(states_smorl[0][0].clone(), num_items_in_session, embeddings_dict[attributes_smorl[0]])

                        intras_smorl += div_smorl
                        precs_smorl += prec_smorl
                        recs_smorl += rec_smorl
                        alpha_ndcg_smorl += alpha_smorl

                    all_results.loc[len(all_results)] = ['SMORL', batch_size, state_size, target_update, learning_rate, 100*precs_smorl/num_tot , 100*recs_smorl/num_tot, 
                                            intras_smorl/num_tot, 100*alpha_ndcg_smorl/num_tot, 100*attribute_prec_smorl/num_tot]

all_results.to_csv(f'all_results_parameters.csv',index=False)