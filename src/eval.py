import pandas as pd
import numpy as np
import torch
import pickle
import argparse
from utils import *
import torch.nn as nn
import scipy.sparse as sparse
import sklearn.preprocessing as pp
import multiprocessing
import sys
import math
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

        #session_emb = emb
        #session_lenght = lenghts.detach().clone() #Get the length of each session in each batch

        #Pack and padd: In the case if lengths are not maximal

        #emb_packed_next = torch.nn.utils.rnn.pack_padded_sequence(session_emb, session_lenght, batch_first=True, enforce_sorted=False)
        #emb_packed_next, _ = torch.nn.utils.rnn.pad_packed_sequence(emb_packed_next, batch_first=True, total_length=self.num_items_in_session)

        #print(emb_packed_next)

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

all_results = pd.DataFrame(columns=['method','nb_attributes','precision','recall','diversity','time','alpha_ndcg','attr_precision'])

nb_exec = 3

params = [
    [f'{0.003}_{100}_{64}'],
]

for execution in range(0, nb_exec):

    for num_add_attr in [0,10,20,30,95]:

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
        mask_users = np.random.choice(users_episode,500)

        for idx_param, param in enumerate(params):

            params_smorl = param[0].split('_')
            batch_size_smorl, state_size_smorl = int(params_smorl[2]),  int(params_smorl[1])

            #Get Smorl Model
            smorl_baseline = SMORL(state_size_smorl, hidden_size, input_size, batch_size_smorl, len(items), num_items_in_session)
            smorl_choice = Attr_choice_Smorl(state_size_smorl, hidden_size, input_size, batch_size_smorl, len(items), len(attributes), num_items_in_session)

            smorl_baseline.load_state_dict(torch.load(embedding_n_results_path+f'models/smorl_qlearning_1_movies_{num_sessions}_{num_items_in_session}_attr_{num_add_attr}_topk'+
            f'_{param[0]}.pth'))

            smorl_choice.load_state_dict(torch.load(embedding_n_results_path+f'models/smorl_attr_1_movies_{num_sessions}_{num_items_in_session}_attr_{num_add_attr}_topk'+
            f'_{param[0]}.pth'))

            smorl_baseline.eval()
            smorl_choice.eval()


            num_tot = 0

            times_smorl = 0
            times_mmr = 0
            times_swap = 0

            attribute_prec_smorl = 0

            num = num_sessions

            intras_smorl = [0 for i in range(num)]
            intras_swap = [0 for i in range(num)]
            intras_mmr = [0 for i in range(num)]
            intras_knn = [0 for i in range(len(attributes))]

            precs_smorl = [0 for i in range(num)]
            precs_swap = [0 for i in range(num)]
            precs_mmr = [0 for i in range(num)]
            precs_knn = [0 for i in range(num)]

            recs_smorl = [0 for i in range(num)]
            recs_swap = [0 for i in range(num)]
            recs_mmr = [0 for i in range(num)]
            recs_knn = [0 for i in range(num)]

            alpha_ndcg_smorl = [0 for i in range(num)]
            alpha_ndcg_swap = [0 for i in range(num)]
            alpha_ndcg_mmr = [0 for i in range(num)]
            alpha_ndcg_knn = [0 for i in range(num)]
            

            def mmr(liste, len_list, mask, utility, embedding, attr, alpha=0.6):
                if len_list == 0:
                    idx = np.argmax(utility*mask)
                else:
                    max_dist = np.array([min_div(liste, i, embedding[attr]) if j==1 else 0 for i,j in enumerate(mask)])
                    lin_combin = (1-alpha) * mask * utility + alpha * max_dist
                    idx = np.argmax(lin_combin)

                return idx

            def swap(liste, item, embedding, attr):
                best_liste = liste.clone().detach()
                best_liste[0] = item
                best_div = intra_div(best_liste, len(liste), embedding[attr]) 

                for i in range(1,len(liste)):
                    loc_liste = liste.clone().detach()
                    loc_liste[i] = item
                    div = intra_div(loc_liste, len(liste), embedding[attr])

                    if div > best_div:
                        best_div = div
                        best_liste = loc_liste

                return best_liste, best_div 

            def attr_same(attr1, attr2):
                if int(attr1) == int(attr2):
                    return 1     
                return 0


            for epoch in mask_users:

                user = epoch
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

                _, pred_smorl, emb = smorl_baseline(state_2, len_state)
                
                user_profile_utility_second = pred_smorl.detach().numpy()[0]
                user_profile_utility_second = user_profile_utility_second / abs(user_profile_utility_second.sum())

                user_profile_second = list(np.argsort(user_profile_utility_second)[::-1])

                state = np.array(user_profile_second[:num_items_in_session]).reshape((1,num_items_in_session))    
                state = torch.Tensor(state).long()

                state_smorl = state_2.detach().clone()
                state_mmr = state.detach().clone()
                state_swap = state.detach().clone()
                state_swap = state_swap[0]

                states_smorl = []
                states_swap = []
                states_mmr = []

                attributes_smorl = []
                attributes_swap = []
                attributes_mmr = []

                prec_attributes_smorl = []
                
                num_ite = num_items_in_session 

                time_smorl = 0
                mask_smorl = np.zeros((1,len(items)))

                #Smorl
                for j in range(num):
                    time_beg = time.time() #time 
                    _, pred_smorl, emb = smorl_baseline(state_smorl, len_state)
                    pred_smorl = pred_smorl.detach()+mask_smorl

                    top_k = torch.topk(pred_smorl,num_items_in_session,1)[1].detach().clone().view(-1,num_items_in_session)

                    k_mask = top_k[0]
                    mask_smorl[0,k_mask] = -10000000000
                    
                    attr = smorl_choice(emb).max(1)[1].view(-1,1)
                    attr_pos = int(attr[0][0])

                    state_smorl = top_k.detach().clone()
                    time_smorl += time.time() - time_beg #time

                    attributes_smorl.append(attr_pos)
                    states_smorl.append(state_smorl.detach().clone())

                if (idx_param == len(params)-1) and (num_tot < 200):
                    time_swap = 0
                    mask_swap = np.zeros(len(items))
                    us_prof = user_profile_second

                    #Swap
                    time_beg = time.time() #time 
                    for aa in range(num):
                        
                        div = []
                        lists = []

                        for idx, attr in enumerate(attributes):
                            s = state_swap.clone().detach()
                            prev_div = intra_div(s.numpy(), len_state, embeddings_dict[idx])

                            for j in range(num_items_in_session, num_items_in_session+num_ite):
                                item = us_prof[j]
                                best_list, best_div = swap(s, item, embeddings_dict, idx)

                                if best_div > prev_div:
                                    s = best_list.detach().clone()
                                    prev_div = best_div

                            div.append(prev_div)
                            lists.append(s.detach().clone())

                        pos_attr = div.index(max(div))
                        state_swap = lists[pos_attr]
                
                        time_swap += time.time() - time_beg

                        attributes_swap.append(pos_attr)
                        states_swap.append(state_swap.clone().detach())

                        mask_swap[state_swap.numpy()] = -100000000000000

                        us_prof = list(np.argsort(user_profile_utility_second+mask_swap)[::-1])
                        state = np.array(us_prof[:num_items_in_session]).reshape((1,num_items_in_session))    
                        state = torch.Tensor(state).long()

                        state_swap = state.detach().clone()
                        state_swap = state_swap[0]

                    time_mmr = 0
                    
                    time_beg = time.time() #time 
                    mask_global = np.ones(len(items)) 
                    #MMR
                    for aa in range(num):
                        div = []
                        lists = []
                        masks = []

                        for idx, attr in enumerate(attributes):
                            mask = mask_global.copy()

                            length_state = 0
                            s = state_mmr.clone().detach()

                            for j in range(num_items_in_session, num_items_in_session+num_ite):

                                if j == num_items_in_session:
                                    selected_item = mmr(None, length_state, mask, user_profile_utility_second, None, None)
                                else:
                                    selected_item = mmr(s.numpy()[0][:length_state], length_state, mask, user_profile_utility_second, embeddings_dict, idx)
                                    
                                mask[selected_item] = 0
                                s[0][length_state] = selected_item
                                length_state += 1

                            div.append( intra_div(s[0], len(s[0]), embeddings_dict[idx])  )
                            lists.append( s.detach() )
                            masks.append( mask )

                        pos_attr = div.index(max(div))
                        state_mmr = lists[pos_attr].detach()
                        mask_global = masks[pos_attr].copy()

                        time_mmr += time.time() - time_beg

                        attributes_mmr.append(pos_attr)
                        states_mmr.append(state_mmr.clone().detach())
                    
                best_attr_smorl = 0
                div_best_smorl = intra_div(state_smorl[0].clone(), num_items_in_session, embeddings_dict[0])

                for i in range(1,len(attributes)):
                    div_smorl = intra_div(state_smorl[0].clone(), num_items_in_session, embeddings_dict[i])
                    
                    if div_smorl > div_best_smorl:
                        div_best_smorl = div_smorl
                        best_attr_smorl = i

                attribute_prec_smorl += attr_same(best_attr_smorl, attributes_smorl[-1])
                

                divs_intra_smorl = []
                divs_intra_swap = []
                divs_intra_mmr = []
                divs_intra_knn = []

                prec_s_smorl = []
                prec_s_swap = []
                prec_s_mmr = []
                prec_s_knn = []

                rec_s_smorl = []
                rec_s_swap = []
                rec_s_mmr = []
                rec_s_knn = []

                a_ndcg_smorl = []
                a_ndcg_swap = []
                a_ndcg_mmr = []
                a_ndcg_knn = []


                for j in range(num):

                    items_pred_smorl = set(states_smorl[j][0].numpy())

                    if (idx_param == len(params)-1) and (num_tot < 200):
                        items_pred_swap = set(states_swap[j].numpy())
                        items_pred_mmr = set(states_mmr[j][0].numpy())
                        items_pred_knn = set(user_profile[j*num_items_in_session:(j+1)*num_items_in_session])
                    
                    test_session = set(test_sessions[j])

                    # Get overall precision and recall

                    prec_smorl = len( test_session.intersection(items_pred_smorl) ) / len(items_pred_smorl)
                    rec_smorl = len( test_session.intersection(items_pred_smorl) ) / len(test_items)

                    if (idx_param == len(params)-1) and (num_tot < 200):
                        prec_swap = len( test_session.intersection(items_pred_swap) ) / len(items_pred_swap)
                        rec_swap = len( test_session.intersection(items_pred_swap) ) / len(test_items)

                        prec_mmr = len( test_session.intersection(items_pred_mmr) ) / len(items_pred_mmr)
                        rec_mmr = len( test_session.intersection(items_pred_mmr) ) / len(test_items)

                        prec_knn = len( test_session.intersection(items_pred_knn) ) / len(items_pred_knn)
                        rec_knn = len( test_session.intersection(items_pred_knn) ) / len(test_items)

                    alpha_smorl = alpha_ndcg(states_smorl[j][0].clone().numpy(), test_session, num_items_in_session)
                    div_smorl = intra_div(states_smorl[j][0].clone(), num_items_in_session, embeddings_dict[attributes_smorl[j]])

                    if (idx_param == len(params)-1) and (num_tot < 200):
                        alpha_swap = alpha_ndcg(states_swap[j].clone().numpy(), test_session, num_items_in_session)
                        alpha_mmr = alpha_ndcg(states_mmr[j][0].clone().numpy(), test_session, num_items_in_session)
                        alpha_knn = alpha_ndcg(user_profile[j*num_items_in_session:(j+1)*num_items_in_session], test_session, num_items_in_session)
                        
                        div_swap = intra_div(states_swap[j].clone(), num_items_in_session, embeddings_dict[attributes_swap[j]])
                        div_mmr = intra_div(states_mmr[j][0].clone(), num_items_in_session, embeddings_dict[attributes_mmr[j]])

                        for j in range(len(attributes)):
                            divs_intra_knn.append( intra_div(user_profile[:num_items_in_session], num_items_in_session, embeddings_dict[j]) )

                    prec_s_smorl.append(prec_smorl)
                    rec_s_smorl.append(rec_smorl)
                    a_ndcg_smorl.append(alpha_smorl)
                    divs_intra_smorl.append( div_smorl )

                    if (idx_param == len(params)-1) and (num_tot < 200):
                        prec_s_swap.append(prec_swap)
                        prec_s_mmr.append(prec_mmr)
                        prec_s_knn.append(prec_knn)

                        rec_s_swap.append(rec_swap)
                        rec_s_mmr.append(rec_mmr)
                        rec_s_knn.append(rec_knn)

                        a_ndcg_swap.append(alpha_swap)
                        a_ndcg_mmr.append(alpha_mmr)
                        a_ndcg_knn.append(alpha_knn)

                        divs_intra_swap.append( div_swap )
                        divs_intra_mmr.append( div_mmr )

                times_smorl += time_smorl
                intras_smorl = [intras_smorl[i]+divs_intra_smorl[i] for i in  range(num)]
                precs_smorl = [precs_smorl[i]+prec_s_smorl[i] for i in  range(num)]
                recs_smorl = [recs_smorl[i]+rec_s_smorl[i] for i in  range(num)]
                alpha_ndcg_smorl = [alpha_ndcg_smorl[i]+a_ndcg_smorl[i] for i in  range(num)]

                if (idx_param == len(params)-1) and (num_tot < 200):
                    times_mmr += time_mmr
                    times_swap += time_swap

                    intras_swap = [intras_swap[i]+divs_intra_swap[i] for i in  range(num)]
                    intras_mmr = [intras_mmr[i]+divs_intra_mmr[i] for i in range(num)]
                    intras_knn = [intras_knn[i]+divs_intra_knn[i] for i in  range(len(attributes))]

                    precs_swap = [precs_swap[i]+prec_s_swap[i] for i in  range(num)]
                    precs_mmr = [precs_mmr[i]+prec_s_mmr[i] for i in range(num)]
                    precs_knn = [precs_knn[i]+prec_s_knn[i] for i in  range(num)]

                    recs_swap = [recs_swap[i]+rec_s_swap[i] for i in  range(num)]
                    recs_mmr = [recs_mmr[i]+rec_s_mmr[i] for i in range(num)]
                    recs_knn = [recs_knn[i]+rec_s_knn[i] for i in  range(num)]

                    alpha_ndcg_swap = [alpha_ndcg_swap[i]+a_ndcg_swap[i] for i in  range(num)]
                    alpha_ndcg_mmr = [alpha_ndcg_mmr[i]+a_ndcg_mmr[i] for i in range(num)]
                    alpha_ndcg_knn = [alpha_ndcg_knn[i]+a_ndcg_knn[i] for i in  range(num)]
                
            print(num_add_attr)

            all_results.loc[len(all_results)] = [f'SMORL_{param[0]}', num_add_attr+5, [100*i/num_tot for i in precs_smorl], [100*i/num_tot for i in recs_smorl], 
                                                    [i/num_tot for i in intras_smorl], times_smorl/num_tot, [100*i/num_tot for i in alpha_ndcg_smorl], 100*attribute_prec_smorl/num_tot]

            if idx_param == len(params)-1:
                if num_tot > 200:
                    num_tot = 200

                all_results.loc[len(all_results)] = ['Swap', num_add_attr+5, [100*i/num_tot for i in precs_swap], [100*i/num_tot for i in recs_swap], 
                                                        [i/num_tot for i in intras_swap], times_swap/num_tot, [100*i/num_tot for i in alpha_ndcg_swap], 0]
                
                all_results.loc[len(all_results)] = ['Mmr', num_add_attr+5, [100*i/num_tot for i in precs_mmr], [100*i/num_tot for i in recs_mmr], 
                                                        [i/num_tot for i in intras_mmr], times_mmr/num_tot, [100*i/num_tot for i in alpha_ndcg_mmr], 0]

                all_results.loc[len(all_results)] = ['Knn', num_add_attr+5, [100*i/num_tot for i in precs_knn], [100*i/num_tot for i in recs_knn], 
                                                        np.array([i/num_tot for i in intras_knn]).mean(), 0, [100*i/num_tot for i in alpha_ndcg_knn], 0]
            
    all_results.to_csv(f'all_results_{execution}_topk.csv',index=False)

for i in range(nb_exec):
    local = pd.read_csv(f'all_results_{i}_topk.csv')

    if i == 0:
        df = local
    else:
        df = df.append(local)

df.to_csv(f'all_results_topk_{num_sessions}.csv',index=False)
