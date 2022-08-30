import pandas as pd
import numpy as np
import torch
import pickle
import argparse
from utils import *
import torch.nn as nn
import sklearn.preprocessing as pp

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

session_size = 300

state_size = 200
hidden_size = 256
input_size = 256

batch_size = 256

num_sessions = args.num_sess
num_items_in_session = args.num_items


with open(data_path+'song2idx_mini.pkl', 'rb') as f:
    song2idx = pickle.load(f)

with open(data_path+f'buffer_dict_mini_transfer_div_{num_sessions}_{num_items_in_session}.pkl', 'rb') as f:
    users_data = pickle.load(f)

with open(data_path+f'test_dict_mini_transfer_div_{num_sessions}_{num_items_in_session}.pkl', 'rb') as f:
    users_data_transfer = pickle.load(f)

all_results = pd.DataFrame(columns=['method','test_classe','precision','recall','diversity','time','alpha_ndcg','attr_precision'])

nb_exec = 1

data = {0:list([]), 1:list([]), 2:list([])}
data_transter = {0:list([]), 1:list([]), 2:list([])}

for idx , (us, local) in enumerate(users_data.items()):
    test_classe = local['classe']

    user_items = local['data_train']
    user_items = user_items[-num_items_in_session:]

    user_items = np.array(user_items).reshape((1,num_items_in_session))

    test_items = local['data_test']
    test_sessions = [set(test_items[i:i+num_items_in_session]) for i in range(0,num_items_in_session*num_sessions,num_items_in_session)]

    test_items = set(test_items)

    data[test_classe].append( [us,user_items,test_sessions,test_items] )

for idx , (us, local) in enumerate(users_data_transfer.items()):
    test_classe = local['classe']

    user_items = local['data_test']
    user_items = user_items[:-num_items_in_session*num_sessions]
    user_items = user_items[-num_items_in_session:]

    user_items = np.array(user_items).reshape((1,num_items_in_session))

    test_items = local['data_test']
    test_items = test_items[-num_items_in_session*num_sessions:]
    test_sessions = [set(test_items[i:i+num_items_in_session]) for i in range(0,num_items_in_session*num_sessions,num_items_in_session)]

    test_items = set(test_items)

    data_transter[test_classe].append( [us,user_items,test_sessions,test_items] )


param = [f'{0.003}_{100}_{64}']

for execution in range(nb_exec):

    num_add_attr = 0

    with open(embedding_n_results_path+f'attributes/attributes_{num_add_attr}', "rb") as f:
        attributes = pickle.load(f)

    embeddings_dict = dict()

    for i,attr in enumerate(attributes):
        embeddings = torch.load(embedding_n_results_path+f'embeddings/embeddings_movies_{attr}_{num_add_attr}.pt')
        embeddings = embeddings[1:,:]
        similarity_matrix = pp.normalize(embeddings, axis=1)
        similarity_matrix = similarity_matrix @ similarity_matrix.T
        similarity_matrix = np.around(similarity_matrix, decimals=3)
        similarity_matrix = 1/(1+similarity_matrix)

        embeddings_dict[i] = similarity_matrix

    items = list(song2idx.keys())
    
    data_sampl = dict() #Sample users

    for test_classe in [0,1,2,-1,-2,-3]:
        #index = np.random.choice( range(len(data[test_classe])) ,6000)
        #data_sampl[test_classe] = np.array(data[test_classe])[index]

        if test_classe < 0:
            data_sampl[test_classe] = data_transter[-1*(test_classe+1)]
        else:
            data_sampl[test_classe] = data[test_classe]

    for classe in [0,1,2]:
        
        print(f' ---- {classe} ----')

        params_smorl = param[0].split('_')
        batch_size_smorl, state_size_smorl = int(params_smorl[2]),  int(params_smorl[1])

        #Get Smorl Model
        smorl_baseline = SMORL(state_size_smorl, hidden_size, input_size, batch_size_smorl, len(items), num_items_in_session)
        smorl_choice = Attr_choice_Smorl(state_size_smorl, hidden_size, input_size, batch_size_smorl, len(items), len(attributes), num_items_in_session)

        smorl_baseline.load_state_dict(torch.load(embedding_n_results_path+f'models/smorl_qlearning_1_movies_{num_sessions}_{num_items_in_session}_attr_{num_add_attr}_topk'+
        f'_{param[0]}_classe_div_{classe}.pth'))

        smorl_choice.load_state_dict(torch.load(embedding_n_results_path+f'models/smorl_attr_1_movies_{num_sessions}_{num_items_in_session}_attr_{num_add_attr}_topk'+
        f'_{param[0]}_classe_div_{classe}.pth'))

        smorl_baseline.eval()
        smorl_choice.eval()

        for test_classe in [-1,0,1,2]:
            
            if test_classe == -1:
                mask_users = data_sampl[-1*(classe+1)]
            else:
                mask_users = data_sampl[test_classe]

            print(f'\t ---- {test_classe} - {len(mask_users)} ----')
            num_tot = 0

            attribute_prec_smorl = 0

            num = 1#num_sessions

            intras_smorl = [0 for i in range(num)]
            precs_smorl = [0 for i in range(num)]
            recs_smorl = [0 for i in range(num)]
            alpha_ndcg_smorl = [0 for i in range(num)]

            for epoch in mask_users:
                
                us, user_items, test_sessions, test_items = epoch[0], epoch[1], epoch[2], epoch[3]

                if len(test_items) < 1:
                    continue

                num_tot += 1
  
                len_state = np.array([num_items_in_session])
                len_state = torch.Tensor(len_state).long()

                state_2 = torch.Tensor(user_items).long()
                state_smorl = state_2.detach().clone()

                states_smorl = []
                attributes_smorl = []
                prec_attributes_smorl = []
                
                mask_smorl = np.zeros((1,len(items)))

                #Smorl
                for j in range(num):
                    _, pred_smorl, emb = smorl_baseline(state_smorl, len_state)
                    pred_smorl = pred_smorl.detach()+mask_smorl

                    top_k = torch.topk(pred_smorl,num_items_in_session,1)[1].detach().clone().view(-1,num_items_in_session)

                    k_mask = top_k[0]
                    mask_smorl[0,k_mask] = -10000000000
                    
                    attr = smorl_choice(emb).max(1)[1].view(-1,1)

                    attr_pos = int(attr[0][0])

                    state_smorl = top_k.detach().clone()

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

                divs_intra_smorl = []
                prec_s_smorl = []
                rec_s_smorl = []
                a_ndcg_smorl = []

                for j in range(num):
                    items_pred_smorl = set(states_smorl[j][0].numpy())

                    test_session = set(test_sessions[j])

                    # Get overall precision and recall

                    prec_smorl = len( test_session.intersection(items_pred_smorl) ) / len(items_pred_smorl)
                    rec_smorl = len( test_session.intersection(items_pred_smorl) ) / len(test_items)
                    alpha_smorl = alpha_ndcg(states_smorl[j][0].clone().numpy(), test_session, num_items_in_session)
                    div_smorl = intra_div(states_smorl[j][0].clone(), num_items_in_session, embeddings_dict[attributes_smorl[j]])

                    prec_s_smorl.append(prec_smorl)
                    rec_s_smorl.append(rec_smorl)
                    a_ndcg_smorl.append(alpha_smorl)
                    divs_intra_smorl.append( div_smorl )

                intras_smorl = [intras_smorl[i]+divs_intra_smorl[i] for i in  range(num)]
                precs_smorl = [precs_smorl[i]+prec_s_smorl[i] for i in  range(num)]
                recs_smorl = [recs_smorl[i]+rec_s_smorl[i] for i in  range(num)]
                alpha_ndcg_smorl = [alpha_ndcg_smorl[i]+a_ndcg_smorl[i] for i in  range(num)]

            all_results.loc[len(all_results)] = [f'SMORL_{classe}', test_classe, [100*i/num_tot for i in precs_smorl], [100*i/num_tot for i in recs_smorl], 
                                                [i/num_tot for i in intras_smorl], 0, [100*i/num_tot for i in alpha_ndcg_smorl], 100*attribute_prec_smorl/num_tot]
          
    all_results.to_csv(f'all_results_{execution}_transfer_div.csv',index=False)

for i in range(nb_exec):
    local = pd.read_csv(f'all_results_{i}_transfer_div.csv')

    if i == 0:
        df = local
    else:
        df = df.append(local)

df.to_csv(f'all_results_topk_{num_sessions}_transfer_div.csv',index=False)
