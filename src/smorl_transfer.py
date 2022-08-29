import pandas as pd
import numpy as np
from utils import *
import torch
import torch.nn as nn
import random
import pickle
import sklearn.preprocessing as pp
import setproctitle
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-attr", type=str, default='no')
parser.add_argument("-num_sess", type=int, default=3)
parser.add_argument("-num_items", type=int, default=5)
args = parser.parse_args()

setproctitle.setproctitle(f'Smorl RL Transfer - {args.num_sess} - {args.num_items}')

class MSRC(nn.Module):
    def __init__(self, state_size, hidden_size, input_size, batch_size, num_items, num_items_in_session = 5):
        super(MSRC, self).__init__()

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

class Attr_choice(nn.Module):
    def __init__(self, state_size, hidden_size, input_size, batch_size, num_items, num_attributes, num_items_in_session = 5):
        super(Attr_choice, self).__init__()

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


for num_add_attr in [0,10,20,30,95]:

    for classe in [0,1,2]:

        data_path = '../Data/data_files/movie_lens/10M/'
        attr = args.attr
        embedding_n_results_path = f'../Data/{attr}_attr/'

        with open(embedding_n_results_path+f'attributes_{num_add_attr}', "rb") as f:
            attributes = pickle.load(f)

        num_sessions= args.num_sess
        num_items_in_session = args.num_items

        hidden_size = 256
        input_size = 256
        num_epochs = 1
        gamma = 0.99

        song2id_name = 'song2idx_mini.pkl'

        if classe == -1:
            file_name = f'buffer_csv_mini_transfer_div_{num_sessions}_{num_items_in_session}_{num_add_attr}.csv'
        else:
            file_name = f'buffer_csv_mini_transfer_div_classe_{classe}_{num_sessions}_{num_items_in_session}_{num_add_attr}.csv'

        with open(data_path+song2id_name, 'rb') as f:
            song2idx = pickle.load(f)

        items = list(song2idx.keys())

        embeddings_dict = dict()

        for i,attr in enumerate(attributes):
            embeddings = torch.load(embedding_n_results_path+f'embeddings_movies_{attr}_{num_add_attr}.pt')
            embeddings = embeddings[1:,:]
            similarity_matrix = pp.normalize(embeddings, axis=1)
            similarity_matrix = similarity_matrix @ similarity_matrix.T
            similarity_matrix = np.around(similarity_matrix, decimals=3)
            similarity_matrix = 1/(1+similarity_matrix)

            embeddings_dict[i] = similarity_matrix

        with open(data_path+file_name, 'rb') as f:
            reply_data = pickle.load(f)

        parameters = [(64,100,0.003)]

        for batch_size, state_size, learning_rate in parameters:

            policy_net = MSRC(state_size, hidden_size, input_size, batch_size, len(items), num_items_in_session)
            target_net = MSRC(state_size, hidden_size, input_size, batch_size, len(items), num_items_in_session)

            optimizer_1 = torch.optim.Adam(policy_net.parameters(), lr=learning_rate)
            optimizer_2 = torch.optim.Adam(target_net.parameters(), lr=learning_rate)

            policy_choice = Attr_choice(state_size, hidden_size, input_size, batch_size, len(items), len(attributes), num_items_in_session)
            target_choice = Attr_choice(state_size, hidden_size, input_size, batch_size, len(items), len(attributes), num_items_in_session)

            optimizer_choice_1 = torch.optim.Adam(policy_choice.parameters(), lr=learning_rate)
            optimizer_choice_2 = torch.optim.Adam(target_choice.parameters(), lr=learning_rate)

            rewards_intra = []
            rewards_inter = []
            reward_prec = []

            losses_1 = []
            losses_2 = []

            losses_choice = []

            num_epochs = 1

            for epoch in range(num_epochs):
                reply_data = reply_data.sample(frac=1)

                for l in range(0, len(reply_data), batch_size):

                    print(classe,' ', num_epochs,' ', l)

                    batch = reply_data[l:l+batch_size]

                    len_batch = len(batch)

                    state = np.array( batch.state.to_list() )
                    len_state = np.array( batch.len_state.to_list() )
                    action = np.array( batch.action.to_list() )
                    action_attr = np.array( batch.sec_action.to_list() )
                    next_state = np.array( batch.next_state.to_list() )
                    len_next_state = np.array( batch.len_next_state.to_list() )
                    done = np.array( batch.done.to_list() )

                    shp = state.shape[0]

                    state = state.reshape((shp,num_items_in_session))
                    next_state = next_state.reshape((shp,num_items_in_session))
                    len_state = len_state.reshape((shp,))
                    len_next_state = len_next_state.reshape((shp,))

                    state = torch.Tensor(state).long()
                    next_state = torch.Tensor(next_state).long()
                    len_state = torch.Tensor(len_state).long()
                    len_next_state = torch.Tensor(len_next_state).long()

                    action = torch.Tensor(action).view(-1,num_items_in_session).long()
                    action_attr = torch.Tensor(action_attr).view(-1,1).long()
                    done = torch.Tensor(done).view(-1)
                    # Optimize Model Below

                    done = (done == 0)

                    pointer = np.random.randint(0, 2)

                    if pointer == 0:
                        main = policy_net
                        target = target_net
                        optimizer = optimizer_1

                        main_attr = policy_choice
                        target_attr = target_choice
                        optimizer_attr = optimizer_choice_1
                    else:
                        main = target_net
                        target = policy_net
                        optimizer = optimizer_2

                        main_attr = target_choice
                        target_attr = policy_choice
                        optimizer_attr = optimizer_choice_2

                    ground_truth = next_state.detach().clone()

                    state_action_values, predidctions, state_embed = main(state.detach(), len_state.detach())
                    state_action_values = torch.gather( state_action_values, 1, action ).view(-1, num_items_in_session)#Q(s,a)
                    state_action_values = state_action_values.mean(1).view(-1)

                    top_k = torch.topk(predidctions,num_items_in_session,1)[1].detach().clone().view(-1,num_items_in_session)

                    next_state_values_main, _, next_state_embed = main(next_state.detach(), len_next_state.detach())
                    max_actions = next_state_values_main.max(1)[1].detach().clone().view(-1,1) #A*

                    next_state_values_target, _, next_state_embed_target = target(next_state.detach(), len_next_state.detach())
                    next_state_values_target = torch.gather( next_state_values_target, 1, max_actions ).detach().clone().view(-1)

                    next_state_values = torch.zeros(len_batch)
                    next_state_values[done] = next_state_values_target[done]

                    #Attribute
                    state_attr_values = main_attr(state_embed.detach())
                    state_attr_values = torch.gather( state_attr_values, 1, action_attr ).view(-1)#Q(s,a)

                    next_state_attr_main = main_attr(next_state_embed.detach())
                    max_actions_attr = next_state_attr_main.max(1)[1].detach().clone().view(-1,1) #A*

                    next_state_attr_target = target_attr(next_state_embed_target.detach())
                    next_state_attr_target = torch.gather( next_state_attr_target, 1, max_actions_attr ).detach().clone().view(-1)

                    next_state_attr_values = torch.zeros(len_batch)
                    next_state_attr_values[done] = next_state_attr_values[done]

                    intra = []
                    inter = []
                    sims = []

                    preds = torch.tensor([])
                    actions = []

                    for i in range(len_batch):
                        inter_preds = predidctions[i]

                        actions.extend(list(action[i].numpy()))
                        preds = torch.cat((preds, inter_preds.repeat(num_items_in_session,1)), 0)

                        local_next_state = top_k[i]
                        attr = max_actions_attr[i][0]

                        intra.append( intra_div(local_next_state, len_next_state[i], embeddings_dict[int(attr)]) )
                        inter.append( 0 )
                        sims.append( 0 )

                    rewards_intra.extend(intra)
                    rewards_inter.extend(inter)
                    reward_prec.extend(sims)

                    intra = torch.Tensor(intra).view(-1)
                    inter = torch.Tensor(inter).view(-1)
                    sims = torch.Tensor(sims).view(-1)

                    reward_batch = intra + inter + sims

                    expected_state_action_values = (next_state_values * gamma) + reward_batch

                    criterion_1 = nn.SmoothL1Loss()
                    loss_1 = criterion_1(state_action_values, expected_state_action_values)

                    criterion_2 = nn.CrossEntropyLoss()
                    loss_2 = criterion_2(preds, torch.tensor(actions).view(-1))

                    expected_state_attr_values = (next_state_attr_values * gamma) + reward_batch

                    criterion_attr = nn.SmoothL1Loss()
                    loss_attr = criterion_attr(state_attr_values,expected_state_attr_values)

                    losses_1.append(loss_1.item())
                    losses_2.append(loss_2.item())

                    losses_choice.append(loss_attr.item())

                    loss = loss_2 + loss_1

                    optimizer.zero_grad()
                    loss.backward()

                    optimizer_attr.zero_grad()
                    loss_attr.backward()

                    for param in main.parameters():
                        param.grad.data.clamp_(-1, 1)

                    for param in main_attr.parameters():
                        param.grad.data.clamp_(-1, 1)

                    optimizer.step()
                    optimizer_attr.step()

            file_part_name = f'_{learning_rate}_{state_size}_{batch_size}_classe_div_{classe}.pth'

            torch.save(policy_net.state_dict(), embedding_n_results_path+ f'models/smorl_qlearning_1_movies_{num_sessions}_{num_items_in_session}_attr_{num_add_attr}_topk'+
            file_part_name)

            torch.save(target_net.state_dict(), embedding_n_results_path+ f'models/smorl_qlearning_2_movies_{num_sessions}_{num_items_in_session}_attr_{num_add_attr}_topk'+
            file_part_name)

            torch.save(policy_choice.state_dict(), embedding_n_results_path+ f'models/smorl_attr_1_movies_{num_sessions}_{num_items_in_session}_attr_{num_add_attr}_topk'+
            file_part_name)

            torch.save(target_choice.state_dict(), embedding_n_results_path+ f'models/smorl_attr_2_movies_{num_sessions}_{num_items_in_session}_attr_{num_add_attr}_topk'+
            file_part_name)

            #dic={'rewards_inter':list(rewards_inter),'rewards_intra':list(rewards_intra),'reward_prec':list(reward_prec)}

            #results=pd.DataFrame(data=dic)
            #results.to_csv(embedding_n_results_path+ f'train_results/smorl_qlearning_rewards_movies_{num_sessions}_{num_items_in_session}_attr_{num_add_attr}_topk'+
            #f'_{learning_rate}_{state_size}_{batch_size}.csv',index=False)

            #dic={'losses_1':list(losses_1),'losses_2':list(losses_2), 'losses_attr':list(losses_choice)}

            #results=pd.DataFrame(data=dic)
            #results.to_csv(embedding_n_results_path+ f'train_results/smorl_qlearning_losses_movies_{num_sessions}_{num_items_in_session}_attr_{num_add_attr}_topk'+
            #f'_{learning_rate}x_{state_size}_{batch_size}.csv',index=False)

            