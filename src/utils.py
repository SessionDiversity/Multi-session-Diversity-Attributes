import numpy as np
import math

def intra_div(session, leng_session, items_rep):  
    items_embed = session
    
    len__ = len(items_embed)

    if len__==1:
        return 0

    sum_sim = 0
    
    for idx, item_i in enumerate(items_embed):
        for item_j in items_embed[idx+1:]:
            sum_sim += items_rep[item_i][item_j]

    return sum_sim/(len__*(len__-1))

def precsion_at_k(session, ground_truth, leng_session):
    return len( set(session).intersection(set(ground_truth)) )/leng_session.numpy()[0]

def similar(item_a, item_b, items_rep):
    a = items_rep[item_a,:]
    b = items_rep[item_b,:]

    return np.dot(a, b)/( np.linalg.norm(a)*np.linalg.norm(b) )

def alpha_ndcg(session, ground_truth, leng_session, alpha=0.5):
    relevance = [int(i in ground_truth) for i in session]

    c_r = [sum(relevance[:i+1]) for i in range(leng_session)] #C(r-1)
    c_r = [(1-alpha)** i for i in c_r] #(1-alpha)^C(r-1)
    n_g = [c_r[i]*relevance[i] for i in range(leng_session)] #I(r)*(1-alpha)^C(r-1)
    g = [i/math.log2(j+2) for j,i in enumerate(n_g)]
    
    relevance.sort(reverse=True)

    c_r = [sum(relevance[:i+1]) for i in range(leng_session)] #C(r-1)
    c_r = [(1-alpha)** i for i in c_r] #(1-alpha)^C(r-1)
    n_g = [c_r[i]*relevance[i] for i in range(leng_session)] #I(r)*(1-alpha)^C(r-1)
    g_2 = [i/math.log2(j+2) for j,i in enumerate(n_g)]

    if sum(g_2) == 0:
        return 0

    if sum(g) > sum(g_2):
        print('TRUE')
        
    return sum(g)/sum(g_2)

def min_div(session, item, items_rep):
    return min([items_rep[i][item] for i in session])
