import argparse
import numpy as np
import pandas as pd
import random
from collections import Counter

parser = argparse.ArgumentParser('deepwalk')
parser.add_argument('-s', '--seed', type=int, default=1,
                        help='Random seed')
parser.add_argument('-c','--company',type = int, default = 13489, help = 'number of companies')
parser.add_argument('-i','--individual', type = int, default = 6855, help = 'number of individuals')
parser.add_argument('-k','--k', type = int, default = 5, help = 'number of top neighbors being extracted')
args, unknown = parser.parse_known_args()

edge_list_cc = np.load('c_c.npy',allow_pickle=True)
edge_list_ic = np.load('c_i.npy', allow_pickle=True)

cc = pd.DataFrame(edge_list_cc).drop_duplicates()
ci = pd.DataFrame(edge_list_ic).drop_duplicates()

#c_c_neigh stores company type of neighbors for each company
c_c_neigh = [[] for i in range(args.company)]
for i in range(args.company):
    cc_temp = cc[cc[0] == 'c' + str(i)]
    c_c_neigh[i] = cc_temp.iloc[:,1].tolist()

#i_c_neigh stores company type of neighbors for each individual
i_c_neigh = [[] for i in range(args.individual)]
for i in range(args.individual):
    ci_temp = ci[ci[0] == 'i' + str(i)]
    i_c_neigh[i] = ci_temp.iloc[:,1].tolist()

for i in range(args.individual):
    for j in i_c_neigh[i]:
        c_c_neigh[int(j[1:])].append('i' + str(i))
        
#generate random walk sequence for each company
c_neigh_list_train = [[] for k in range(args.company)]
i_neigh_list_train = [[] for k in range(args.individual)]
node_n = [args.company,args.individual]
for i in range(2):
    for j in range(node_n[i]):
        if i == 0:
            neigh_temp = c_c_neigh[j]
            neigh_train = c_neigh_list_train[j]
            curNode = 'c' + str(j)
        elif i == 1:
            neigh_temp = i_c_neigh[j]
            neigh_train = i_neigh_list_train[j]
            curNode = 'i' + str(j)
        if len(neigh_temp):
            neigh_L = 0
            while neigh_L < 200:
                rand_p = random.random()
                if rand_p > 0.5:
                    if curNode[0] == 'c':
                        curNode = random.choice(c_c_neigh[int(curNode[1:])])
                        neigh_train.append(curNode)
                        neigh_L += 1
                    elif curNode[0] == 'i':
                        curNode = random.choice(i_c_neigh[int(curNode[1:])])
                        neigh_train.append(curNode)
                        neigh_L += 1
                else:
                    if i == 0:
                        curNode = ('c' + str(j))
                    elif i == 1:
                        curNode = ('i' + str(j))

#c_i_train_list only include individual nodes in the random walk sequence
c_i_train_list = [[] for i in range(args.company)]
for i in range(len(c_neigh_list_train)):
    for item in c_neigh_list_train[i]:
        if item[0] == 'i':
            c_i_train_list[i].append(item)
            

#find top-k neighbors for each company
#no_i_neigh stores company which do not have or can't find their individual type of neighbors
c_top_neigh = [[] for i in range(args.company)]
no_i_neigh = []
for i in range(args.company):
    most_common = Counter(c_i_train_list[i]).most_common(args.k)
    if len(most_common):
        for item in most_common:
            c_top_neigh[i].append(item[0])
        while len(c_top_neigh[i]) < args.k:
            neigh = random.choice(most_common)
            c_top_neigh[i].append(neigh[0])
    else:
        no_i_neigh.append(i)

np.save('c_top_neigh.npy',c_top_neigh)

print('----top k individual type neighbors for each company has been stored in c_top_neigh.npy----')
