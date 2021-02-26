import numpy as np
from sklearn.preprocessing import normalize

sorted_ind_f = np.load('individual_feature.npy', allow_pickle= True)
c_neigh_list = np.load('c_top_neigh.npy', allow_pickle= True)


for i in range(sorted_ind_f.shape[0]):
    for j in range(sorted_ind_f.shape[1]):
        sorted_ind_f[i][j] = int(sorted_ind_f[i][j])
sorted_ind_f = sorted_ind_f.astype(float)

c_new_neigh = np.zeros((13489, 5))
for i in range(c_neigh_list.shape[0]):
    if len(c_neigh_list[i]):
        for j in range(5):
            c_new_neigh[i][j] = int(c_neigh_list[i][j][1:])

c_embedding = np.zeros((13489, 5, 42))
for i in range(c_embedding.shape[0]):
    if np.all(c_new_neigh[i] == 0):
        pass
    else:
        for j in range(c_embedding.shape[1]):
            i_neigh_feature = sorted_ind_f[int(c_new_neigh[i][j])]
            c_embedding[i][j] = i_neigh_feature

c_ineigh_feature = c_embedding
for i in range(len(c_embedding)):
  c_embedding[i] = normalize(c_embedding[i], axis = 1)
np.save('c_ineigh_feature.npy', c_embedding)

print("---each company's individual type neighborhood has been stored in c_ineigh_feature.npy----")