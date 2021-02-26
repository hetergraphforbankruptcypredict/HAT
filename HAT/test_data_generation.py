import pickle
import numpy as np

with open('networks_features_labels.pkl', 'rb') as f:
    data = pickle.load(f)
    
    
company_bankrupt_before2019 = np.load('company_bankrupt_before2019.npy', allow_pickle=True)
label_ci_l = np.load('label_ci_l.npy', allow_pickle=True)

num_classes = 2
length = 13489

float_mask = np.zeros(length)
train_idx = np.array(company_bankrupt_before2019)
val_idx = 0
test_idx = 0
for i in range(2):
    if i == 0:
        cic_mask = (label_ci_l == i)
        float_mask[cic_mask] = np.random.permutation(np.linspace(0, 1, cic_mask.sum()))
        train_idx_1 = np.where((float_mask <= 0.68)&(float_mask > 0))[0]
        train_idx = np.append(train_idx, train_idx_1)
        val_idx = np.where((float_mask > 0.68) & (float_mask <= 0.84))[0]
        test_idx = np.where(float_mask > 0.84)[0]
    elif i == 1:
        cic_mask = (label_ci_l == i)
        bankruptcynode = np.array(list(set(np.where(cic_mask == True)[0]) - set(company_bankrupt_before2019)))
        float_mask = np.zeros(length)
        float_mask[bankruptcynode] = np.random.permutation(np.linspace(0, 1, len(bankruptcynode)))
        val_idx_1 = np.where(float_mask >= 0.5)[0]
        test_idx_1 = np.where((float_mask <= 0.5) & (float_mask > 0))[0]
        val_idx = np.append(val_idx, val_idx_1)
        test_idx = np.append(test_idx, test_idx_1)

data['train_idx'] = train_idx
data['val_idx'] = val_idx
data['test_idx'] = test_idx

with open('test_data.pkl', 'wb') as fo:
    data = pickle.dump(data,fo)

print('----test_data is now ready, you can now run the main.py----')
