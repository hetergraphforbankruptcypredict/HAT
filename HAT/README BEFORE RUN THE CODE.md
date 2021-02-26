# Heterogeneous Graph Attention Network for SME's Bankruptcy Prediction

This source code is developed based on the implementation of HAN using DGL, which can be found on https://github.com/dmlc/dgl/tree/master/examples/pytorch/han.

## How to run the code?

Simply `%run main.py` for reproducing our work on our dataset.

If you want to regenerate top-k type-specific neighbors and the test set and validation set. Please run the following code in order:


%run random_walk_withrestart.py
%run c_ineigh_feature_generation.py
%run test_data_generation.py
%run main.py

## About the dataset

To evaluate our proposed model in bankruptcy prediction, we collect and build a real-world dataset, which contains the board member network and the shareholder network for 13489 companies in China, from multiple public sources such as the National Enterprise Credit Information Publicity System (gsxt.gdgs.gov.cn) and China Judgment Online (zxgk.court.gov.cn). Specifically, we randomly selected 1000 companies, which located in a south-eastern city in China and went bankrupt in 2018. Then, we extend the network by collecting all the shareholders and board members for these firms and repeat this process for the collected nodes twice. Finally, the original 1000 nodes network has been extended to a much larger network with 13489 nodes.

This dataset was collected in early October 2020.

Explanation of dataset files:
c_c.npy --> company to company edges
c_i.npy --> company to individual edges
c_top_neigh.npy --> top 5 individual neighbors' nodes for each company 
c_ineigh_feature --> top 5 individual neighbors' features for each company 
company_bankrupt_before2019.npy --> company nodes which bankrupt before 2019
label_ci_l.npy --> labels
test_data.pkl --> including graph inputs/ features/ training/validation/test set index

*Please noted that, with the concern of privacy protection, this dataset has been preprocessed and anonymized. The statistics of our dataset is summarized as follows.


Dataset's statistics
| Statistic                                 |Datasets|
\# of company nodes                 13489 
\# of individual nodes                  6855
\# of total edges                        209195
\# of $CIC_s$ edges                   53874
\# of $CIC_b$ edges                 139413
\# of $CC_s$ edges                    15908



Labels Statistics
| Statistic                                   |Labels |
\# of bankrupt company             3566  
\# of non-bankruptcy company  9923
\# of bankrupt  before 2019        2432
\# of bankrupt  in 2019                848
\# of bankrupt  in 2020                216


