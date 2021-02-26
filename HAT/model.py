import torch
import torch.nn as nn
import torch.nn.functional as F

from dgl.nn.pytorch import GATConv

#the attention layer in the heterogeneous neighborhood encoding layer
class FeatureAttention(nn.Module):
    
    def __init__(self, in_size, hidden_size = 16):
        super(FeatureAttention, self).__init__()

        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size), 
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )
        
    def forward(self, z):
        w = self.project(z)                            # (13489, 5, 1)
        beta = torch.softmax(w, dim=1)                 # (13489, 5, 1)
        
        return (beta*z).sum(1)                         # (N, D * K)

#metapath-level attention and network level attention
class SemanticAttention(nn.Module):
    def __init__(self, in_size, hidden_size=128):
        super(SemanticAttention, self).__init__()

        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )

    def forward(self, z):
        w = self.project(z).mean(0)                    # (M, 1)
        beta = torch.softmax(w, dim=0)                 # (M, 1)
        beta = beta.expand((z.shape[0],) + beta.shape) # (N, M, 1)
        
        return (beta * z).sum(1)                       # (N, D * K)

class HANLayer(nn.Module):
    """
    HAN layer.

    Arguments
    ---------
    num_meta_paths : number of homogeneous graphs generated from the metapaths.
    in_size : input feature dimension
    out_size : output feature dimension
    layer_num_heads : number of attention heads
    dropout : Dropout probability

    Inputs
    ------
    g : list[DGLGraph]
        List of graphs
    h : tensor
        Input features

    Outputs
    -------
    tensor
        The output feature
    """
    def __init__(self, num_meta_paths, in_size, out_size, layer_num_heads, dropout):
        super(HANLayer, self).__init__()

        # One GAT layer for each meta path based adjacency matrix
        self.gat_layers = nn.ModuleList()
        for i in range(num_meta_paths):
            self.gat_layers.append(GATConv(in_size, out_size, layer_num_heads,
                                           dropout, dropout, activation=F.elu))
        self.semantic_attention = SemanticAttention(in_size=out_size * layer_num_heads)
        self.num_meta_paths = num_meta_paths

    def forward(self, gs, h):
        semantic_embeddings = []

        for i, g in enumerate(gs):
            semantic_embeddings.append(self.gat_layers[i](g, h).flatten(1))
            
        semantic_embeddings = torch.stack(semantic_embeddings, dim=1)                  # (N, M, D * K)

        return self.semantic_attention(semantic_embeddings)                            # (N, D * K)

#HAN uses a dual-level attention
class HAN(nn.Module):
    def __init__(self, num_meta_paths, in_size, hidden_size, num_heads, dropout):
        super(HAN, self).__init__()
        self.i_dim = 42
        self.FeatureAttention = FeatureAttention(self.i_dim)
        self.layers = nn.ModuleList()
        self.layers.append(HANLayer(num_meta_paths, in_size, hidden_size, num_heads[0], dropout))
        for l in range(1, len(num_heads)):
            self.layers.append(HANLayer(num_meta_paths, hidden_size * num_heads[l-1],
                                        hidden_size, num_heads[l], dropout))
        
        
    def forward(self, g, h, c_ineigh_feature):
        c_i_embedding = self.FeatureAttention(c_ineigh_feature)
        h = torch.cat((h,c_i_embedding),1)
        
        for gnn in self.layers:
            h = gnn(g, h)

        return h
        
#HAT uses a triple level attention
class HAT(nn.Module):
    def __init__(self, num_meta_paths, in_size, hidden_size, out_size, num_heads, dropout):
        super(HAT, self).__init__()
        self.network_emebdding_indim = 144  #network_embedding's dimension generated from HAN
        self.network_embedding_outdim = 64
        self.project = nn.Linear(self.network_emebdding_indim, self.network_embedding_outdim)
        self.network_attention = SemanticAttention(in_size = self.network_embedding_outdim)
        self.layers = HAN(num_meta_paths, in_size, hidden_size, num_heads, dropout)
        self.predict = nn.Linear(self.network_embedding_outdim, out_size)
        
    def forward(self, g, h, c_ineigh_feature, num_of_network):
        network_embedding = []
        for i in range(num_of_network):
            network_embedding.append(self.layers(g[i], h, c_ineigh_feature))
        
        for i in range(len(network_embedding)):
            network_embedding[i] = self.project(network_embedding[i])
        
        network_embedding = torch.stack(network_embedding, dim = 1)
        global_embedding = self.network_attention(network_embedding)
        
        return self.predict(global_embedding)
    
    
    
        

    
   
