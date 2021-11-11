import torch
import torch.nn as nn
import torch.nn.functional as F

from math import sqrt
from torch.nn import init
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
from torch_geometric.nn.inits import zeros

class WGCNConv(MessagePassing):
    def __init__(self, in_channels, out_channels, node_num, num_edge, num_gene_edge, device):
        super(WGCNConv, self).__init__(aggr='add')
        self.node_num = node_num
        self.num_edge = num_edge
        self.num_gene_edge = num_gene_edge
        self.num_drug_edge = num_edge - num_gene_edge

        self.num_gene = 8002
        self.num_drug = 38

        self.lin = torch.nn.Linear(in_channels, out_channels)

        ##### [edge_weight] FOR ALL EDGES IN ONE [(gene+drug)] GRAPH #####
        consider_drug_edge = False
        ### [gene_edge_weight] [num_gene_edge / 59241] ###
        std_gene_edge = torch.nn.init.calculate_gain('relu')
        self.gene_edge_weight = torch.nn.Parameter((torch.randn(self.num_gene_edge) * std_gene_edge).to(device))
        # [drug_edge_weight] [num_drug_edge = num_edge - num_gene_edge] [/107*2=214]
        if consider_drug_edge == True:
            std_drug_edge = torch.nn.init.calculate_gain('relu')
            self.drug_edge_weight = torch.nn.Parameter((torch.randn(self.num_drug_edge) * std_drug_edge).to(device))
        else:
            self.drug_edge_weight = torch.ones(self.num_drug_edge).to(device)
        # [edge_weight] = [gene_edge_weight] + [drug_edge_weight] [59241+214=59455]
        self.edge_weight = torch.cat((self.gene_edge_weight, self.drug_edge_weight), 0)

        ### [loop_gene_edge_weight] [num_gene / 8002] ###
        std_loop_gene_edge = torch.nn.init.calculate_gain('relu')
        self.loop_gene_edge_weight = torch.nn.Parameter((torch.randn(self.num_gene) * std_loop_gene_edge).to(device))
        # [loop_drug_edge_weight] [num_drug / 38]
        if consider_drug_edge == True:
            std_loop_drug_edge = torch.nn.init.calculate_gain('relu')
            self.loop_drug_edge_weight = torch.nn.Parameter((torch.randn(self.num_drug) * std_loop_drug_edge).to(device))
        else:
            self.loop_drug_edge_weight = torch.ones(self.num_drug).to(device)
        # [loop_edge_weight] = [loop_gene_edge_weight] + [loop_drug_edge_weight] [8002+38=8040]
        self.loop_edge_weight = torch.cat((self.loop_gene_edge_weight, self.loop_drug_edge_weight), 0)

        # import pdb; pdb.set_trace()



    def forward(self, x, edge_index):
        # [batch_size]
        batch_size = int(x.shape[0] / self.node_num)
        # TEST PARAMETERS
        # import pdb; pdb.set_trace()
        print(torch.sum(self.gene_edge_weight))
        print(torch.sum(self.loop_gene_edge_weight))

        ### [edge_index, x] ###
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        x = self.lin(x)

        ### [edge_weight] ###
        # [batch_edge_weight] [N*59455]
        batch_edge_weight = self.edge_weight.repeat(1, batch_size)
        # [batch_loop_edge_weight] [N*8040]
        batch_loop_edge_weight = self.loop_edge_weight.repeat(1, batch_size)
        # [batch_addloop_edge_weight] [N*59455] + [N*8040]
        batch_addloop_edge_weight = torch.cat((batch_edge_weight, batch_loop_edge_weight), 1)
        edge_weight = batch_addloop_edge_weight

        # Step 3: Compute normalization.
        # [row] FOR 1st LINE && [col] FOR 2nd LINE
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-1/2)
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        # Check [ norm[0:59241] == norm[59455:118696] ]

        # Step 4-5: Start propagating messages.
        return self.propagate(edge_index, x=x, norm=norm, batch_addloop_edge_weight=edge_weight)

    def message(self, x_j, norm, batch_addloop_edge_weight):
        # [x_j] has shape [E, out_channels]
        # import pdb; pdb.set_trace()
        # Step 4: Normalize node features.
        weight_norm = torch.mul(norm, batch_addloop_edge_weight)
        return weight_norm.view(-1, 1) * x_j

    def update(self, aggr_out):
        # [aggr_out] has shape [N, out_channels]
        # import pdb; pdb.set_trace()
        return aggr_out


class WGCNDecoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, embedding_dim, decoder_dim, node_num, num_edge, num_gene_edge, device):
        super(WGCNDecoder, self).__init__()
        self.node_num = node_num
        self.embedding_dim = embedding_dim
        self.device = device
        self.conv_first, self.conv_block, self.conv_last = self.build_conv_layer(
                    input_dim, hidden_dim, embedding_dim, node_num, num_edge, num_gene_edge)
        
        self.act = nn.ReLU()
        self.act2 = nn.LeakyReLU(negative_slope = 0.1)

        self.parameter1 = torch.nn.Parameter(torch.randn(embedding_dim, decoder_dim).to(device='cuda'))
        self.parameter2 = torch.nn.Parameter(torch.randn(decoder_dim, decoder_dim).to(device='cuda'))

    def build_conv_layer(self, input_dim, hidden_dim, embedding_dim, node_num, num_edge, num_gene_edge):
        # conv_first [input_dim, hidden_dim]
        conv_first = WGCNConv(in_channels=input_dim, out_channels=hidden_dim,
                node_num=node_num, num_edge=num_edge, num_gene_edge=num_gene_edge, device=self.device)
        # conv_block [hidden_dim, hidden_dim]
        conv_block = WGCNConv(in_channels=hidden_dim, out_channels=hidden_dim,
                node_num=node_num, num_edge=num_edge, num_gene_edge=num_gene_edge, device=self.device)
        # conv_last [hidden_dim, embedding_dim]
        conv_last = WGCNConv(in_channels=hidden_dim, out_channels=embedding_dim,
                node_num=node_num, num_edge=num_edge, num_gene_edge=num_gene_edge, device=self.device)
        return conv_first, conv_block, conv_last

    def forward(self, x, edge_index, drug_index, label):
        x = self.conv_first(x, edge_index)
        x = self.act(x)

        x = self.conv_block(x, edge_index)
        x = self.act(x)

        x = self.conv_last(x, edge_index)
        x = self.act(x)
        # import pdb; pdb.set_trace()
        # x = torch.reshape(x, (-1, self.node_num, self.embedding_dim))
        drug_index = torch.reshape(drug_index, (-1, 2))

        # EMBEDDING DECODER TO [ypred]
        batch_size, drug_num = drug_index.shape
        ypred = torch.zeros(batch_size, 1).to(device='cuda')
        for i in range(batch_size):
            drug_a_idx = int(drug_index[i, 0]) - 1
            drug_b_idx = int(drug_index[i, 1]) - 1
            drug_a_embedding = x[drug_a_idx]
            drug_b_embedding = x[drug_b_idx]
            product1 = torch.matmul(drug_a_embedding, self.parameter1)
            product2 = torch.matmul(product1, self.parameter2)
            product3 = torch.matmul(product2, torch.transpose(self.parameter1, 0, 1))
            output = torch.matmul(product3, drug_b_embedding.reshape(-1, 1))
            ypred[i] = output
        # print(self.parameter1)
        # print(torch.sum(self.parameter1))
        # print(self.parameter2)
        return ypred

    def loss(self, pred, label):
        pred = pred.to(device='cuda')
        label = label.to(device='cuda')
        loss = F.mse_loss(pred.squeeze(), label)
        # print(pred)
        # print(label)
        return loss