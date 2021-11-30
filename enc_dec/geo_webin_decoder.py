import torch
import torch.nn as nn
import torch.nn.functional as F

from math import sqrt
from torch.nn import init
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
from torch_geometric.nn.inits import zeros

class WeBGNNConv(MessagePassing):
    def __init__(self, in_channels, out_channels, node_num, num_edge, num_gene_edge, device):
        super(WeBGNNConv, self).__init__(aggr='add')
        self.node_num = node_num
        self.num_edge = num_edge
        self.num_gene_edge = num_gene_edge
        self.num_drug_edge = num_edge - num_gene_edge

        self.up_proj = torch.nn.Linear(in_channels, out_channels, bias=False)
        self.down_proj = torch.nn.Linear(in_channels, out_channels, bias=False)
        self.bias_proj = torch.nn.Linear(in_channels, out_channels, bias=False)

        ##### [edge_weight] FOR ALL EDGES IN ONE [(gene+drug)] GRAPH #####
        ### [up_gene_edge_weight] [num_gene_edge / 59241] ###
        up_std_gene_edge = torch.nn.init.calculate_gain('relu')
        self.up_gene_edge_weight = torch.nn.Parameter((torch.randn(self.num_gene_edge) * up_std_gene_edge).to(device))
        ### [down_gene_edge_weight] [num_gene_edge / 59241] ###
        down_std_gene_edge = torch.nn.init.calculate_gain('relu')
        self.down_gene_edge_weight = torch.nn.Parameter((torch.randn(self.num_gene_edge) * down_std_gene_edge).to(device))


    def forward(self, x, edge_index):
        # [batch_size]
        batch_size = int(x.shape[0] / self.node_num)
        # TEST PARAMETERS
        # import pdb; pdb.set_trace()
        print(torch.sum(self.up_gene_edge_weight))
        print(torch.sum(self.down_gene_edge_weight))

        ### [edge_index, x] ###
        up_edge_index = edge_index
        up_x = self.up_proj(x)
        down_edge_index = torch.flipud(edge_index)
        down_x = self.down_proj(x)
        bias_x = self.bias_proj(x)

        ### [edge_weight] ###
        # [up_edge_weight] = [up_gene_edge_weight] + [up_drug_edge_weight] [59241+214=59455]
        up_drug_edge_weight = torch.ones(self.num_drug_edge).to(device='cuda') # [/107*2=214]
        up_edge_weight = torch.cat((self.up_gene_edge_weight, up_drug_edge_weight), 0)
        # [down_edge_weight] = [down_gene_edge_weight] + [down_drug_edge_weight] [59241+214=59455]
        down_drug_edge_weight = torch.ones(self.num_drug_edge).to(device='cuda')
        down_edge_weight = torch.cat((self.down_gene_edge_weight, down_drug_edge_weight), 0) # [/107*2=214]
        # [batch_up/down_edge_weight] [N*59455]
        batch_up_edge_weight = up_edge_weight.repeat(1, batch_size)
        batch_down_edge_weight = down_edge_weight.repeat(1, batch_size)

        # Step 3: Compute normalization.
        # [row] FOR 1st LINE && [col] FOR 2nd LINE
        # [up]
        up_row, up_col = up_edge_index
        up_deg = degree(up_col, x.size(0), dtype=x.dtype)
        up_deg_inv_sqrt = up_deg.pow(-1)
        up_norm = up_deg_inv_sqrt[up_col]
        # [down]
        down_row, down_col = down_edge_index
        down_deg = degree(down_col, x.size(0), dtype=x.dtype)
        down_deg_inv_sqrt = down_deg.pow(-1)
        down_norm = down_deg_inv_sqrt[down_col]
        # Check [ norm[0:59241] == norm[59455:118696] ]

        # Step 4-5: Start propagating messages.
        x_up = self.propagate(up_edge_index, x=up_x, norm=up_norm, edge_weight=batch_up_edge_weight)
        x_down = self.propagate(down_edge_index, x=down_x, norm=down_norm, edge_weight=batch_down_edge_weight)
        x_bias = bias_x
        # import pdb; pdb.set_trace()
        concat_x = torch.cat((x_up, x_down, x_bias), dim=-1)
        concat_x = F.normalize(concat_x, p=2, dim=-1)
        return concat_x

    def message(self, x_j, norm, edge_weight):
        # [x_j] has shape [E, out_channels]
        # import pdb; pdb.set_trace()
        # Step 4: Normalize node features.
        weight_norm = torch.mul(norm, edge_weight)
        return weight_norm.view(-1, 1) * x_j

    def update(self, aggr_out):
        # [aggr_out] has shape [N, out_channels]
        # import pdb; pdb.set_trace()
        return aggr_out


class WeBInDecoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, embedding_dim, decoder_dim, node_num, num_edge, num_gene_edge, device):
        super(WeBInDecoder, self).__init__()
        self.device = device
        drug_input_dim, rna_input_dim, cmeth_input_dim, cnv_input_dim, mut_input_dim = input_dim
        drug_hidden_dim, rna_hidden_dim, cmeth_hidden_dim, cnv_hidden_dim, mut_hidden_dim = hidden_dim
        drug_embedding_dim, rna_embedding_dim, cmeth_embedding_dim, cnv_embedding_dim, mut_embedding_dim = embedding_dim
        ### BUILD UP INCEPTION GRAPH
        # [drug]
        self.drug_conv_first, self.drug_conv_block, self.drug_conv_last = self.build_conv_layer(
                    input_dim=drug_input_dim, hidden_dim=drug_hidden_dim, embedding_dim=drug_embedding_dim,
                    node_num, num_edge, num_gene_edge)
        self.drug_proj = torch.nn.Linear((drug_embedding_dim*3), (drug_embedding_dim*3), bias=True)
        # [rna]
        self.rna_conv_first, self.rna_conv_block, self.rna_conv_last = self.build_conv_layer(
                    input_dim=rna_input_dim, hidden_dim=rna_hidden_dim, embedding_dim=rna_embedding_dim,
                    node_num, num_edge, num_gene_edge)
        self.rna_proj = torch.nn.Linear((rna_embedding_dim*3), (rna_embedding_dim*3), bias=True)
        # [cmeth]
        self.cmeth_conv_first, self.cmeth_conv_block, self.cmeth_conv_last = self.build_conv_layer(
                    input_dim=cmeth_input_dim, hidden_dim=cmeth_hidden_dim, embedding_dim=cmeth_embedding_dim,
                    node_num, num_edge, num_gene_edge)
        self.cmeth_proj = torch.nn.Linear((cmeth_embedding_dim*3), (cmeth_embedding_dim*3), bias=True)
        # [cnv]
        self.cnv_conv_first, self.cnv_conv_block, self.cnv_conv_last = self.build_conv_layer(
                    input_dim=cnv_input_dim, hidden_dim=cnv_hidden_dim, embedding_dim=cnv_embedding_dim,
                    node_num, num_edge, num_gene_edge)
        self.cnv_proj = torch.nn.Linear((cnv_embedding_dim*3), (cnv_embedding_dim*3), bias=True)
        # [mut]
        self.mut_conv_first, self.mut_conv_block, self.mut_conv_last = self.build_conv_layer(
                    input_dim=mut_input_dim, hidden_dim=mut_hidden_dim, embedding_dim=mut_embedding_dim,
                    node_num, num_edge, num_gene_edge)
        self.mut_proj = torch.nn.Linear((mut_embedding_dim*3), (mut_embedding_dim*3), bias=True)
        
        self.act = nn.ReLU()
        self.act2 = nn.LeakyReLU(negative_slope = 0.1)

        self.parameter1 = torch.nn.Parameter(torch.randn(int(embedding_dim*3), decoder_dim).to(device='cuda'))
        self.parameter2 = torch.nn.Parameter(torch.randn(decoder_dim, decoder_dim).to(device='cuda'))

    def build_conv_layer(self, input_dim, hidden_dim, embedding_dim, node_num, num_edge, num_gene_edge):
        # conv_first [input_dim, hidden_dim]
        conv_first = WeBGNNConv(in_channels=input_dim, out_channels=hidden_dim,
                node_num=node_num, num_edge=num_edge, num_gene_edge=num_gene_edge, device=self.device)
        # conv_block [hidden_dim, hidden_dim/3]
        conv_block = WeBGNNConv(in_channels=int(hidden_dim*3), out_channels=hidden_dim,
                node_num=node_num, num_edge=num_edge, num_gene_edge=num_gene_edge, device=self.device)
        # conv_last [hidden_dim, embedding_dim]
        conv_last = WeBGNNConv(in_channels=int(hidden_dim*3), out_channels=embedding_dim,
                node_num=node_num, num_edge=num_edge, num_gene_edge=num_gene_edge, device=self.device)
        return conv_first, conv_block, conv_last

    def forward(self, x, edge_index, drug_index, label):
        ### DECOMPOSE FEATURE [x] => [Drug, RNA, CMeth, CNV, MUT(AMP,DEL)]
        import pdb; pdb.set_trace()
        x_drug = torch.reshape(x[:,0:2], (x.shape[0], 2))
        x_rna = torch.reshape(x[:,2], (x.shape[0], 1))
        x_cmeth = torch.reshape(x[:,3], (x.shape[0], 1))
        x_cnv = torch.reshape(x[:,4], (x.shape[0], 1))
        x_mut = torch.reshape(x[:,5:7], (x.shape[0], 2))
        # DRUG INCEPTION
        x_drug = self.drug_conv_first(x_drug, edge_index)
        x_drug = self.act2(x_drug)
        x_drug = self.drug_conv_block(x_drug, edge_index)
        x_drug = self.act2(x_drug)
        x_drug = self.drug_conv_last(x_drug, edge_index)
        x_drug = self.act2(x_drug)
        x_drug = self.drug_proj(x_drug)
        # RNA INCEPTION
        x_rna = self.rna_conv_first(x_rna, edge_index)
        x_rna = self.act2(x_rna)
        x_rna = self.rna_conv_block(x_rna, edge_index)
        x_rna = self.act2(x_rna)
        x_rna = self.rna_conv_last(x_rna, edge_index)
        x_rna = self.act2(x_rna)
        x_rna = self.rna_proj(x_rna)
        # CMETH INCEPTION
        x_cmeth = self.cmeth_conv_first(x_cmeth, edge_index)
        x_cmeth = self.act2(x_cmeth)
        x_cmeth = self.cmeth_conv_block(x_cmeth, edge_index)
        x_cmeth = self.act2(x_cmeth)
        x_cmeth = self.cmeth_conv_last(x_cmeth, edge_index)
        x_cmeth = self.act2(x_cmeth)
        x_cmeth = self.cmeth_proj(x_cmeth)
        # CNV INCEPTION
        x_cnv = self.cnv_conv_first(x_cnv, edge_index)
        x_cnv = self.act2(x_cnv)
        x_cnv = self.cnv_conv_block(x_cnv, edge_index)
        x_cnv = self.act2(x_cnv)
        x_cnv = self.cnv_conv_last(x_cnv, edge_index)
        x_cnv = self.act2(x_cnv)
        x_cnv = self.cnv_proj(x_cnv)
        # MUT INCEPTION
        x_mut = self.mut_conv_first(x_mut, edge_index)
        x_mut = self.act2(x_mut)
        x_mut = self.mut_conv_block(x_mut, edge_index)
        x_mut = self.act2(x_mut)
        x_mut = self.mut_conv_last(x_mut, edge_index)
        x_mut = self.act2(x_mut)
        x_mut = self.mut_proj(x_mut)
        ### CONCAT ALL PARTS
        x = torch.cat([x_drug, x_rna, x_cmeth, x_cnv, x_mut], dim=0)

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
        print(self.parameter1)
        print(torch.sum(self.parameter1))
        # print(self.parameter2)
        return ypred

    def loss(self, pred, label):
        pred = pred.to(device='cuda')
        label = label.to(device='cuda')
        loss = F.mse_loss(pred.squeeze(), label)
        # print(pred)
        # print(label)
        return loss