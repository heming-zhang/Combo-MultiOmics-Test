import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
from torch_geometric.nn.inits import zeros

class GINConv(MessagePassing):
    def __init__(self, in_channels, out_channels, eps=0.01, train_eps=False):
        super(GINConv, self).__init__(aggr='add')
        self.mlp = torch.nn.Sequential(torch.nn.Linear(in_channels, 2 * in_channels), torch.nn.ReLU(),
                                       torch.nn.Linear(2 * in_channels, out_channels))
        self.initial_eps = eps
        if train_eps:
            self.eps = torch.nn.Parameter(torch.Tensor([eps]))
        else:
            self.register_buffer('eps', torch.Tensor([eps]))
        self.reset_parameters()

    def weights_init(self,m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight.data)
            zeros(m.bias.data)

    def reset_parameters(self):
        self.mlp.apply(self.weights_init)
        self.eps.data.fill_(self.initial_eps)

    def forward(self, x, edge_index):
        x_n = self.propagate(edge_index, x=x, edge_attr=None)
        return self.mlp((1 + self.eps) * x + x_n)

    def message(self, x_j):
        return x_j

    def update(self, aggr_out):
        return aggr_out


class GINDecoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, embedding_dim, decoder_dim, node_num):
        super(GINDecoder, self).__init__()
        self.node_num = node_num
        self.embedding_dim = embedding_dim
        self.conv_first, self.conv_block, self.conv_last = self.build_conv_layer(
                    input_dim, hidden_dim, embedding_dim)
        
        self.act = nn.ReLU()
        self.act2 = nn.LeakyReLU(negative_slope = 0.1)

        self.parameter1 = torch.nn.Parameter(torch.randn(embedding_dim, decoder_dim).to(device='cuda'))
        self.parameter2 = torch.nn.Parameter(torch.randn(decoder_dim, decoder_dim).to(device='cuda'))

    def build_conv_layer(self, input_dim, hidden_dim, embedding_dim):
        conv_first = GINConv(in_channels=input_dim, out_channels=hidden_dim)
        conv_block = GINConv(in_channels=hidden_dim, out_channels=hidden_dim)
        conv_last = GINConv(in_channels=hidden_dim, out_channels=embedding_dim)
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
        return ypred

    def loss(self, pred, label):
        pred = pred.to(device='cuda')
        label = label.to(device='cuda')
        loss = F.mse_loss(pred.squeeze(), label)
        # print(pred)
        # print(label)
        return loss