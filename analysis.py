import os
import pdb
import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from matplotlib.pyplot import figure
from sklearn.preprocessing import normalize
from matplotlib.ticker import PercentFormatter

class NetAnalyse():
    def __init__(self):
        pass

    def prepare_network(self, drug_gene_edge_weight):
        ### GET [node_num_dict] FOR WHOLE NET NODES
        kegg_gene_num_dict_df = pd.read_csv('./datainfo/filtered_data/kegg_gene_num_dict.csv')
        drug_num_dict_df = pd.read_csv('./datainfo/filtered_data/drug_num_dict.csv')
        kegg_gene_num_dict_df = kegg_gene_num_dict_df.rename(columns={'kegg_gene': 'node_name', 'gene_num': 'node_num'})
        kegg_gene_num_dict_df['node_type'] = ['gene'] * kegg_gene_num_dict_df.shape[0]
        drug_num_dict_df = drug_num_dict_df.rename(columns={'Drug': 'node_name', 'drug_num': 'node_num'})
        drug_num_dict_df['node_type'] = ['drug'] * drug_num_dict_df.shape[0]
        node_num_dict_df = pd.concat([kegg_gene_num_dict_df, drug_num_dict_df])
        node_num_dict_df = node_num_dict_df[['node_num', 'node_name', 'node_type']]
        node_num_dict_df.to_csv('./datainfo/analysis_data/node_num_dict.csv', index=False, header=True)
        ### GET [drug-gene] EDGES
        final_drugbank_num_df = pd.read_csv('./datainfo/filtered_data/final_drugbank_num.csv')
        final_drugbank_num_df = final_drugbank_num_df.rename(columns={'Drug': 'src', 'Target': 'dest'})
        final_drugbank_num_df['src_name'] = list(final_drugbank_num_df['src'])
        final_drugbank_num_df['dest_name'] = list(final_drugbank_num_df['dest'])
        final_drugbank_num_df['weight'] = [drug_gene_edge_weight] * (final_drugbank_num_df.shape[0])
        node_dict = dict(zip(list(node_num_dict_df['node_num']), list(node_num_dict_df['node_name'])))
        final_drugbank_edge_df = final_drugbank_num_df.replace({'src_name': node_dict, 'dest_name': node_dict})
        final_drugbank_edge_df = final_drugbank_edge_df[['src', 'src_name', 'dest', 'dest_name', 'weight']]
        final_drugbank_edge_df.to_csv('./datainfo/analysis_data/drug_edge_weight.csv', header=True, index=False)

    def average_layer_weight(self, conv_edge_weight_list, net_type):
        ### AVERAGE [edge_weight] 
        kegg_gene_interaction_df = pd.read_csv('./datainfo/filtered_data/kegg_gene_num_interaction.csv')
        num_edge = kegg_gene_interaction_df.shape[0]
        mean_conv_edge_weight = np.zeros((num_edge, 1))
        for edge_weight in conv_edge_weight_list: 
            abs_edge_weight = np.absolute(edge_weight.cpu().data.numpy()).reshape((num_edge, 1))
            absnorm_edge_weight = normalize(abs_edge_weight, norm='max', axis=0)
            mean_conv_edge_weight += absnorm_edge_weight
            # print(net_type, torch.sum(edge_weight))
        mean_conv_edge_weight = mean_conv_edge_weight / len(conv_edge_weight_list)
        ### SAVE [edge_weight] INTO [kegg_gene_interaction]
        gene_edge_weight_df = kegg_gene_interaction_df
        gene_edge_weight_df['src_name'] = list(gene_edge_weight_df['src'])
        gene_edge_weight_df['dest_name'] = list(gene_edge_weight_df['dest'])
        gene_edge_weight_df['weight'] = mean_conv_edge_weight.reshape(num_edge,).tolist()
        kegg_gene_num_dict_df = pd.read_csv('./datainfo/filtered_data/kegg_gene_num_dict.csv')
        gene_dict = dict(zip(list(kegg_gene_num_dict_df['gene_num']), list(kegg_gene_num_dict_df['kegg_gene'])))
        gene_edge_weight_df = gene_edge_weight_df.replace({'src_name': gene_dict, 'dest_name': gene_dict})
        gene_edge_weight_df = gene_edge_weight_df[['src', 'src_name', 'dest', 'dest_name', 'weight']]
        gene_edge_weight_df.to_csv('./datainfo/analysis_data/gene_edge_weight_' + net_type + '.csv', header=True, index=False)
        ### COMBINE [gene-gene] AND [drug-gene] EDGES
        final_drugbank_edge_df = pd.read_csv('./datainfo/analysis_data/drug_edge_weight.csv')
        gene_edge_weight_df['edge_type'] = ['gene-gene'] * gene_edge_weight_df.shape[0]
        final_drugbank_edge_df['edge_type'] = ['drug-gene'] * (final_drugbank_edge_df.shape[0])
        network_edge_weight_df = pd.concat([gene_edge_weight_df, final_drugbank_edge_df])
        network_edge_weight_df.to_csv('./datainfo/analysis_data/network_edge_weight_' + net_type + '.csv', header=True, index=False)

    def load_param(self, file_path, device):
        ### LOAD PARAMETERS FROM SAVED TRAIN MODEL
        model_param = torch.load(file_path, map_location=device)
        all_conv_edge_weight_list = [param for name, param in model_param.items() if 'gene_edge_weight' in name]
        omics_conv_edge_weight_list = [param for name, param in model_param.items() if 'gene_edge_weight' in name and 'drug' not in name]
        rna_conv_edge_weight_list = [param for name, param in model_param.items() if 'gene_edge_weight' in name and 'rna' in name]
        cmeth_conv_edge_weight_list = [param for name, param in model_param.items() if 'gene_edge_weight' in name and 'cmeth' in name]
        cnv_conv_edge_weight_list = [param for name, param in model_param.items() if 'gene_edge_weight' in name and 'cnv' in name]
        mut_conv_edge_weight_list = [param for name, param in model_param.items() if 'gene_edge_weight' in name and 'mut' in name]
        ### AVERAGE [edge_weight] AND SAVE [edge_weight] INTO [kegg_gene_interaction]
        NetAnalyse().average_layer_weight(all_conv_edge_weight_list, net_type='all')
        NetAnalyse().average_layer_weight(omics_conv_edge_weight_list, net_type='omics')
        NetAnalyse().average_layer_weight(rna_conv_edge_weight_list, net_type='rna')
        NetAnalyse().average_layer_weight(cmeth_conv_edge_weight_list, net_type='cmeth')
        NetAnalyse().average_layer_weight(cnv_conv_edge_weight_list, net_type='cnv')
        NetAnalyse().average_layer_weight(mut_conv_edge_weight_list, net_type='mut')


    def filter_net(self, percentile):
        gene_edge_weight_df = pd.read_csv('./datainfo/analysis_data/gene_edge_weight.csv')
        mean_conv_edge_weight = np.array(gene_edge_weight_df['weight'])
        percentile_threshold = np.percentile(mean_conv_edge_weight, percentile)
        print(percentile_threshold)

    def net_threshold_stat(self):
        gene_edge_weight_df = pd.read_csv('./datainfo/analysis_data/gene_edge_weight.csv')
        ### HISTOGRAM
        gene_edge_weight_df = gene_edge_weight_df[gene_edge_weight_df['weight'] > 0.1]
        data = list(gene_edge_weight_df['weight'])
        plt.hist(data, weights = np.ones(len(data)) / len(data), bins=50)
        plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
        plt.savefig('./datainfo/plot/net_threshold_hist.png', dpi=300)
        ### PARETO PLOT
        melt_edge_weight_df = gene_edge_weight_df[['weight']]
        edge_weight_category=pd.cut(melt_edge_weight_df.weight,bins=[0.1,0.15,0.2,0.3,0.5,1.0],
                                labels=['<0.15','0.15~0.2','0.2~0.3','0.3~0.5','>0.5'])
        melt_edge_weight_df.insert(1,'edge_weight_category',edge_weight_category)
        df_pareto=melt_edge_weight_df.groupby(by=['edge_weight_category']).sum()
        df_pareto = df_pareto.sort_values(by='weight', ascending=False)
        df_pareto['cumperc'] = df_pareto['weight'].cumsum()/df_pareto['weight'].sum()*100
        fig, ax = plt.subplots()
        ax.bar(df_pareto.index, df_pareto['weight'])
        ax2 = ax.twinx()
        ax2.plot(df_pareto.index, df_pareto['cumperc'], color='red', marker="D", ms=4)
        ax2.yaxis.set_major_formatter(PercentFormatter())
        plt.savefig('./datainfo/plot/net_threshold_edge_weight_pareto.png', dpi=300)
        ### BOX PLOT
        melt_edge_weight_df.boxplot(column=['weight'])
        plt.savefig('./datainfo/plot/net_threshold_boxplot.png', dpi=300)


    def net_stat(self):
        gene_edge_weight_df = pd.read_csv('./datainfo/analysis_data/gene_edge_weight.csv')
        ### HISTOGRAM
        gene_edge_weight_df = gene_edge_weight_df[gene_edge_weight_df['weight'] > 0.1]
        data = list(gene_edge_weight_df['weight'])
        plt.hist(data, weights = np.ones(len(data)) / len(data))
        plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
        # hist = gene_edge_weight_df.hist(column=['weight'], bins=50, density=True)
        plt.savefig('./datainfo/plot/net_hist.png', dpi=300)
        gene_edge_weight_df['weight'].plot.kde(bw_method=0.3)
        plt.xscale('log')
        plt.title('Edge weight of network kde distribution')
        plt.savefig('./datainfo/plot/net_kde.png', dpi=300)
        plt.show()
        ### BOX PLOT
        gene_edge_weight_df.boxplot(column=['weight'])
        plt.savefig('./datainfo/plot/net_boxplot.png', dpi=300)
        ### PARETO PLOT
        melt_edge_weight_df = gene_edge_weight_df[['weight']]
        edge_weight_category=pd.cut(melt_edge_weight_df.weight,bins=[-0.01,0.001,0.01,0.1,0.15,0.2,1.0],
                                labels=['<0.001','0.001~0.01','0.01~0.1','0.1~0.15','0.15~0.2','>0.2'])
        melt_edge_weight_df.insert(1,'edge_weight_category',edge_weight_category)
        df_pareto=melt_edge_weight_df.groupby(by=['edge_weight_category']).sum()
        df_pareto = df_pareto.sort_values(by='weight', ascending=False)
        df_pareto['cumperc'] = df_pareto['weight'].cumsum()/df_pareto['weight'].sum()*100
        fig, ax = plt.subplots()
        ax.bar(df_pareto.index, df_pareto['weight'])
        ax2 = ax.twinx()
        ax2.plot(df_pareto.index, df_pareto['cumperc'], color='red', marker="D", ms=4)
        ax2.yaxis.set_major_formatter(PercentFormatter())
        plt.savefig('./datainfo/plot/net_edge_weight_pareto.png', dpi=300)


if __name__ == "__main__":
    file_path = './datainfo/result/webin/epoch_200_inc_all/best_train_model.pt'
    device = torch.device('cuda:0')
    if os.path.exists('./datainfo/analysis_data') == False:
        os.mkdir('./datainfo/analysis_data')
    NetAnalyse().prepare_network(drug_gene_edge_weight=0.5)
    NetAnalyse().load_param(file_path, device)
    # NetAnalyse().filter_net(percentile=98)
    # NetAnalyse().net_stat()
    # NetAnalyse().net_threshold_stat()
