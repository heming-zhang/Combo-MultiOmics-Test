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

    def load_param(self, file_path, device):
        ### LOAD PARAMETERS FROM SAVED TRAIN MODEL
        model_param = torch.load(file_path, map_location=device)
        for name, param in model_param.items():
            if 'gene_edge_weight' in name:
                print(name)
                print(param)
        import pdb; pdb.set_trace()
        conv_edge_weight_list = [param for name, param in model_param.items() if 'gene_edge_weight' in name]
        kegg_gene_interaction_df = pd.read_csv('./datainfo/filtered_data/kegg_gene_num_interaction.csv')
        num_edge = kegg_gene_interaction_df.shape[0]
        mean_conv_edge_weight = np.zeros((num_edge, 1))
        for edge_weight in conv_edge_weight_list: 
            abs_edge_weight = np.absolute(edge_weight.cpu().data.numpy()).reshape((num_edge, 1))
            absnorm_edge_weight = normalize(abs_edge_weight, norm='max', axis=0)
            mean_conv_edge_weight += absnorm_edge_weight
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
        gene_edge_weight_df.to_csv('./datainfo/analysis_data/gene_edge_weight.csv', header=True, index=False)

    def combine_net(self, drug_gene_edge_weight):
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
        ### GET [gene-gene] EDGES
        gene_edge_weight_df = pd.read_csv('./datainfo/analysis_data/gene_edge_weight.csv')
        gene_edge_weight_df['edge_type'] = ['gene-gene'] * gene_edge_weight_df.shape[0]
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
        final_drugbank_edge_df['edge_type'] = ['drug-gene'] * (final_drugbank_edge_df.shape[0])
        ### COMBINE [gene-gene, drug-gene] EDGES
        network_edge_weight_df = pd.concat([gene_edge_weight_df, final_drugbank_edge_df])
        network_edge_weight_df.to_csv('./datainfo/analysis_data/network_edge_weight.csv', header=True, index=False)

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

    
class ScoreAnalyse():
    def __init__(self):
        pass

    def pred_result(self):
        ### TRAIN PRED JOINTPLOT
        train_pred_df = pd.read_csv('./datainfo/result/epoch_50/TrainingPred_50.txt')
        sns.set_style('whitegrid')
        sns.jointplot(data=train_pred_df, x='Score', y='Pred Score', size=10, kind='reg')
        train_pearson = train_pred_df.corr(method='pearson')['Pred Score'][0]
        plt.legend(['Training Pearson =' + str(train_pearson)])
        plt.savefig('./datainfo/plot/trainpred_corr.png', dpi=300)
        ### TEST PRED JOINTPLOT
        test_pred_df = pd.read_csv('./datainfo/result/epoch_50/TestPred50.txt')
        comb_testpred_df = pd.read_csv('./datainfo/filtered_data/split_input_1.csv')
        comb_testpred_df['Pred Score'] = list(test_pred_df['Pred Score'])
        comb_testpred_df.to_csv('./datainfo/result/epoch_50/combine_testpred.csv', index=False, header=True)
        sns.set_style('whitegrid')
        sns.jointplot(data=comb_testpred_df, x='Score', y='Pred Score', size=10, kind='reg')
        test_pearson = test_pred_df.corr(method='pearson')['Pred Score'][0]
        plt.legend(['Test Pearson =' + str(test_pearson)])
        plt.savefig('./datainfo/plot/testpred_corr.png', dpi=300)
        ### HISTOGRAM
        hist = test_pred_df.hist(column=['Score', 'Pred Score'], bins=20)
        plt.savefig('./datainfo/plot/testpred_hist.png', dpi=300)
        ### BOX PLOT
        testpred_df = comb_testpred_df[['Cell Line Name', 'Pred Score']]
        testpred_df['Type'] = ['Prediction Score']*testpred_df.shape[0]
        testpred_df = testpred_df.rename(columns={'Pred Score': 'Drug Score'})
        test_df = comb_testpred_df[['Cell Line Name', 'Score']]
        test_df['Type'] = ['Input Score']*test_df.shape[0]
        test_df = test_df.rename(columns={'Score': 'Drug Score'})
        comb_score_df = pd.concat([testpred_df, test_df])
        a4_dims = (20, 15)
        fig, ax = plt.subplots(figsize=a4_dims)
        sns.set_context('paper')
        sns.boxplot(ax=ax, x='Cell Line Name', y='Drug Score', hue='Type', data=comb_score_df)
        plt.xticks(rotation = 45, ha = 'right')
        plt.savefig('./datainfo/plot/testpred_compare_cell_line_boxplot.png', dpi=600)
        plt.show()
        ### PARETO PLOT
        melt_testpred_input_df = comb_testpred_df[['Score']].abs()
        testpred_score_category=pd.cut(melt_testpred_input_df.Score,bins=[-0.1,10,20,30,40,50,150],
                                labels=['<10','10~-20','20~30','30~40','40~50','>50'])
        melt_testpred_input_df.insert(1,'testpred_score_category',testpred_score_category)
        df_pareto=melt_testpred_input_df.groupby(by=['testpred_score_category']).sum()
        df_pareto = df_pareto.sort_values(by='Score', ascending=False)
        df_pareto['cumperc'] = df_pareto['Score'].cumsum()/df_pareto['Score'].sum()*100
        fig, ax = plt.subplots()
        ax.bar(df_pareto.index, df_pareto['Score'])
        ax2 = ax.twinx()
        ax2.plot(df_pareto.index, df_pareto['cumperc'], color='red', marker="D", ms=4)
        ax2.yaxis.set_major_formatter(PercentFormatter())
        plt.savefig('./datainfo/plot/testpred_input_score_pareto.png', dpi=300)
        print(comb_testpred_df.describe())

class InputAnalyse():
    def __init__(self):
        pass

    def input_rna(self):
        final_rna_df = pd.read_csv('./datainfo/filtered_data/final_rna.csv')
        final_rna_df = final_rna_df.drop(columns=['symbol'])
        ### STRIP PLOT
        a4_dims = (15, 13)
        fig, ax = plt.subplots(figsize=a4_dims)
        sns.set_context('paper')
        sns.stripplot(ax=ax, x='variable', y='value', data=pd.melt(final_rna_df))
        plt.xticks(rotation = 45, ha = 'right')
        plt.savefig('./datainfo/plot/final_rna_stripplot.png', dpi=600)
        final_rna_des_df = final_rna_df.describe()
        final_rna_des_df.to_csv('./datainfo/plot/final_rna_descriptive.csv', header=True, index=False)
        ### PARETO PLOT
        melt_final_rna_df = pd.melt(final_rna_df)
        final_rna_category=pd.cut(melt_final_rna_df.value,bins=[-0.1,50,100,500,1000,10000],
                        labels=['<50','50-100','100-500','500-1000', '>1000'])
        melt_final_rna_df.insert(2,'rna_expression_category',final_rna_category)
        df_pareto=melt_final_rna_df.groupby(by=['rna_expression_category']).sum()
        df_pareto = df_pareto.sort_values(by='value', ascending=False)
        df_pareto['cumperc'] = df_pareto['value'].cumsum()/df_pareto['value'].sum()*100
        fig, ax = plt.subplots(figsize=(9,6))
        ax.bar(df_pareto.index, df_pareto['value'])
        ax2 = ax.twinx()
        ax2.plot(df_pareto.index, df_pareto['cumperc'], color='red', marker="D", ms=4)
        ax2.yaxis.set_major_formatter(PercentFormatter())
        ax2.set_xlabel('Gene Expression Value Range')
        ax2.set_ylabel('Accumulated Percent')
        plt.savefig('./datainfo/plot/final_rna_pareto.png', dpi=300)
        ## HISTOGRAM
        filter_melt_final_rna_df = melt_final_rna_df[melt_final_rna_df['value']<100]
        hist = filter_melt_final_rna_df.hist(column=['value'], bins=50, density=True)
        plt.xlabel('Gene Expression Value')
        plt.ylabel('Frequency')
        plt.savefig('./datainfo/plot/final_rna_hist.png', dpi=300)
        


    def input_cell_line(self):
        # ### HISTOGRAM
        dl_input_df = pd.read_csv('./datainfo/filtered_data/final_dl_input.csv')
        hist = dl_input_df.hist(column=['Score'], bins=20)
        plt.xlabel('Score')
        plt.ylabel('Frequency')
        plt.savefig('./datainfo/plot/dl_input_hist.png', dpi=300)
        ### BOX PLOT
        a4_dims = (15, 12)
        fig, ax = plt.subplots(figsize=a4_dims)
        sns.set_context('paper')
        sns.boxplot(ax=ax, x='Cell Line Name', y='Score', data=dl_input_df)
        plt.xticks(rotation = 45, ha = 'right')
        plt.savefig('./datainfo/plot/dl_input_cell_line_score_boxplot.png', dpi=600)
        ## PARETO PLOT
        melt_dl_input_df = dl_input_df[['Score']].abs()
        dl_score_category=pd.cut(melt_dl_input_df.Score,bins=[0.1,10,20,30,40,50,150],
                                labels=['<10','10~-20','20~30','30~40','40~50','>50'])
        melt_dl_input_df.insert(1,'dl_score_category',dl_score_category)
        df_pareto=melt_dl_input_df.groupby(by=['dl_score_category']).sum()
        df_pareto = df_pareto.sort_values(by='Score', ascending=False)
        df_pareto['cumperc'] = df_pareto['Score'].cumsum()/df_pareto['Score'].sum()*100
        fig, ax = plt.subplots(figsize=(9,6))
        ax.bar(df_pareto.index, df_pareto['Score'])
        ax2 = ax.twinx()
        ax2.plot(df_pareto.index, df_pareto['cumperc'], color='red', marker="D", ms=4)
        ax2.yaxis.set_major_formatter(PercentFormatter())
        ax2.set_xlabel('Score Value Range')
        ax2.set_ylabel('Accumulated Percent')
        plt.savefig('./datainfo/plot/dl_input_score_pareto.png', dpi=300)
        print(dl_input_df.describe())


if __name__ == "__main__":
    file_path = './datainfo/result/epoch_50/best_train_model.pt'
    device = torch.device('cuda:0')
    if os.path.exists('./datainfo/analysis_data') == False:
        os.mkdir('./datainfo/analysis_data')
    NetAnalyse().load_param(file_path, device)
    # NetAnalyse().combine_net(drug_gene_edge_weight=0.3)
    # NetAnalyse().filter_net(percentile=98)
    # NetAnalyse().net_stat()
    # NetAnalyse().net_threshold_stat()

    if os.path.exists('./datainfo/plot') == False:
        os.mkdir('./datainfo/plot')
    # ScoreAnalyse().pred_result()
    # InputAnalyse().input_rna()
    # InputAnalyse().input_cell_line()
