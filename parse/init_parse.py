import os
import re
import numpy as np
import pandas as pd

from numpy import savetxt
from sklearn.model_selection import train_test_split

class ReadFile():
    def __init__(self, dir_opt):
        self.dir_opt = dir_opt

    def combo_input(self):
        # # # INTIALIZE [NCI60 DrugScreen Data]
        print('----- READING NCI60 DRUG SCREEN RAW DATA -----')
        if os.path.exists('../datainfo/init_data') == False:
            os.mkdir('../datainfo/init_data')
        dl_input_df = pd.read_csv('../datainfo/aa_raw_data/NCI60/DeepLearningInput.csv')
        dl_input_df = dl_input_df.groupby(['Drug A', 'Drug B', 'Cell Line Name']).agg({'Score':'mean'}).reset_index()
        # REMOVE SINGLE DRUG SCREEN DATA [Actual Fact Shows No Single Drug]
        dl_input_deletion_list = []
        dl_input_df = dl_input_df.fillna('missing')
        for row in dl_input_df.itertuples():
            if row[1] == 'missing' or row[2] == 'missing':
                dl_input_deletion_list.append(row[0])
        dl_input_df = dl_input_df.drop(dl_input_df.index[dl_input_deletion_list]).reset_index(drop=True)
        dl_input_df.to_csv('../datainfo/init_data/DeepLearningInput.csv', index=False, header=True)
        # # # PROFILE [Number of Drugs / Number of Cell Lines]
        drug_list = list(set(list(dl_input_df['Drug A']) + list(dl_input_df['Drug B'])))
        cell_line_list = list(set(list(dl_input_df['Cell Line Name'])))
        print('----- NUMBER OF DRUGS IN NCI ALMANAC: ' + str(len(drug_list)) + ' -----')
        print('----- NUMBER OF CELL LINES IN NCI ALMANAC: ' + str(len(cell_line_list)) + ' -----')
        print(dl_input_df.shape)
    
    def gdsc_rnaseq(self):
        # # # INTIALIZE [GDSC RNA Sequence Data]
        print('----- READING GDSC RNA Sequence RAW DATA -----')
        dir_opt = self.dir_opt
        rna_df = pd.read_csv('../datainfo/aa_raw_data/GDSC/rnaseq_20191101/rnaseq_fpkm_20191101.csv', low_memory=False)
        rna_df = rna_df.fillna('missing')
        rna_df.to_csv('../datainfo/init_data/gdsc_rnaseq.csv', index=False, header=True)
        print(rna_df.shape)
        # AFTER THIS NEED SOME MANUAL OPERATIONS TO CHANGE COLUMNS AND ROWS NAMES

    def gdsc_cnv(self):
        dir_opt = self.dir_opt
        cnv_df = pd.read_csv('../datainfo/aa_raw_data/GDSC/cnv_20191101/cnv_gistic_20191101.csv', low_memory=False)
        cnv_df = cnv_df.fillna('missing')
        cnv_df.to_csv('../datainfo/init_data/gdsc_cnv.csv', index = False, header = True)
        print(cnv_df.shape)
        # AFTER THIS NEED SOME MANUAL OPERATIONS TO CHANGE COLUMNS AND ROWS NAMES

    def ccle_meth(self):
        ccle_meth1_df = pd.read_table('../datainfo/aa_raw_data/CCLE/meth/CCLE_RRBS_TSS1kb_20181022.txt', delimiter='\t')
        ccle_meth1_df = ccle_meth1_df.replace('    NaN', np.nan)
        ccle_meth1_df = ccle_meth1_df.replace('     NA', np.nan)
        ccle_meth1_df.drop(ccle_meth1_df.tail(1).index, inplace=True)
        print(ccle_meth1_df)
        # REPLACE ALL [locus_id] WITH GENE NAMES
        ccle_meth1_gene_dict = {}
        for row in ccle_meth1_df.itertuples():
            gene = row[1].split('_')[0]
            ccle_meth1_gene_dict[row[1]] = gene
        ccle_meth1_df = ccle_meth1_df.replace({'locus_id': ccle_meth1_gene_dict})
        # REMOVE CERTAIN [CpG_sites_hg19, avg_coverage] COLUMNS
        ccle_meth1_df = ccle_meth1_df.drop(columns=['CpG_sites_hg19', 'avg_coverage'])
        ccle_meth1_df = ccle_meth1_df.sort_values(by=['locus_id']).reset_index(drop=True)
        # FETCH THE MAXIMUM VALUE WITH REPEATED GENE NAMES
        ccle_meth1_df = ccle_meth1_df.groupby('locus_id').agg('max').reset_index()
        print(ccle_meth1_df)
        # # REPLACE ALL WITH FIRST NAME IN CELL LINES
        # ccle_meth1_df = pd.read_table('./datainfo/mid_data/ccle_methylation.txt', delimiter = ',')
        ccle_meth1_cell_line_dict = {}
        ccle_meth_oricell_line = list(ccle_meth1_df.columns)[1:]
        for oricell_line in ccle_meth_oricell_line:
            cell_line = oricell_line.split('_')[0]
            ccle_meth1_cell_line_dict[oricell_line] = cell_line
        ccle_meth1_df = ccle_meth1_df.rename(columns=ccle_meth1_cell_line_dict)
        ccle_meth1_df = ccle_meth1_df.fillna('missing')
        ccle_meth1_df = ccle_meth1_df.loc[:, (ccle_meth1_df != 'missing').any(axis=0)]
        print(ccle_meth1_df)
        # FINALLY [17180 GENES, 833 CELL LINES]
        ccle_meth1_df = ccle_meth1_df.replace('missing', 0.0)
        ccle_meth1_df.to_csv('../datainfo/init_data/ccle_methylation.csv', index=False, header=True)

    def ccle_mutation(self):
        ccle_mutation_df = pd.read_csv('../datainfo/aa_raw_data/CCLE/mutation/BinaryCalls/CCLE_MUT_CNA_AMP_DEL_binary_Revealer.txt', delimiter = '\t')
        ccle_mutation_df = ccle_mutation_df.drop(columns=['Description'])
        # print(ccle_mutation_df)
        # COMBINE TWO LISTS [TT_OESOPHAGUS, TT_THYROID]
        tt_oeso = list(ccle_mutation_df['TT_OESOPHAGUS'])
        tt_thy = list(ccle_mutation_df['TT_THYROID'])
        tt_list = []
        for (oeso, thy) in zip(tt_oeso, tt_thy):
            if oeso == 1 or thy == 1: tt_list.append(1)
            else: tt_list.append(0)
        ccle_mutation_df = ccle_mutation_df.drop(columns = ['TT_OESOPHAGUS', 'TT_THYROID'])
        ccle_mutation_df.insert(1, 'TT', tt_list)
        print(ccle_mutation_df)
        # NAME LIST OF DIFFERENT TYPES
        all_list = []
        mut_list = []
        del_list = []
        amp_list = []
        # INDEX LIST OF DIFFERENT TYPES
        mut_idx_list = []
        del_idx_list = []
        amp_idx_list = []
        for row in ccle_mutation_df.itertuples():
            gene_info = row[1].split('_')
            gene = gene_info[0]
            if '.' in gene: 
                gene = gene.split('.')[0]
            if 'MUT' in gene_info[-1]: 
                mut_list.append(gene)
            if 'DEL' in gene_info[-1]:
                del_list.append(gene)
                del_idx_list.append(row[0])
            if 'AMP' in gene_info[-1]:
                amp_list.append(gene)
                amp_idx_list.append(row[0])
            if gene not in all_list:
                all_list.append(gene)
        # # # REMOVE SUFFIX [del]
        ccle_mutation_del_df = ccle_mutation_df.loc[del_idx_list].sort_values(by = ['Name']).reset_index(drop = True)
        # GENE LIST
        ccle_mutation_del_gene_list = list(ccle_mutation_del_df['Name'])
        ccle_mutation_del_gene_maplist = [gene.split('_')[0].upper() for gene in ccle_mutation_del_gene_list]
        ccle_mutation_del_gene_dict = dict(zip(ccle_mutation_del_gene_list, ccle_mutation_del_gene_maplist))
        ccle_mutation_del_df = ccle_mutation_del_df.replace({'Name': ccle_mutation_del_gene_dict})
        # CELL LINE LIST
        ccle_mutation_del_cell_line = list(ccle_mutation_del_df.columns)[1:]
        ccle_mutation_del_cell_maplist = [cell_line.split('_')[0] for cell_line in ccle_mutation_del_cell_line]
        ccle_mutation_del_cell_dict = dict(zip(ccle_mutation_del_cell_line, ccle_mutation_del_cell_maplist))
        ccle_mutation_del_df = ccle_mutation_del_df.rename(columns = ccle_mutation_del_cell_dict) 
        ccle_mutation_del_df.to_csv('../datainfo/init_data/ccle_mutation_del.csv', index = False, header = True)
        print(ccle_mutation_del_df)
        # # # REMOVE SUFFIX [amp]
        # GENE LIST
        ccle_mutation_amp_df = ccle_mutation_df.loc[amp_idx_list].sort_values(by = ['Name']).reset_index(drop = True)
        ccle_mutation_amp_gene_list = list(ccle_mutation_amp_df['Name'])
        ccle_mutation_amp_gene_maplist = [gene.split('_')[0].upper() for gene in ccle_mutation_amp_gene_list]
        ccle_mutation_amp_gene_dict = dict(zip(ccle_mutation_amp_gene_list, ccle_mutation_amp_gene_maplist))
        ccle_mutation_amp_df = ccle_mutation_amp_df.replace({'Name': ccle_mutation_amp_gene_dict})
        # CELL LINE LIST
        ccle_mutation_amp_cell_line = list(ccle_mutation_amp_df.columns)[1:]
        ccle_mutation_amp_cell_maplist = [cell_line.split('_')[0] for cell_line in ccle_mutation_amp_cell_line]
        ccle_mutation_amp_cell_dict = dict(zip(ccle_mutation_amp_cell_line, ccle_mutation_amp_cell_maplist))
        ccle_mutation_amp_df = ccle_mutation_amp_df.rename(columns = ccle_mutation_amp_cell_dict) 
        ccle_mutation_amp_df.to_csv('../datainfo/init_data/ccle_mutation_amp.csv', index = False, header = True)
        print(ccle_mutation_amp_df)

    def kegg(self):
        dir_opt = self.dir_opt
        kegg_df = pd.read_table('../datainfo/aa_raw_data/KEGG/All_Kegg_edges.txt', delimiter='\t')
        src_list = list(kegg_df['src'])
        dest_list = list(kegg_df['dest'])
        # ADJUST ALL GENES TO UPPERCASE
        up_src_list = []
        for src in src_list:
            up_src = src.upper()
            up_src_list.append(up_src)
        up_dest_list = []
        for dest in dest_list:
            up_dest = dest.upper()
            up_dest_list.append(up_dest)
        up_kegg_conn_dict = {'src': up_src_list, 'dest': up_dest_list}
        up_kegg_df = pd.DataFrame(up_kegg_conn_dict)
        up_kegg_df.to_csv('../datainfo/init_data/up_kegg.csv', index=False, header=True)
        kegg_gene_list = list(set(src_list + dest_list))
        print('----- NUMBER OF GENES IN KEGG: ' + str(len(kegg_gene_list)) + ' -----')
        print(up_kegg_df.shape)

    def biogrid(self):
        biogrid_df = pd.read_table('../datainfo/aa_raw_data/BioGrid/BIOGRID-ALL-3.5.174.mitab.Symbol.txt', delimiter = '\t')
        eh_list = list(biogrid_df['e_h'])
        et_list = list(biogrid_df['e_t'])
        # ADJUST ALL GENES TO UPPERCASE
        up_eh_list = []
        for eh in eh_list:
            up_eh = eh.upper()
            up_eh_list.append(up_eh)
        up_et_list = []
        for et in et_list:
            up_et = et.upper()
            up_et_list.append(up_et)
        up_biogrid_conn_dict = {'e_h': up_eh_list, 'e_t': up_et_list}
        up_biogrid_df = pd.DataFrame(up_biogrid_conn_dict)
        print(up_biogrid_df)
        up_biogrid_df.to_csv('../datainfo/init_data/up_biogrid.csv', index = False, header = True)

    def string(self):
        dir_opt = self.dir_opt
        string_df = pd.read_csv('../datainfo/aa_raw_data/STRING/9606.protein.links.detailed.v11.0_sym.csv', low_memory=False)
        src_list = list(string_df['Source'])
        tar_list = list(string_df['Target'])
        # ADJUST ALL GENES TO UPPERCASE
        up_src_list = []
        for src in src_list:
            up_src = src.upper()
            up_src_list.append(up_src)
        up_tar_list = []
        for tar in tar_list:
            up_tar = tar.upper()
            up_tar_list.append(up_tar)
        up_string_conn_dict = {'Source': up_src_list, 'Target': up_tar_list}
        up_string_df = pd.DataFrame(up_string_conn_dict)
        print(up_string_df)
        up_string_df.to_csv('../datainfo/init_data/up_string.csv', index = False, header = True)

    def drugbank(self):
        # INITIALIZE THE DRUG BANK INTO [.csv] FILE
        drugbank_df = pd.read_table('../datainfo/aa_raw_data/DrugBank/drug_tar_drugBank_all.txt', delimiter='\t')
        drug_list = list(set(list(drugbank_df['Drug'])))
        target_gene_list = list(set(list(drugbank_df['Target'])))
        print('----- NUMBER OF DRUGS IN DrugBank: ' + str(len(drug_list)) + ' -----')
        print('----- NUMBER OF GENES IN DrugBank: ' + str(len(target_gene_list)) + ' -----')
        drugbank_df.to_csv('../datainfo/init_data/drugbank.csv', index=False, header=True)



def init_parse():
    dir_opt = '/datainfo'
    # ReadFile(dir_opt).combo_input()
    # ReadFile(dir_opt).gdsc_rnaseq()
    # ReadFile(dir_opt).gdsc_cnv()
    # ReadFile(dir_opt).ccle_meth()
    ReadFile(dir_opt).ccle_mutation()
    # ReadFile(dir_opt).kegg()
    # ReadFile(dir_opt).biogrid()
    # ReadFile(dir_opt).string()
    # ReadFile(dir_opt).drugbank()

if __name__ == "__main__":
    init_parse()