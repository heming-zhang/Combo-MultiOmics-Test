import os
import re
import numpy as np
import pandas as pd

from lxml import etree
from pubchempy import *


'''
Pin the number of cell lines by following tables:
(1) DeepLearningInput.csv
(2) GDSC RNA-Seq
(3) GDSC CNV
(4) CCLE Meth
'''
class CellLineAnnotation():
    def __init__(self):
        pass

    def parse_cell_xml(self):
        # # # INTIALIZE THE CELL LINE ANNOTATION
        cellosaurus_doc = etree.parse('../datainfo/aa_raw_data/Annotation/cellosaurus.xml')
        count = 0
        accession_list = []
        identifier_list = []
        synonym_list = []
        species_list = []
        # KEEP ONLY ['Homo Sapiens'] CELL LINE
        for att in cellosaurus_doc.xpath('//cell-line'):
            count += 1
            print('-------------', count)
            accession = att.xpath('.//accession[@type="primary"]/text()')[0]
            identifier = att.xpath('.//name[@type="identifier"]/text()')[0]
            synonym = att.xpath('.//name[@type="synonym"]/text()')
            species = att.xpath('.//cv-term[@terminology="NCBI-Taxonomy"]/text()')
            if 'Homo sapiens' not in species:
                print(species)
                continue
            accession_list.append(accession)
            identifier_list.append(identifier)
            synonym_list.append(synonym)
            species_list.append(species)
        cell_annotation_df = pd.DataFrame({'Accession': accession_list,
                                           'Identifier': identifier_list,
                                           'Synonym': synonym_list,
                                           'Species': species_list})
        cell_annotation_df.to_csv('../datainfo/init_data/cell_annotation.csv', index=False, header=True)

    def dl_cell_annotation(self):
        # # # REPLACE ALL CELL LINE NAME WITH [Accession]
        cell_annotation_df = pd.read_csv('../datainfo/init_data/cell_annotation_manual.csv', keep_default_na=False)
        # # # [DeepLearningInput.csv]
        dl_input_df = pd.read_csv('../datainfo/init_data/DeepLearningInput.csv')
        dl_input_cell_line_list = sorted(list(set(list(dl_input_df['Cell Line Name']))))
        dl_cell_accession_list = []
        dl_cell_identifier_list = []
        dl_cell_synonym_list = []
        for dl_cell in dl_input_cell_line_list:
            for row in cell_annotation_df.itertuples():
                if (dl_cell == row.Identifier) or (dl_cell in eval(row.Synonym)):
                    dl_cell_accession_list.append(row.Accession)
                    dl_cell_identifier_list.append(row.Identifier)
                    dl_cell_synonym_list.append(eval(row.Synonym))
        dl_cell_annotation_df = pd.DataFrame({'dl_cell': dl_input_cell_line_list,
                                              'Accession': dl_cell_accession_list,
                                              'Identifier': dl_cell_identifier_list,
                                              'Synonym': dl_cell_synonym_list})
        dl_cell_annotation_df.to_csv('../datainfo/init_data/dl_cell_annotation.csv', index=False, header=True)
        
    def omics_cell(self):
        # # # [dl_cell_annotation]
        dl_cell_annotation_df = pd.read_csv('../datainfo/init_data/dl_cell_annotation.csv', keep_default_na=False)
        # # # [CCLE Methylation]
        cmeth_df = pd.read_csv('../datainfo/init_data/ccle_methylation.csv', low_memory=False)
        cmeth_cell_line_list = sorted(list(cmeth_df.columns)[1:])
        cmeth_selected_cell_list = []
        cmeth_cell_accession_list = []
        for cmeth_cell in cmeth_cell_line_list:
            for row in dl_cell_annotation_df.itertuples():
                if (cmeth_cell == row.Identifier) or (cmeth_cell in eval(row.Synonym)):
                    cmeth_selected_cell_list.append(cmeth_cell)
                    cmeth_cell_accession_list.append(row.Accession)
        cmeth_cell_accession_df = pd.DataFrame({'cmeth_cell': cmeth_selected_cell_list,
                                                'Accession': cmeth_cell_accession_list})  
        dl_cmeth_cell_annotation_df = pd.merge(dl_cell_annotation_df, cmeth_cell_accession_df, \
                            how='left', left_on='Accession', right_on='Accession')
        dl_cmeth_cell_annotation_df = dl_cmeth_cell_annotation_df.dropna().reset_index(drop=True)
        # # # [GDSC RNASeq]
        rna_df = pd.read_csv('../datainfo/init_data/gdsc_rnaseq_manual.csv', low_memory=False)
        rna_cell_line_list = sorted(list(rna_df.columns)[1:])
        rna_selected_cell_list = []
        rna_cell_accession_list = []
        for rna_cell in rna_cell_line_list:
            for row in dl_cmeth_cell_annotation_df.itertuples():
                if (rna_cell == row.Identifier) or (rna_cell in eval(row.Synonym)):
                    rna_selected_cell_list.append(rna_cell)
                    rna_cell_accession_list.append(row.Accession)
        rna_cell_accession_df = pd.DataFrame({'rna_cell': rna_selected_cell_list,
                                              'Accession': rna_cell_accession_list})  
        dl_cmeth_rna_cell_annotation_df = pd.merge(dl_cmeth_cell_annotation_df, rna_cell_accession_df, \
                            how='left', left_on='Accession', right_on='Accession')
        dl_cmeth_rna_cell_annotation_df = dl_cmeth_rna_cell_annotation_df.dropna().reset_index(drop=True)
        # # # [GDSC CNV]
        cnv_df = pd.read_csv('../datainfo/init_data/gdsc_cnv_manual.csv', low_memory=False)
        cnv_cell_line_list = sorted(list(cnv_df.columns)[1:])
        cnv_selected_cell_list = []
        cnv_cell_accession_list = []
        for cnv_cell in cnv_cell_line_list:
            for row in dl_cmeth_cell_annotation_df.itertuples():
                if (cnv_cell == row.Identifier) or (cnv_cell in eval(row.Synonym)):
                    cnv_selected_cell_list.append(cnv_cell)
                    cnv_cell_accession_list.append(row.Accession)
        cnv_cell_accession_df = pd.DataFrame({'cnv_cell': cnv_selected_cell_list,
                                              'Accession': cnv_cell_accession_list})
        dl_cmeth_rna_cnv_cell_annotation_df = pd.merge(dl_cmeth_rna_cell_annotation_df, cnv_cell_accession_df, \
                            how='left', left_on='Accession', right_on='Accession')
        dl_cmeth_rna_cnv_cell_annotation_df = dl_cmeth_rna_cnv_cell_annotation_df.dropna().reset_index(drop=True)
        dl_cmeth_rna_cnv_cell_annotation_df.to_csv('../datainfo/init_data/omics_cell_annotation.csv', index=False, header=True)  
        print(dl_cmeth_rna_cnv_cell_annotation_df)

    def tail_cell(self):
        # READ [omics_cell_annotation]
        omics_cell_df = pd.read_csv('../datainfo/init_data/omics_cell_annotation.csv')
        # TAIL [NCI ALMANAC] CELL LINEs
        new_dl_input_cell_line_list = list(omics_cell_df['dl_cell'])
        dl_input_df = pd.read_csv('../datainfo/init_data/DeepLearningInput.csv')
        tail_cell_dl_input_df = dl_input_df[dl_input_df['Cell Line Name'].isin(new_dl_input_cell_line_list)].reset_index(drop=True)
        tail_cell_dl_input_df.to_csv('../datainfo/mid_cell_line/tail_cell_dl_input.csv', index=False, header=True)
        # TAIL [GDSC RNA-Seq] CELL LINEs // KEEP [RNA-Seq] CELL LINE NAMES CONSISTENT WITH [NCI ALMANAC]
        omics_rna_cell_line_list = list(omics_cell_df['rna_cell'])
        rna_df = pd.read_csv('../datainfo/init_data/gdsc_rnaseq_manual.csv', low_memory=False)
        tail_cell_rna_df = rna_df[omics_rna_cell_line_list]
        nci_rna_cell_line_dict = dict(zip(omics_cell_df.rna_cell, omics_cell_df.dl_cell))
        tail_cell_rna_df = tail_cell_rna_df.rename(columns=nci_rna_cell_line_dict)
        tail_cell_rna_df.insert(0, 'symbol', list(rna_df['symbol']))
        tail_cell_rna_df.to_csv('../datainfo/mid_cell_line/tail_cell_rna.csv', index=False, header=True)
        # TAIL [GDSC CNV] CELL LINEs // KEEP [CNV] CELL LINE NAMES CONSISTENT WITH [NCI ALMANAC]
        omics_cnv_cell_line_list = list(omics_cell_df['cnv_cell'])
        cnv_df = pd.read_csv('../datainfo/init_data/gdsc_cnv_manual.csv', low_memory=False)
        tail_cell_cnv_df = cnv_df[omics_cnv_cell_line_list]
        nci_cnv_cell_line_dict = dict(zip(omics_cell_df.cnv_cell, omics_cell_df.dl_cell))
        tail_cell_cnv_df = tail_cell_cnv_df.rename(columns=nci_cnv_cell_line_dict)
        tail_cell_cnv_df.insert(0, 'symbol', list(cnv_df['symbol']))
        tail_cell_cnv_df = tail_cell_cnv_df.reset_index(drop=True)
        print(tail_cell_cnv_df)
        tail_cell_cnv_df.to_csv('../datainfo/mid_cell_line/tail_cell_cnv.csv', index=False, header=True)
        # TAIL [CCLE Methylation] // KEEP [CMeth] CELL LINE NAMES CONSISTENT WITH [NCI ALMANAC]
        omics_cmeth_cell_line_list = list(omics_cell_df['cmeth_cell'])
        cmeth_df = pd.read_csv('../datainfo/init_data/ccle_methylation.csv', low_memory=False)
        tail_cell_cmeth_df = cmeth_df[omics_cmeth_cell_line_list]
        nci_cmeth_cell_line_dict = dict(zip(omics_cell_df.cmeth_cell, omics_cell_df.dl_cell))
        tail_cell_cmeth_df = tail_cell_cmeth_df.rename(columns=nci_cmeth_cell_line_dict)
        tail_cell_cmeth_df.insert(0, 'locus_id', list(cmeth_df['locus_id']))
        tail_cell_cmeth_df.to_csv('../datainfo/mid_cell_line/tail_cell_cmeth.csv', index=False, header=True)


'''
Pin the number of genes from following intersection of tables:
(1) KEGG ['src', 'dest']
(2) GDSC RNASeq ['symbol']
(3) DrugBank ['Target']
'''
class GeneAnnotation():
    def __init__(self):
        pass

    def gdsc_cnv_tail_overzero(self):
        tail_cell_cnv_df = pd.read_csv('../datainfo/mid_cell_line/tail_cell_cnv.csv')
        tail_cell_cnv_gene_deletion_list = [row[0] for row in tail_cell_cnv_df.itertuples() if list(row[2:]).count('missing')>4]
        print(len(tail_cell_cnv_gene_deletion_list))

    def kegg_omics_intersect(self):
        # # GET [8002] KEGG GENES
        kegg_df = pd.read_csv('../datainfo/init_data/up_kegg.csv')
        kegg_gene_list = sorted(list(set(list(kegg_df['src']) + list(kegg_df['dest']))))
        # # GET INTERSETED GENES
        tail_cell_rna_df = pd.read_csv('../datainfo/mid_cell_line/tail_cell_rna.csv', low_memory=False)
        tail_cell_rna_df = tail_cell_rna_df.sort_values(by=['symbol'])
        rna_gene_list = list(tail_cell_rna_df['symbol'])
        tail_cell_cnv_df = pd.read_csv('../datainfo/mid_cell_line/tail_cell_cnv.csv', low_memory=False)
        tail_cell_cnv_df = tail_cell_cnv_df.sort_values(by=['symbol'])
        cnv_gene_list = list(tail_cell_cnv_df['symbol'])
        tail_cell_cmeth_df = pd.read_csv('../datainfo/mid_cell_line/tail_cell_cmeth.csv', low_memory=False)
        tail_cell_cmeth_df = tail_cell_cmeth_df.sort_values(by=['locus_id'])
        cmeth_gene_list = list(tail_cell_cmeth_df['locus_id'])
        omics_gene_set = set(rna_gene_list).intersection(set(cnv_gene_list)).intersection(set(cmeth_gene_list))
        omics_gene_list = sorted(list(omics_gene_set))
        # # # LEFT JOIN TO AUTO MAP
        kegg_gene_df = pd.DataFrame(data=kegg_gene_list, columns=['kegg_gene'])
        omics_gene_df = pd.DataFrame(data=omics_gene_list, columns=['omics_gene'])
        kegg_omics_gene_df = pd.merge(kegg_gene_df, omics_gene_df, how='left', left_on='kegg_gene', right_on='omics_gene')
        kegg_omics_gene_df = kegg_omics_gene_df.dropna().reset_index(drop=True) # [8002] => [4099] GENEs
        # TAIL [KEGG] GENE => [3066] GENES
        kegg_gene_deletion_list = [gene for gene in kegg_gene_list if gene not in list(kegg_omics_gene_df['kegg_gene'])]
        kegg_gene_deletion_index = [row[0] for row in kegg_df.itertuples() \
                                    if row[1] in kegg_gene_deletion_list or row[2] in kegg_gene_deletion_list]
        new_kegg_df = kegg_df.drop(kegg_gene_deletion_index).reset_index(drop=True)
        new_kegg_df = new_kegg_df.sort_values(by=['src', 'dest'])
        new_kegg_df.to_csv('../datainfo/filtered_data/kegg_gene_interaction.csv', index=False, header=True)
        new_kegg_gene_list = sorted(list(set(list(new_kegg_df['src']) + list(new_kegg_df['dest']))))
        new_kegg_gene_df = pd.DataFrame(data=new_kegg_gene_list, columns=['kegg_gene'])
        new_kegg_gene_df.to_csv('../datainfo/filtered_data/kegg_gene_annotation.csv', index=False, header=True)
        # TAIL [GDSC RNASeq] => [3066] GENES
        tail_cell_gene_rna_df = tail_cell_rna_df[tail_cell_rna_df['symbol'].isin(new_kegg_gene_list)].reset_index(drop=True)
        tail_cell_gene_rna_df.to_csv('../datainfo/mid_gene/tail_cell_gene_rna.csv', index=False, header=True)
        print(tail_cell_gene_rna_df)
        # TAIL [GDSC CNV] => [3066] GENES
        tail_cell_gene_cnv_df = tail_cell_cnv_df[tail_cell_cnv_df['symbol'].isin(new_kegg_gene_list)].reset_index(drop=True)
        tail_cell_gene_cnv_df.to_csv('../datainfo/mid_gene/tail_cell_gene_cnv.csv', index=False, header=True)
        print(tail_cell_gene_cnv_df)
        # TAIL [CCLE Methylation] => [3066] GENES
        tail_cell_gene_cmeth_df = tail_cell_cmeth_df[tail_cell_cmeth_df['locus_id'].isin(new_kegg_gene_list)].reset_index(drop=True)
        tail_cell_gene_cmeth_df.to_csv('../datainfo/mid_gene/tail_cell_gene_cmeth.csv', index=False, header=True)
        print(tail_cell_gene_cmeth_df)

    def kegg_drugbank_gene_intersect(self):
        # TAIL [DrugBank] GENE
        kegg_gene_annotation_df = pd.read_csv('../datainfo/filtered_data/kegg_gene_annotation.csv')
        kegg_gene_list = list(kegg_gene_annotation_df['kegg_gene'])
        drugbank_df = pd.read_csv('../datainfo/init_data/drugbank.csv')
        new_drugbank_df = drugbank_df[drugbank_df['Target'].isin(kegg_gene_list)].reset_index(drop=True)
        new_drugbank_df.to_csv('../datainfo/mid_gene/tail_gene_drugbank.csv', index=False, header=True)


'''
Pin the number of drug from following tables:
(1) tail_gene_drugbank.csv
(2) DeepLearningInput.csv
'''
class DrugAnnotation():
    def __init__(self):
        pass

    # def pubchem_cid_dl_drug(self):
    #     tail_cell_dl_input_df = pd.read_csv('../datainfo/mid_cell_line/tail_cell_dl_input.csv')
    #     tail_cell_dl_input_drug_list = sorted(list(set(list(tail_cell_dl_input_df['Drug A']) + list(tail_cell_dl_input_df['Drug B']))))
    #     tail_cell_dl_input_selected_drug_list = []
    #     tail_cell_dl_input_drug_cid_list = []
    #     for drug in tail_cell_dl_input_drug_list:
    #         if get_cids(drug) ==[]: continue
    #         else: 
    #             tail_cell_dl_input_selected_drug_list.append(drug)
    #             tail_cell_dl_input_drug_cid_list.append('cid'+str(get_cids(drug)[0]))
    #     dl_input_cid_df = pd.DataFrame({'dl_input': tail_cell_dl_input_selected_drug_list, 'dl_input_cid': tail_cell_dl_input_drug_cid_list})
    #     dl_input_cid_df.to_csv('../datainfo/mid_drug/dl_input_cid.csv', index=False, header=True)

    # def pubchem_cid_drugbank_drug(self):
    #     tail_gene_drugbank_df = pd.read_csv('../datainfo/mid_gene/tail_gene_drugbank.csv')
    #     tail_gene_drugbank_drug_list = sorted(list(set(list(tail_gene_drugbank_df['Drug']))))
    #     tail_gene_drugbank_selected_drug_list = []
    #     tail_gene_drugbank_drug_cid_list = []
    #     for drug in tail_gene_drugbank_drug_list:
    #         if get_cids(drug) ==[]: continue
    #         else: 
    #             tail_gene_drugbank_selected_drug_list.append(drug)
    #             tail_gene_drugbank_drug_cid_list.append('cid'+str(get_cids(drug)[0]))
    #     drugbank_cid_df = pd.DataFrame({'drugbank': tail_gene_drugbank_selected_drug_list, 'drugbank_cid': tail_gene_drugbank_drug_cid_list})
    #     drugbank_cid_df.to_csv('../datainfo/mid_drug/drugbank_cid.csv', index=False, header=True)

    # def nci_drugbank_cid_intersect(self):
    #     dl_input_cid_df = pd.read_csv('../datainfo/mid_drug/dl_input_cid.csv')
    #     drugbank_cid_df = pd.read_csv('../datainfo/mid_drug/drugbank_cid.csv')
    #     nci_drugbank_df = pd.merge(dl_input_cid_df, drugbank_cid_df, how='left', left_on='dl_input_cid', right_on='drugbank_cid')
    #     nci_drugbank_df = nci_drugbank_df.dropna().reset_index(drop=True)
    #     print(nci_drugbank_df)

    def nci_drugbank_drug_intersect(self):
        tail_cell_dl_input_df = pd.read_csv('../datainfo/mid_cell_line/tail_cell_dl_input.csv')
        tail_cell_dl_input_drug_list = sorted(list(set(list(tail_cell_dl_input_df['Drug A']) + list(tail_cell_dl_input_df['Drug B']))))
        tail_gene_drugbank_df = pd.read_csv('../datainfo/mid_gene/tail_gene_drugbank.csv')
        tail_gene_drugbank_drug_list = sorted(list(set(list(tail_gene_drugbank_df['Drug']))))
        # # # LEFT JOIN [NCI ALMANAC]
        dl_input_drug_uprmv_list = [drug.replace('-', '').replace('_', '').upper() for drug in tail_cell_dl_input_drug_list]
        dl_input_drug_uprmv_df = pd.DataFrame({'input_drug': tail_cell_dl_input_drug_list, 'input_uprmv': dl_input_drug_uprmv_list})
        drugbank_uprmv_druglist = []
        for drug in tail_gene_drugbank_drug_list:
            tmp_drug = drug.replace('-', '')
            tmp_drug = tmp_drug.replace('(', '').replace(')', '')
            tmp_drug = tmp_drug.replace('[', '').replace(']', '')
            tmp_drug = tmp_drug.replace('{', '').replace('}', '')
            tmp_drug = tmp_drug.replace(',', '').upper()
            drugbank_uprmv_druglist.append(tmp_drug)
        drugbank_uprmv_drug_df = pd.DataFrame({'drugbank_uprmv': drugbank_uprmv_druglist, 'drugbank_drug': tail_gene_drugbank_drug_list})
        nci_drugbank_drug_df = pd.merge(dl_input_drug_uprmv_df, drugbank_uprmv_drug_df, how='left', left_on='input_uprmv', right_on='drugbank_uprmv')
        nci_drugbank_drug_df = nci_drugbank_drug_df.dropna().reset_index(drop=True)
        # TAIL [NCI ALMANAC] DRUGS
        dl_input_drug_deletion_list = list(set([drug for drug in tail_cell_dl_input_drug_list if drug not in list(nci_drugbank_drug_df['input_drug'])]))
        dl_input_drug_deletion_index = [row[0] for row in tail_cell_dl_input_df.itertuples() \
                                    if row[1] in dl_input_drug_deletion_list or row[2] in dl_input_drug_deletion_list]
        tail_cell_drug_dl_input_df = tail_cell_dl_input_df.drop(dl_input_drug_deletion_index).reset_index(drop=True)
        tail_cell_drug_dl_input_df.to_csv('../datainfo/mid_drug/tail_cell_drug_dl_input.csv', index=False, header=True)
        # TAIL [DrugBank] DRUGS // REPLACE [DrugBank] DRUGs CONSISTENT WITH [NCI ALMANAC]
        tail_gene_drug_drugbank_df = tail_gene_drugbank_df[tail_gene_drugbank_df['Drug']\
                                    .isin(list(nci_drugbank_drug_df['drugbank_drug']))].reset_index(drop=True)
        nci_drugbank_drug_dict = dict(zip(nci_drugbank_drug_df.drugbank_drug, nci_drugbank_drug_df.input_drug))
        tail_gene_drug_drugbank_df = tail_gene_drug_drugbank_df.replace({'Drug': nci_drugbank_drug_dict})
        tail_gene_drug_drugbank_df.to_csv('../datainfo/mid_drug/tail_gene_drug_drugbank.csv', index=False, header=True)
        # print(nci_drugbank_drug_df)

'''
Recheck the number of cell lines after tailing drugs in NCI ALMANAC:
(1) tail_cell_drug_dl_input.csv
(2) tail_cell_gene_rna.csv
'''
class RecheckFinal():
    def __init__(self):
        pass
    
    def recheck_cell_line(self):
        # TAILED [DeepLearning]
        tail_cell_drug_dl_input_df = pd.read_csv('../datainfo/mid_drug/tail_cell_drug_dl_input.csv')
        tail_cell_drug_dl_input_cell_list = list(set(list(tail_cell_drug_dl_input_df['Cell Line Name'])))
        # TAILED [GDSC RNA_seq]
        tail_cell_gene_rna_df = pd.read_csv('../datainfo/mid_gene/tail_cell_gene_rna.csv')
        tail_cell_gene_rna_cell_list = tail_cell_gene_rna_df.columns[1:]
        recheck_cell_line = [cell_line for cell_line in tail_cell_gene_rna_cell_list if cell_line not in tail_cell_drug_dl_input_cell_list]
        if len(recheck_cell_line)==0 : print('NO MORE CHECK NEEDED')

    def final(self):
        # [DeepLearningInput]
        tail_cell_drug_dl_input_df = pd.read_csv('../datainfo/mid_drug/tail_cell_drug_dl_input.csv')
        # tail_cell_drug_dl_input_df['Score'] = (tail_cell_drug_dl_input_df['Score'] - tail_cell_drug_dl_input_df['Score'].mean()) / tail_cell_drug_dl_input_df['Score'].std()    
        tail_cell_drug_dl_input_df.to_csv('../datainfo/filtered_data/final_dl_input.csv', index=False, header=True)
        # [RNA-Seq]
        tail_cell_gene_rna_df = pd.read_csv('../datainfo/mid_gene/tail_cell_gene_rna.csv')
        tail_cell_gene_rna_df = tail_cell_gene_rna_df.replace(['missing'], 0.0)
        tail_cell_gene_rna_df.to_csv('../datainfo/filtered_data/final_rna.csv', index=False, header=True)
        # [CNV]
        tail_cell_gene_cnv_df = pd.read_csv('../datainfo/mid_gene/tail_cell_gene_cnv.csv')
        tail_cell_gene_cnv_df.to_csv('../datainfo/filtered_data/final_cnv.csv', index=False, header=True)
        # [Methylation]
        tail_cell_gene_cmeth_df = pd.read_csv('../datainfo/mid_gene/tail_cell_gene_cmeth.csv')
        # print(tail_cell_gene_cmeth_df.isnull().values.any())
        # tail_cell_gene_cmeth_df = tail_cell_gene_cmeth_df.replace(['missing'], 0.0)
        tail_cell_gene_cmeth_df.to_csv('../datainfo/filtered_data/final_cmeth.csv', index=False, header=True)
        # [DrugBank]
        tail_gene_drug_drugbank_df = pd.read_csv('../datainfo/mid_drug/tail_gene_drug_drugbank.csv')
        tail_gene_drug_drugbank_df = tail_gene_drug_drugbank_df.sort_values(by=['Drug', 'Target'])
        tail_gene_drug_drugbank_df.to_csv('../datainfo/filtered_data/final_drugbank.csv', index=False, header=True)



if os.path.exists('../datainfo/mid_gene') == False:
    os.mkdir('../datainfo/mid_gene')
if os.path.exists('../datainfo/mid_cell_line') == False:
    os.mkdir('../datainfo/mid_cell_line')
if os.path.exists('../datainfo/mid_drug') == False:
    os.mkdir('../datainfo/mid_drug')
if os.path.exists('../datainfo/filtered_data') == False:
    os.mkdir('../datainfo/filtered_data')


# CellLineAnnotation().parse_cell_xml()
# CellLineAnnotation().dl_cell_annotation()
# CellLineAnnotation().omics_cell()
# CellLineAnnotation().tail_cell()

# GeneAnnotation().gdsc_cnv_tail_overzero()
# GeneAnnotation().kegg_omics_intersect()
# GeneAnnotation().kegg_drugbank_gene_intersect()

# DrugAnnotation().pubchem_cid_dl_drug()
# DrugAnnotation().pubchem_cid_drugbank_drug()
# DrugAnnotation().nci_drugbank_cid_intersect()
# DrugAnnotation().nci_drugbank_drug_intersect()

# RecheckFinal().recheck_cell_line()
RecheckFinal().final()