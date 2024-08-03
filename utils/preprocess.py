from copy import deepcopy
import numpy as np
import scipy.sparse
import networkx as nx
import pandas as pd
import dask.dataframe as dd
import time


def  get_instances_from_link_length2_v2(link, M, type_mask): # sparse version
    paths = []
    graph = scipy.sparse.csr_matrix(M.shape, dtype=bool)
    heads = (type_mask == link[0]).nonzero()[0]
    tails = (type_mask == link[-1]).nonzero()[0]
    # for tail in tails:
    #     mask[heads][:,tail] = True
    # for head in heads:
    #     mask[tails][:,head] = True
    # mask = np.logical_or(mask, tmp)
    # get link graph for link instances searching
    graph[min(heads):max(heads)+1,min(tails):max(tails)+1] = M[min(heads):max(heads)+1,min(tails):max(tails)+1]
    graph = graph + graph.T
    # graph = M.multiply(mask.tocsr()).astype(int)
    localtime = time.asctime(time.localtime(time.time()))
    print(f'{localtime}, finding pairs...')
    for head in heads:
        tails = graph[head].nonzero()[1]
        if len(tails) > 0:
            tmp = [[head, p] for p in tails]
        else: continue
        paths = paths + tmp
        localtime = time.asctime(time.localtime(time.time()))
        print('\r' + f"{localtime}, instances of {link} have been found, counts is {len(tmp)}.", end = '', flush = True)            

    return paths



def get_intances_by_pair(M, type_mask, ontology, prefix_operator):
    pass
    """
    :param M: the raw adjacency matrix
    :param type_mask: an array of types of all node
    :param ontology: a dict of decomposed ontology, which contains stem link, branch link and corresponding slave links.
    :return: a list of python dictionaries, consisting of corresponding link instances
    """
    outs = {}
    # get stem intances
    link = ontology['stem']
    # print('stem searching starts!')
    stem_instances = get_instances_from_link_by_pair(link, M, type_mask, prefix_operator)
    outs['stem'] = stem_instances
    
    
    # get branch intances
    if 'branch' in ontology.keys():
        branch_instances={} 
        for key in ontology['branch'].keys():
            link = ontology['branch'][key]
            # print(f'branch{key} searching starts!')
            branch_paths = get_instances_from_link_by_pair(link, M, type_mask, prefix_operator)
            branch_instances[key] = branch_paths
        outs['branch'] = branch_instances
                    
    
    # get stem_slave intances
    if 'stem_slave' in ontology.keys():
        stem_slave_instances=[]
        cnt = 0 
        for link in ontology['stem_slave']:
            # print(f'stem_slave{cnt} searching starts!')
            cnt += 1
            stem_slave_paths = get_instances_from_link_by_pair(link, M, type_mask, prefix_operator)
            stem_slave_instances.append(stem_slave_paths)
        outs['stem_slave'] = stem_slave_instances
    
    
    # get branch_slave intances
    if 'branch_slave' in ontology.keys():
        branch_slave_instances={}
        for key in ontology['branch_slave'].keys():
            tmp = []
            cnt = 0
            for link in ontology['branch_slave'][key]:
                # print(f'branch{key}slave{cnt} searching starts!')
                cnt += 1
                branch_slave_paths = get_instances_from_link_by_pair(link, M, type_mask, prefix_operator)  
                tmp.append(branch_slave_paths)
            branch_slave_instances[key] = tmp
        outs['branch_slave'] = branch_slave_instances
    
    return outs

def  get_instances_from_link_by_pair(link, M, type_mask, prefix_operator):
    pairs = []
    pair_instance = []
    for i in range(len(link )-1):
        pair = (link[i], link[i+1])
        pairs.append(pair)
        tmp = (M[prefix_operator[pair[0]]: prefix_operator[pair[0]+1], prefix_operator[pair[1]]: prefix_operator[pair[1]+1]] == 1).nonzero()
        tmp = np.stack((tmp[0] + prefix_operator[pair[0]], tmp[1] + prefix_operator[pair[1]])).T
        pair_instance.append(tmp)
        localtime = time.asctime(time.localtime(time.time()))
        # print('\r' + f"{localtime}, instances of {pair} have been found, counts is {len(tmp)}.", end = '', flush = True)

    # print('\nmerging path...\n')
    base = pairs[0]
    base_table = pd.DataFrame(pair_instance[0], columns = base)
    for pair, table in zip(pairs[1:],pair_instance[1:]):
        table = pd.DataFrame(table, columns = pair)
        base_table = base_table.join(table.set_index(pair[0]), on=pair[0], lsuffix='', rsuffix='_b', how='left').reset_index()
        base_table = base_table.dropna()
        if 'index' in base_table.columns:
            base_table.drop('index',axis=1,inplace=True)
        if sum(['_b' in str(p) for p in base_table.columns]):
            base_table.drop(str(pair[0] + '_b', axis = 1, inplace = True))
    
    base_table = base_table.drop_duplicates()

    return base_table.dropna().astype('int').values


def get_ontology_subgraphs(ontology, link_intances):
    branch_flag = 'branch' in ontology.keys()
    stem_slave_flag = 'stem_slave' in ontology.keys()
    branch_slave_flag = 'branch_slave' in ontology.keys()
    
    stem_df = pd.DataFrame(link_intances['stem'])
    subgraph = deepcopy(stem_df)
    
    switcher = [stem_slave_flag, branch_flag, branch_slave_flag]
    # all possible cases
    cases = np.array([[0,0,0],
                      [1,0,0],
                      [0,1,0],
                      [0,1,1],
                      [1,1,0],
                      [1,1,1]],dtype=bool)
    
    if (switcher == cases[0]).all():
        pass
    
    if (switcher == cases[1]).all():
        for i in range(len(link_intances['stem_slave'])):
            stem_slave_df = pd.DataFrame(link_intances['stem_slave'][i])
            left_head_idx = ontology['stem'].index(ontology['stem_slave'][i][0])
            left_tail_idx = ontology['stem'].index(ontology['stem_slave'][i][-1])
            right_tail_idx = len(ontology['stem_slave'][i]) - 1
            subgraph = subgraph.join(stem_slave_df.set_index([0,right_tail_idx]), lsuffix='', rsuffix='_ss'+str(i), on=[left_head_idx, left_tail_idx], how='left')
            subgraph.reset_index(inplace= True)
            if 'index' in subgraph.columns:
                subgraph.drop('index',axis=1,inplace=True)

    if (switcher == cases[2]).all():
        for key in link_intances['branch'].keys():
            branch_df = pd.DataFrame(link_intances['branch'][key])
            left_head_idx = ontology['stem'].index(ontology['branch'][key][0])
            subgraph = subgraph.join(branch_df.set_index(0), on = left_head_idx, lsuffix='', rsuffix='_b'+str(key), how='left' )
            subgraph.reset_index(inplace= True)
            if 'index' in subgraph.columns:
                subgraph.drop('index',axis=1,inplace=True)
    
    if (switcher == cases[3]).all():
        for key in link_intances['branch'].keys():
            branch_df = pd.DataFrame(link_intances['branch'][key])
            left_head_idx = ontology['stem'].index(ontology['branch'][key][0])

            for i in range(len(link_intances['branch_slave'][key])):
                branch_slave_df = pd.DataFrame(link_intances['branch_slave'][key][i])
                left_head_idx = ontology['branch'][key].index(ontology['branch_slave'][key][i][0])
                left_tail_idx = ontology['branch'][key].index(ontology['branch_slave'][key][i][-1])
                right_tail_idx = len(ontology['branch_slave'][key][i]) - 1
                branch_df = branch_df.join(branch_slave_df.set_index([0,right_tail_idx]), lsuffix='', rsuffix='_bs'+str(i), on=[left_head_idx, left_tail_idx], how='left')
                branch_df.reset_index(inplace= True)
                if 'index' in branch_df.columns:
                    branch_df.drop('index',axis=1,inplace=True)
                
            subgraph = subgraph.join(branch_df.set_index(0), on = left_head_idx, lsuffix='', rsuffix='_b'+str(key), how='left' )
            subgraph.reset_index(inplace= True)
            if 'index' in subgraph.columns:
                subgraph.drop('index',axis=1,inplace=True)
                
    
    if (switcher == cases[4]).all():
        for i in range(len(link_intances['stem_slave'])):
            stem_slave_df = pd.DataFrame(link_intances['stem_slave'][i])
            left_head_idx = ontology['stem'].index(ontology['stem_slave'][i][0])
            left_tail_idx = ontology['stem'].index(ontology['stem_slave'][i][-1])
            right_tail_idx = len(ontology['stem_slave'][i]) - 1
            subgraph = subgraph.join(stem_slave_df.set_index([0,right_tail_idx]), lsuffix='', rsuffix='_ss'+str(i), on=[left_head_idx, left_tail_idx], how='left')
            subgraph.reset_index(inplace= True)
            if 'index' in subgraph.columns:
                subgraph.drop('index',axis=1,inplace=True)
        for key in link_intances['branch'].keys():
            branch_df = pd.DataFrame(link_intances['branch'][key])
            left_head_idx = ontology['stem'].index(ontology['branch'][key][0])
            subgraph = subgraph.join(branch_df.set_index(0), on = left_head_idx, lsuffix='', rsuffix='_b'+str(key), how='left' )
            subgraph.reset_index(inplace= True)
            if 'index' in subgraph.columns:
                subgraph.drop('index',axis=1,inplace=True)

    
    if (switcher == cases[5]).all():
        for i in range(len(link_intances['stem_slave'])):
            stem_slave_df = pd.DataFrame(link_intances['stem_slave'][i])
            left_head_idx = ontology['stem'].index(ontology['stem_slave'][i][0])
            left_tail_idx = ontology['stem'].index(ontology['stem_slave'][i][-1])
            right_tail_idx = len(ontology['stem_slave'][i]) - 1
            subgraph = subgraph.join(stem_slave_df.set_index([0,right_tail_idx]), lsuffix='', rsuffix='_ss'+str(i), on=[left_head_idx, left_tail_idx], how='left')
            subgraph.reset_index(inplace= True)
            if 'index' in subgraph.columns:
                subgraph.drop('index',axis=1,inplace=True)
                
        for key in link_intances['branch'].keys():
            branch_df = pd.DataFrame(link_intances['branch'][key])
            left_head_idx = ontology['stem'].index(ontology['branch'][key][0])

            for i in range(len(link_intances['branch_slave'][key])):
                branch_slave_df = pd.DataFrame(link_intances['branch_slave'][key][i])
                left_head_idx = ontology['branch'][key].index(ontology['branch_slave'][key][i][0])
                left_tail_idx = ontology['branch'][key].index(ontology['branch_slave'][key][i][-1])
                right_tail_idx = len(ontology['branch_slave'][key][i]) - 1
                branch_df = branch_df.join(branch_slave_df.set_index([0,right_tail_idx]), lsuffix='', rsuffix='_bs'+str(i), on=[left_head_idx, left_tail_idx], how='left')
                branch_df.reset_index(inplace= True)
                if 'index' in branch_df.columns:
                    branch_df.drop('index',axis=1,inplace=True)
                
            subgraph = subgraph.join(branch_df.set_index(0), on = left_head_idx, lsuffix='', rsuffix='_b'+str(key), how='left' )
            subgraph.reset_index(inplace= True)
            if 'index' in subgraph.columns:
                subgraph.drop('index',axis=1,inplace=True)
    
    subgraph = subgraph.T.drop_duplicates().T
    subgraph = subgraph.dropna()
    subgraph = subgraph
    return subgraph
    # for row in subgraph.values:
    #     ontology_subgraphs.append(row2grahp(M,row))   
                
    # return ontology_subgraphs,subgraph


def get_ontology_subgraphs_v2(ontology, link_intances):
    branch_flag = 'branch' in ontology.keys()
    stem_slave_flag = 'stem_slave' in ontology.keys()
    branch_slave_flag = 'branch_slave' in ontology.keys()
    
    stem_df = pd.DataFrame(link_intances['stem'], columns = ontology['stem'])
    subgraph = deepcopy(stem_df)
    
    switcher = [stem_slave_flag, branch_flag, branch_slave_flag]
    # all possible cases
    cases = np.array([[0,0,0],
                      [1,0,0],
                      [0,1,0],
                      [0,1,1],
                      [1,1,0],
                      [1,1,1]],dtype=bool)
    
    if (switcher == cases[0]).all():
        pass
    
    if (switcher == cases[1]).all():
        for i in range(len(link_intances['stem_slave'])):
            stem_slave_df = pd.DataFrame(link_intances['stem_slave'][i], columns = ontology['stem_slave'][i])
            head_idx = ontology['stem_slave'][i][0]
            tail_idx = ontology['stem_slave'][i][-1]
            subgraph = subgraph.join(stem_slave_df.set_index([head_idx,tail_idx]), lsuffix='', rsuffix='_ss'+str(i), on=[head_idx, tail_idx], how='left')
            subgraph.reset_index(inplace= True)
            if 'index' in subgraph.columns:
                subgraph.drop('index',axis=1,inplace=True)

    if (switcher == cases[2]).all():
        for key in link_intances['branch'].keys():
            branch_df = pd.DataFrame(link_intances['branch'][key], columns = ontology['branch'][key])
            head_idx = ontology['branch'][key][0]
            subgraph = subgraph.join(branch_df.set_index(head_idx), on = head_idx, lsuffix='', rsuffix='_b'+str(key), how='left' )
            subgraph.reset_index(inplace= True)
            if 'index' in subgraph.columns:
                subgraph.drop('index',axis=1,inplace=True)
    
    if (switcher == cases[3]).all():
        for key in link_intances['branch'].keys():
            branch_df = pd.DataFrame(link_intances['branch'][key], columns = ontology['branch'][key])
            branch_head_idx = ontology['branch'][key][0]

            for i in range(len(link_intances['branch_slave'][key])):
                branch_slave_df = pd.DataFrame(link_intances['branch_slave'][key][i], columns = ontology['branch_slave'][key][i])
                head_idx = ontology['branch_slave'][key][0]
                tail_idx = ontology['branch_slave'][key][-1]
                branch_df = branch_df.join(branch_slave_df.set_index([head_idx,tail_idx]), lsuffix='', rsuffix='_bs'+str(i), on=[head_idx, tail_idx], how='left')
                branch_df.reset_index(inplace= True)
                if 'index' in branch_df.columns:
                    branch_df.drop('index',axis=1,inplace=True)
                
            subgraph = subgraph.join(branch_df.set_index(branch_head_idx), on = branch_head_idx, lsuffix='', rsuffix='_b'+str(key), how='left' )
            subgraph.reset_index(inplace= True)
            if 'index' in subgraph.columns:
                subgraph.drop('index',axis=1,inplace=True)
                
    
    if (switcher == cases[4]).all():
        for i in range(len(link_intances['stem_slave'])):
            stem_slave_df = pd.DataFrame(link_intances['stem_slave'][i], columns = ontology['stem_slave'][i])
            head_idx = ontology['stem_slave'][i][0]
            tail_idx = ontology['stem_slave'][i][-1]
            subgraph = subgraph.join(stem_slave_df.set_index([head_idx,tail_idx]), lsuffix='', rsuffix='_ss'+str(i), on=[head_idx, tail_idx], how='left')
            subgraph.reset_index(inplace= True)
            if 'index' in subgraph.columns:
                subgraph.drop('index',axis=1,inplace=True)
            subgraph = subgraph.dropna()
        for key in link_intances['branch'].keys():
            if (len(ontology['branch'][key]) == 2) & (ontology['branch'][key][0] == ontology['branch'][key][1]): #selfloop
                branch_df = pd.DataFrame(link_intances['branch'][key], columns = [ontology['branch'][key][0],str(ontology['branch'][key][1])])
            else:branch_df = pd.DataFrame(link_intances['branch'][key], columns = ontology['branch'][key])
            branch_head_idx = ontology['branch'][key][0]
            subgraph = subgraph.join(branch_df.set_index(branch_head_idx), on = branch_head_idx, lsuffix='', rsuffix='_b'+str(key), how='left' )
            subgraph.reset_index(inplace= True)
            if 'index' in subgraph.columns:
                subgraph.drop('index',axis=1,inplace=True)

    
    if (switcher == cases[5]).all():
        for i in range(len(link_intances['stem_slave'])):
            stem_slave_df = pd.DataFrame(link_intances['stem_slave'][i], columns = ontology['stem_slave'][i])
            head_idx = ontology['stem_slave'][i][0]
            tail_idx = ontology['stem_slave'][i][-1]
            subgraph = subgraph.join(stem_slave_df.set_index([head_idx,tail_idx]), lsuffix='', rsuffix='_ss'+str(i), on=[head_idx, tail_idx], how='left')
            subgraph.reset_index(inplace= True)
            if 'index' in subgraph.columns:
                subgraph.drop('index',axis=1,inplace=True)
                
        for key in link_intances['branch'].keys():
            branch_df = pd.DataFrame(link_intances['branch'][key], columns = ontology['branch'][key])
            branch_head_idx = ontology['branch'][key][0]

            for i in range(len(link_intances['branch_slave'][key])):
                branch_slave_df = pd.DataFrame(link_intances['branch_slave'][key][i], columns = ontology['branch_slave'][key][i])
                head_idx = ontology['branch_slave'][key][0]
                tail_idx = ontology['branch_slave'][key][-1]
                branch_df = branch_df.join(branch_slave_df.set_index([head_idx,tail_idx]), lsuffix='', rsuffix='_bs'+str(i), on=[head_idx, tail_idx], how='left')
                branch_df.reset_index(inplace= True)
                if 'index' in branch_df.columns:
                    branch_df.drop('index',axis=1,inplace=True)
                
            subgraph = subgraph.join(branch_df.set_index(branch_head_idx), on = branch_head_idx, lsuffix='', rsuffix='_b'+str(key), how='left' )
            subgraph.reset_index(inplace= True)
            if 'index' in subgraph.columns:
                subgraph.drop('index',axis=1,inplace=True)
    
    subgraph = subgraph.T.drop_duplicates().T
    subgraph = subgraph.dropna()
    subgraph = subgraph.astype('int')
    return subgraph


def get_ontology_subgraphs_v3(ontology, link_intances):
    branch_flag = 'branch' in ontology.keys()
    stem_slave_flag = 'stem_slave' in ontology.keys()
    branch_slave_flag = 'branch_slave' in ontology.keys()
    
    stem_df = dd.from_pandas(pd.DataFrame(link_intances['stem'], columns = ontology['stem']), npartitions=12)
    subgraph = deepcopy(stem_df)
    
    switcher = [stem_slave_flag, branch_flag, branch_slave_flag]
    # all possible cases
    cases = np.array([[0,0,0],
                      [1,0,0],
                      [0,1,0],
                      [0,1,1],
                      [1,1,0],
                      [1,1,1]],dtype=bool)
    
    if (switcher == cases[0]).all():
        pass
    
    if (switcher == cases[1]).all():
        for i in range(len(link_intances['stem_slave'])):
            stem_slave_df = dd.from_pandas(pd.DataFrame(link_intances['stem_slave'][i], columns = ontology['stem_slave'][i]), npartitions=12)
            head_idx = ontology['stem_slave'][i][0]
            tail_idx = ontology['stem_slave'][i][-1]
            subgraph = dd.merge(subgraph, stem_slave_df, on = [head_idx,tail_idx], suffixes=(None,'_y'), how='inner')
            if sum(['_' in str(p) for p in subgraph.columns]):
                subgraph = subgraph.drop(str(head_idx)+'_y', axis = 1)
            subgraph = subgraph.dropna()
            subgraph = subgraph.drop_duplicates()

    if (switcher == cases[2]).all():
        for key in link_intances['branch'].keys():
            if (len(ontology['branch'][key]) == 2) & (ontology['branch'][key][0] == ontology['branch'][key][1]): #selfloop
                branch_df = dd.from_pandas(pd.DataFrame(link_intances['branch'][key], columns = [ontology['branch'][key][0], ontology['branch'][key][1] + 0.1]), npartitions=12) # +0.1 to avoid dupicated col_name
                print('self loop')
            else:
                branch_df = dd.from_pandas(pd.DataFrame(link_intances['branch'][key], columns = ontology['branch'][key]), npartitions=12)
            branch_head_idx = ontology['branch'][key][0]
            subgraph = dd.merge(subgraph, branch_df, on = branch_head_idx, suffixes=(None,'_y'), how='inner' )
            if sum(['_' in str(p) for p in subgraph.columns]):
                subgraph = subgraph.drop(str(branch_head_idx)+'_y', axis = 1)
            subgraph = subgraph.dropna()
            subgraph = subgraph.drop_duplicates()
    
    if (switcher == cases[3]).all():
        for key in link_intances['branch'].keys():
            if (len(ontology['branch'][key]) == 2) & (ontology['branch'][key][0] == ontology['branch'][key][1]): #selfloop
                branch_df = dd.from_pandas(pd.DataFrame(link_intances['branch'][key], columns = [ontology['branch'][key][0], ontology['branch'][key][1] + 0.1]), npartitions=12)
                print('self loop')
            else:
                branch_df = dd.from_pandas(pd.DataFrame(link_intances['branch'][key], columns = ontology['branch'][key]), npartitions=12)
            branch_head_idx = ontology['branch'][key][0]

            for i in range(len(link_intances['branch_slave'][key])):
                branch_slave_df = dd.from_pandas(pd.DataFrame(link_intances['branch_slave'][key][i], columns = ontology['branch_slave'][key][i]), npartitions=12)
                head_idx = ontology['branch_slave'][key][0]
                tail_idx = ontology['branch_slave'][key][-1]
                branch_df = dd.merge(branch_df, branch_slave_df, on=[head_idx, tail_idx], suffixes=(None,'_y'), how='inner')
                if sum(['_' in str(p) for p in subgraph.columns]):
                    subgraph = subgraph.drop(str(head_idx)+'_y', axis = 1)
                subgraph = subgraph.dropna()
                subgraph = subgraph.drop_duplicates()
                
            subgraph = dd.merge(subgraph, branch_df, on = branch_head_idx, suffixes=(None,'_y'), how='inner' )
            if sum(['_' in str(p) for p in subgraph.columns]):
                subgraph = subgraph.drop(str(branch_head_idx)+'_y', axis = 1)
            subgraph = subgraph.dropna()
            subgraph = subgraph.drop_duplicates()
                
    
    if (switcher == cases[4]).all():
        for i in range(len(link_intances['stem_slave'])):
            stem_slave_df = dd.from_pandas(pd.DataFrame(link_intances['stem_slave'][i], columns = ontology['stem_slave'][i]), npartitions=12)
            head_idx = ontology['stem_slave'][i][0]
            tail_idx = ontology['stem_slave'][i][-1]
            subgraph = dd.merge(subgraph, stem_slave_df, on = [head_idx,tail_idx], suffixes=(None,'_y'), how='inner')
            if sum(['_' in str(p) for p in subgraph.columns]):
                subgraph = subgraph.drop(str(head_idx)+'_y', axis = 1)
            subgraph = subgraph.dropna()
            subgraph = subgraph.drop_duplicates()
        
        for key in link_intances['branch'].keys():
            if (len(ontology['branch'][key]) == 2) & (ontology['branch'][key][0] == ontology['branch'][key][1]): #selfloop
                branch_df = dd.from_pandas(pd.DataFrame(link_intances['branch'][key], columns = [ontology['branch'][key][0],ontology['branch'][key][1] + 0.1]), npartitions=12)
                print('self loop')
            else:
                branch_df = dd.from_pandas(pd.DataFrame(link_intances['branch'][key], columns = ontology['branch'][key]), npartitions=12)
            branch_head_idx = ontology['branch'][key][0]
            subgraph = dd.merge(subgraph, branch_df, on = branch_head_idx, suffixes=(None,'_y'), how='inner' )
            if sum(['_' in str(p) for p in subgraph.columns]):
                subgraph = subgraph.drop(str(branch_head_idx)+'_y', axis = 1)
            subgraph = subgraph.dropna()
            subgraph = subgraph.drop_duplicates()

    
    if (switcher == cases[5]).all():
        for i in range(len(link_intances['stem_slave'])):
            stem_slave_df = dd.from_pandas(pd.DataFrame(link_intances['stem_slave'][i], columns = ontology['stem_slave'][i]), npartitions=12)
            head_idx = ontology['stem_slave'][i][0]
            tail_idx = ontology['stem_slave'][i][-1]
            subgraph = dd.merge(subgraph, stem_slave_df, on = [head_idx,tail_idx], suffixes=(None,'_y'), how='inner')
            if sum(['_' in str(p) for p in subgraph.columns]):
                subgraph = subgraph.drop(str(head_idx)+'_y', axis = 1)
            subgraph = subgraph.dropna()
            subgraph = subgraph.drop_duplicates()
                
        for key in link_intances['branch'].keys():
            if (len(ontology['branch'][key]) == 2) & (ontology['branch'][key][0] == ontology['branch'][key][1]): #selfloop
                branch_df = dd.from_pandas(pd.DataFrame(link_intances['branch'][key], columns = [ontology['branch'][key][0],str(ontology['branch'][key][1])]), npartitions=12)
                print('self loop')
            else:
                branch_df = dd.from_pandas(pd.DataFrame(link_intances['branch'][key], columns = ontology['branch'][key]), npartitions=12)
            branch_head_idx = ontology['branch'][key][0]

            for i in range(len(link_intances['branch_slave'][key])):
                branch_slave_df = dd.from_pandas(pd.DataFrame(link_intances['branch_slave'][key][i], columns = ontology['branch_slave'][key][i]), npartitions=12)
                head_idx = ontology['branch_slave'][key][0]
                tail_idx = ontology['branch_slave'][key][-1]
                branch_df = dd.merge(branch_df, branch_slave_df, on=[head_idx, tail_idx], suffixes=(None,'_y'), how='inner')
                if sum(['_' in str(p) for p in subgraph.columns]):
                    subgraph = subgraph.drop(str(head_idx)+'_y', axis = 1)
                subgraph = subgraph.dropna()
                subgraph = subgraph.drop_duplicates()
                
            subgraph = dd.merge(subgraph, branch_df, on = branch_head_idx, suffixes=(None,'_y'), how='inner' )
            if sum(['_' in str(p) for p in subgraph.columns]):
                subgraph = subgraph.drop(str(branch_head_idx)+'_y', axis = 1)
            subgraph = subgraph.dropna()
            subgraph = subgraph.drop_duplicates()
    
    subgraph = subgraph.dropna()
    subgraph = subgraph.drop_duplicates()
    subgraph = subgraph.astype('int')

    return subgraph.compute()


def row2grahp(M,row):
    mask = np.zeros_like(M, dtype=bool)
    mask[np.ix_(row,row)] = True
        # get ontology subgraph from masked adj matrix
    masked_adj = (M * mask).astype(int)
    ontology_subgraph = nx.from_numpy_matrix(masked_adj,create_using=nx.Graph)
    return ontology_subgraph


def row2graph_v2(M,row):
    g = nx.Graph()
    g.add_nodes_from(row)
    pair = np.argwhere(M[row][:,row] == True)
    edge_list = [(row[h],row[t]) for h,t in pair]
    g.add_edges_from(edge_list)
    return g

def row2grahp_v3(M,row):
    mask = scipy.sparse.csr_matrix(M.shape, dtype=bool)
    mask[np.ix_(row,row)] = True
        # get ontology subgraph from masked adj matrix
    masked_adj = (M * mask).astype(int)
    ontology_subgraph = nx.from_numpy_matrix(masked_adj,create_using=nx.Graph)
    return ontology_subgraph

def find_res_adj(M, subgraph):
    mask = np.ones_like(M, dtype=bool)
    for row in subgraph.values:
        mask[np.ix_(row,row)] = False
        '''
        for source in row:
            for target in row:
                mask[source, target] = False
                mask[target, source] = False
                '''
    res_adj_tmp = (M * mask).astype(int)
    '''
    tmp_g = nx.from_numpy_matrix(res_adj_tmp,create_using=nx.Graph)
    tmp_nodes = list(tmp_g.nodes)
    for node in tmp_nodes:
        mask[node, :] = True
        mask[:, node] = True
    res_adj = (M * mask).astype(int)
    '''
    return res_adj_tmp


def find_res_adj2(M, subgraph):
    mask = scipy.sparse.lil_matrix(M.shape, dtype=bool)
    for row in subgraph.values:
        mask[np.ix_(row,row)] = True
        '''
        for source in row:
            for target in row:
                mask[source, target] = False
                mask[target, source] = False
                '''
    # mask_reverse = scipy.sparse.lil_matrix(M.shape, dtype=bool) # mask_reverse == (~mask)
    # for i in range(mask.shape[0]):
    #     idx = (~(mask[i] != 0)).nonzero()[0]
    #     mask_reverse[i, idx] = True
        
    res_adj_tmp = (M > mask.tocsr()).astype(int)  # M > mask equals to M 'not' mask, which means M & (~mask)
    '''
    tmp_g = nx.from_numpy_matrix(res_adj_tmp,create_using=nx.Graph)
    tmp_nodes = list(tmp_g.nodes)
    for node in tmp_nodes:
        mask[node, :] = True
        mask[:, node] = True
    res_adj = (M * mask).astype(int)
    '''
    return res_adj_tmp 
 

def find_incomplete_subgraph(M, type_mask, ontology_pairs, res_adj):
    outs_pairs = []
    outs_graphs = []
    res_pairs = []
    
    for pair in ontology_pairs:
        instances = get_instances_from_link_length2_v2(pair, res_adj, type_mask)
        if len(instances) > 0:
            res_pairs = res_pairs + [pd.DataFrame(instances,columns=pair,dtype=int)]
        else:
            res_pairs = res_pairs + [pd.DataFrame(data=None,columns=pair,dtype=int)]

    res_pairs_bkp = deepcopy(res_pairs)
    ontology_pairs_bkp = deepcopy(ontology_pairs)
    i = 0
    for type in set(type_mask):
        flt = [type in p for p in ontology_pairs_bkp] # select which pair contains type
        if len(flt) == 0:
            continue
        selected = [res_pairs_bkp[p] for p in np.array(flt).nonzero()[0]]
        res_pairs_bkp = [res_pairs_bkp[p] for p in (~np.array(flt)).nonzero()[0]] # once selected, remove instances from source 
        ontology_pairs_bkp = [ontology_pairs_bkp[p] for p in (~np.array(flt)).nonzero()[0]]
        for df in selected:
            if i == 0:
                base = deepcopy(df)
                i += 1
                continue
            base_col = np.array(base.columns)
            df_col = np.array(df.columns)
            head = df_col[0] in base_col
            tail = df_col[1] in base_col
            if head and tail:
                tmp_base = deepcopy(base)
                for _, inctance in df.iterrows():
                    head_instance = inctance[df_col[0]] in base[df_col[0]].values
                    tail_instance = inctance[df_col[1]] in base[df_col[1]].values
                    if not (head_instance or tail_instance):
                            tmp_base = tmp_base.append(inctance,ignore_index=True)
                            continue
                    else:
                        for row, value in base.iterrows():
                            head_instance = inctance[df_col[0]] in value.values
                            tail_instance = inctance[df_col[1]] in value.values
                            if head_instance and tail_instance:
                                continue
                            elif head_instance:
                                if pd.isnull(value[df_col[1]]) and pd.isnull(tmp_base.loc[row,df_col[1]]):
                                    tmp_base.loc[row,df_col[1]] = inctance[df_col[1]]
                                else:
                                    tmp = base.iloc[row]
                                    tmp[df_col[1]] = inctance[df_col[1]]
                                    tmp_base = tmp_base.append(tmp,ignore_index=True)
                            elif tail_instance: 
                                if pd.isnull(value[df_col[0]]) and pd.isnull(tmp_base.loc[row,df_col[0]]):
                                    tmp_base.loc[row,df_col[0]] = inctance[df_col[0]]
                                else:
                                    tmp = base.iloc[row]
                                    tmp[df_col[0]] = inctance[df_col[0]]
                                    tmp_base = tmp_base.append(tmp,ignore_index=True)
                base = tmp_base
            elif head:
                base = base.merge(df, on=df_col[0], how='outer')
            elif tail:
                base = base.merge(df, on=df_col[1], how='outer')
                
    for check_table in res_pairs:
        base_set = set(tuple(p) for p in base[check_table.columns].drop_duplicates().dropna().astype(int).values)
        check_set = set(tuple(p) for p in check_table.values)

        if not (check_set == base_set):
            for incorrect_pair in list(base_set - check_set):
                base = base[~((base[check_table.columns[0]] == incorrect_pair[0])&(base[check_table.columns[1]] == incorrect_pair[1]))]
    
    for _,value in base.iterrows():
        outs_pairs = outs_pairs + [list(value.dropna().values.astype(int))]
    for row in outs_pairs:
        outs_graphs = outs_graphs + [row2grahp(M,row)]
                
    return outs_graphs, outs_pairs


def get_node_ontology_dict(M,ontology_subgraphs, subgraph):
    node_ontology = {}
    node_ontology_pairs = {}
    nodes = range(len(M))
    for node in nodes:
        i = 0
        indicators = np.argwhere(subgraph.values == node)
        tmp_neighbor_dict = {}
        tmp_dict = {}
        if len(indicators) == 0:
            continue
        for row,col in indicators:
            tmp_dict[i] = ontology_subgraphs[row]
            neighbors = subgraph.values[row]
            neighbors = neighbors[~(neighbors == neighbors[col])]
            for neighbor in neighbors:
                neighbor_path = nx.shortest_path(ontology_subgraphs[row],target=node, source=neighbor)
                tmp_neighbor_dict[i] =  tmp_neighbor_dict.get(i, []) + [neighbor_path]
            i += 1
        node_ontology[node] = tmp_dict
        node_ontology_pairs[node] = tmp_neighbor_dict
        
    return node_ontology, node_ontology_pairs