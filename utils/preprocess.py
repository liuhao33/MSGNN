from copy import deepcopy
import numpy as np
import scipy.sparse
import networkx as nx
import pandas as pd
import dask.dataframe as dd
import time

def plan_transform_v1(plan):
    transformed_plan = {}
    transformed_plan['stem'] = plan['main']
    if len(plan['ms']) > 0:
        transformed_plan['stem_slave'] = plan['ms']
    if 'res' in plan.keys():
        transformed_plan['branch'] = {}
        for i, res in enumerate(plan['res']):
            transformed_plan['branch'][i] = res['main']
            if len(res['ms']) > 0:
                transformed_plan['branch_slave'] = {}
                transformed_plan['branch_slave'][i] = res['ms']
    return transformed_plan


def plan_transform_v2(plan):
    assert len(plan.keys()) < 3, "Error: function currently is compatible with recursion depth < 3 only."
        
    transformed_plan = {}
    transformed_plan['stem'] = plan[0][0]['main']  # depth 1
    if 'ms' in plan[0][0].keys():
        transformed_plan['stem_slave'] = plan[0][0]['ms']
    if len(plan.keys()) == 2:
        transformed_plan['branch'] = {}  # depth 2
        transformed_plan['branch_slave'] = {}
        for i in plan[1].keys():
            transformed_plan['branch'][i] = plan[1][i]['main']
            if 'ms' in plan[1][i].keys():
                transformed_plan['branch_slave'][i] = plan[1][i]['ms']
    
        if len(transformed_plan['branch_slave']) == 0:
            del transformed_plan['branch_slave'] 
    
    return transformed_plan


def get_intances(M, type_mask, schema, prefix_operator):
    """
    :param M: the raw adjacency matrix
    :param type_mask: an array of types of all node
    :param schema: a dict of schema decomposition plan, which contains stem link, branch link and corresponding slave links.
    :return: a list of python dictionaries, consisting of corresponding link instances
    """
    outs = {}
    # get stem intances
    link = schema['stem']
    # print('stem searching starts!')
    stem_instances = get_instances_from_link(link, M, type_mask, prefix_operator)
    outs['stem'] = stem_instances
    
    # get branch intances
    if 'branch' in schema.keys():
        branch_instances={} 
        for key in schema['branch'].keys():
            link = schema['branch'][key]
            # print(f'branch{key} searching starts!')
            branch_paths = get_instances_from_link(link, M, type_mask, prefix_operator)
            branch_instances[key] = branch_paths
        outs['branch'] = branch_instances
                    
    
    # get stem_slave intances
    if 'stem_slave' in schema.keys():
        stem_slave_instances=[]
        cnt = 0 
        for link in schema['stem_slave']:
            # print(f'stem_slave{cnt} searching starts!')
            cnt += 1
            stem_slave_paths = get_instances_from_link(link, M, type_mask, prefix_operator)
            stem_slave_instances.append(stem_slave_paths)
        outs['stem_slave'] = stem_slave_instances
    
    
    # get branch_slave intances
    if 'branch_slave' in schema.keys():
        branch_slave_instances={}
        for key in schema['branch_slave'].keys():
            tmp = []
            cnt = 0
            for link in schema['branch_slave'][key]:
                # print(f'branch{key}slave{cnt} searching starts!')
                cnt += 1
                branch_slave_paths = get_instances_from_link(link, M, type_mask, prefix_operator)  
                tmp.append(branch_slave_paths)
            branch_slave_instances[key] = tmp
        outs['branch_slave'] = branch_slave_instances
    
    return outs

def get_instances_from_link(link, M, type_mask, prefix_operator):
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


def get_schema_subgraphs(schema, link_intances):
    branch_flag = 'branch' in schema.keys()
    stem_slave_flag = 'stem_slave' in schema.keys()
    branch_slave_flag = 'branch_slave' in schema.keys()
    
    stem_df = pd.DataFrame(link_intances['stem'], columns = schema['stem'])
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
            stem_slave_df = pd.DataFrame(link_intances['stem_slave'][i], columns = schema['stem_slave'][i])
            head_idx = schema['stem_slave'][i][0]
            tail_idx = schema['stem_slave'][i][-1]
            subgraph = subgraph.join(stem_slave_df.set_index([head_idx,tail_idx]), lsuffix='', rsuffix='_ss'+str(i), on=[head_idx, tail_idx], how='left')
            subgraph.reset_index(inplace= True)
            if 'index' in subgraph.columns:
                subgraph.drop('index',axis=1,inplace=True)

    if (switcher == cases[2]).all():
        for key in link_intances['branch'].keys():
            branch_df = pd.DataFrame(link_intances['branch'][key], columns = schema['branch'][key])
            head_idx = schema['branch'][key][0]
            subgraph = subgraph.join(branch_df.set_index(head_idx), on = head_idx, lsuffix='', rsuffix='_b'+str(key), how='left' )
            subgraph.reset_index(inplace= True)
            if 'index' in subgraph.columns:
                subgraph.drop('index',axis=1,inplace=True)
    
    if (switcher == cases[3]).all():
        for key in link_intances['branch'].keys():
            branch_df = pd.DataFrame(link_intances['branch'][key], columns = schema['branch'][key])
            branch_head_idx = schema['branch'][key][0]

            for i in range(len(link_intances['branch_slave'][key])):
                branch_slave_df = pd.DataFrame(link_intances['branch_slave'][key][i], columns = schema['branch_slave'][key][i])
                head_idx = schema['branch_slave'][key][0]
                tail_idx = schema['branch_slave'][key][-1]
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
            stem_slave_df = pd.DataFrame(link_intances['stem_slave'][i], columns = schema['stem_slave'][i])
            head_idx = schema['stem_slave'][i][0]
            tail_idx = schema['stem_slave'][i][-1]
            subgraph = subgraph.join(stem_slave_df.set_index([head_idx,tail_idx]), lsuffix='', rsuffix='_ss'+str(i), on=[head_idx, tail_idx], how='left')
            subgraph.reset_index(inplace= True)
            if 'index' in subgraph.columns:
                subgraph.drop('index',axis=1,inplace=True)
            subgraph = subgraph.dropna()
        for key in link_intances['branch'].keys():
            if (len(schema['branch'][key]) == 2) & (schema['branch'][key][0] == schema['branch'][key][1]): 
                branch_df = pd.DataFrame(link_intances['branch'][key], columns = [schema['branch'][key][0],str(schema['branch'][key][1])])
            else:branch_df = pd.DataFrame(link_intances['branch'][key], columns = schema['branch'][key])
            branch_head_idx = schema['branch'][key][0]
            subgraph = subgraph.join(branch_df.set_index(branch_head_idx), on = branch_head_idx, lsuffix='', rsuffix='_b'+str(key), how='left' )
            subgraph.reset_index(inplace= True)
            if 'index' in subgraph.columns:
                subgraph.drop('index',axis=1,inplace=True)

    
    if (switcher == cases[5]).all():
        for i in range(len(link_intances['stem_slave'])):
            stem_slave_df = pd.DataFrame(link_intances['stem_slave'][i], columns = schema['stem_slave'][i])
            head_idx = schema['stem_slave'][i][0]
            tail_idx = schema['stem_slave'][i][-1]
            subgraph = subgraph.join(stem_slave_df.set_index([head_idx,tail_idx]), lsuffix='', rsuffix='_ss'+str(i), on=[head_idx, tail_idx], how='left')
            subgraph.reset_index(inplace= True)
            if 'index' in subgraph.columns:
                subgraph.drop('index',axis=1,inplace=True)
                
        for key in link_intances['branch'].keys():
            branch_df = pd.DataFrame(link_intances['branch'][key], columns = schema['branch'][key])
            branch_head_idx = schema['branch'][key][0]

            for i in range(len(link_intances['branch_slave'][key])):
                branch_slave_df = pd.DataFrame(link_intances['branch_slave'][key][i], columns = schema['branch_slave'][key][i])
                head_idx = schema['branch_slave'][key][0]
                tail_idx = schema['branch_slave'][key][-1]
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


def get_schema_subgraphs_parallel(schema, link_intances):
    branch_flag = 'branch' in schema.keys()
    stem_slave_flag = 'stem_slave' in schema.keys()
    branch_slave_flag = 'branch_slave' in schema.keys()
    
    stem_df = dd.from_pandas(pd.DataFrame(link_intances['stem'], columns = schema['stem']), npartitions=12)
    subgraph = deepcopy(stem_df)
    
    switcher = [stem_slave_flag, branch_flag, branch_slave_flag]
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
            stem_slave_df = dd.from_pandas(pd.DataFrame(link_intances['stem_slave'][i], columns = schema['stem_slave'][i]), npartitions=12)
            head_idx = schema['stem_slave'][i][0]
            tail_idx = schema['stem_slave'][i][-1]
            subgraph = dd.merge(subgraph, stem_slave_df, on = [head_idx,tail_idx], suffixes=(None,'_y'), how='inner')
            if sum(['_' in str(p) for p in subgraph.columns]):
                subgraph = subgraph.drop(str(head_idx)+'_y', axis = 1)
            subgraph = subgraph.dropna()
            subgraph = subgraph.drop_duplicates()

    if (switcher == cases[2]).all():
        for key in link_intances['branch'].keys():
            if (len(schema['branch'][key]) == 2) & (schema['branch'][key][0] == schema['branch'][key][1]): #selfloop
                branch_df = dd.from_pandas(pd.DataFrame(link_intances['branch'][key], columns = [schema['branch'][key][0], schema['branch'][key][1] + 0.1]), npartitions=12) # +0.1 to avoid dupicated col_name
                print('self loop')
            else:
                branch_df = dd.from_pandas(pd.DataFrame(link_intances['branch'][key], columns = schema['branch'][key]), npartitions=12)
            branch_head_idx = schema['branch'][key][0]
            subgraph = dd.merge(subgraph, branch_df, on = branch_head_idx, suffixes=(None,'_y'), how='inner' )
            if sum(['_' in str(p) for p in subgraph.columns]):
                subgraph = subgraph.drop(str(branch_head_idx)+'_y', axis = 1)
            subgraph = subgraph.dropna()
            subgraph = subgraph.drop_duplicates()
    
    if (switcher == cases[3]).all():
        for key in link_intances['branch'].keys():
            if (len(schema['branch'][key]) == 2) & (schema['branch'][key][0] == schema['branch'][key][1]): #selfloop
                branch_df = dd.from_pandas(pd.DataFrame(link_intances['branch'][key], columns = [schema['branch'][key][0], schema['branch'][key][1] + 0.1]), npartitions=12)
                print('self loop')
            else:
                branch_df = dd.from_pandas(pd.DataFrame(link_intances['branch'][key], columns = schema['branch'][key]), npartitions=12)
            branch_head_idx = schema['branch'][key][0]

            for i in range(len(link_intances['branch_slave'][key])):
                branch_slave_df = dd.from_pandas(pd.DataFrame(link_intances['branch_slave'][key][i], columns = schema['branch_slave'][key][i]), npartitions=12)
                head_idx = schema['branch_slave'][key][0]
                tail_idx = schema['branch_slave'][key][-1]
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
            stem_slave_df = dd.from_pandas(pd.DataFrame(link_intances['stem_slave'][i], columns = schema['stem_slave'][i]), npartitions=12)
            head_idx = schema['stem_slave'][i][0]
            tail_idx = schema['stem_slave'][i][-1]
            subgraph = dd.merge(subgraph, stem_slave_df, on = [head_idx,tail_idx], suffixes=(None,'_y'), how='inner')
            if sum(['_' in str(p) for p in subgraph.columns]):
                subgraph = subgraph.drop(str(head_idx)+'_y', axis = 1)
            subgraph = subgraph.dropna()
            subgraph = subgraph.drop_duplicates()
        
        for key in link_intances['branch'].keys():
            if (len(schema['branch'][key]) == 2) & (schema['branch'][key][0] == schema['branch'][key][1]): #selfloop
                branch_df = dd.from_pandas(pd.DataFrame(link_intances['branch'][key], columns = [schema['branch'][key][0],schema['branch'][key][1] + 0.1]), npartitions=12)
                print('self loop')
            else:
                branch_df = dd.from_pandas(pd.DataFrame(link_intances['branch'][key], columns = schema['branch'][key]), npartitions=12)
            branch_head_idx = schema['branch'][key][0]
            subgraph = dd.merge(subgraph, branch_df, on = branch_head_idx, suffixes=(None,'_y'), how='inner' )
            if sum(['_' in str(p) for p in subgraph.columns]):
                subgraph = subgraph.drop(str(branch_head_idx)+'_y', axis = 1)
            subgraph = subgraph.dropna()
            subgraph = subgraph.drop_duplicates()

    
    if (switcher == cases[5]).all():
        for i in range(len(link_intances['stem_slave'])):
            stem_slave_df = dd.from_pandas(pd.DataFrame(link_intances['stem_slave'][i], columns = schema['stem_slave'][i]), npartitions=12)
            head_idx = schema['stem_slave'][i][0]
            tail_idx = schema['stem_slave'][i][-1]
            subgraph = dd.merge(subgraph, stem_slave_df, on = [head_idx,tail_idx], suffixes=(None,'_y'), how='inner')
            if sum(['_' in str(p) for p in subgraph.columns]):
                subgraph = subgraph.drop(str(head_idx)+'_y', axis = 1)
            subgraph = subgraph.dropna()
            subgraph = subgraph.drop_duplicates()
                
        for key in link_intances['branch'].keys():
            if (len(schema['branch'][key]) == 2) & (schema['branch'][key][0] == schema['branch'][key][1]): #selfloop
                branch_df = dd.from_pandas(pd.DataFrame(link_intances['branch'][key], columns = [schema['branch'][key][0],str(schema['branch'][key][1])]), npartitions=12)
                print('self loop')
            else:
                branch_df = dd.from_pandas(pd.DataFrame(link_intances['branch'][key], columns = schema['branch'][key]), npartitions=12)
            branch_head_idx = schema['branch'][key][0]

            for i in range(len(link_intances['branch_slave'][key])):
                branch_slave_df = dd.from_pandas(pd.DataFrame(link_intances['branch_slave'][key][i], columns = schema['branch_slave'][key][i]), npartitions=12)
                head_idx = schema['branch_slave'][key][0]
                tail_idx = schema['branch_slave'][key][-1]
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
    masked_adj = (M * mask).astype(int)
    ontology_subgraph = nx.from_numpy_matrix(masked_adj,create_using=nx.Graph)
    return ontology_subgraph


def get_node_schema_dict(M,schema_subgraphs, subgraph):
    node_schema = {}
    node_schema_pairs = {}
    nodes = range(len(M))
    for node in nodes:
        i = 0
        indicators = np.argwhere(subgraph.values == node)
        tmp_neighbor_dict = {}
        tmp_dict = {}
        if len(indicators) == 0:
            continue
        for row,col in indicators:
            tmp_dict[i] = schema_subgraphs[row]
            neighbors = subgraph.values[row]
            neighbors = neighbors[~(neighbors == neighbors[col])]
            for neighbor in neighbors:
                neighbor_path = nx.shortest_path(schema_subgraphs[row],target=node, source=neighbor)
                tmp_neighbor_dict[i] =  tmp_neighbor_dict.get(i, []) + [neighbor_path]
            i += 1
        node_schema[node] = tmp_dict
        node_schema_pairs[node] = tmp_neighbor_dict
        
    return node_schema, node_schema_pairs