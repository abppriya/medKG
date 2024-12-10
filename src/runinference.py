
import argparse
import json
import logging
import os
import random
import torch
import torch.nn as nn
import collections
import pickle

import numpy as np
from torch.utils.data import DataLoader
from models import KGReasoning
from dataloader import TestDataset, TrainDataset, SingledirectionalOneShotIterator
from util import flatten_query, list2tuple, parse_time, set_global_seed, eval_tuple
from orkg import ORKG, Hosts, graph

client = ORKG(host=Hosts.PRODUCTION)

def getorkglabel(rid, id2ent):
    rid_entity = id2ent[rid]
    if(rid_entity.startswith("R")) :
        resp = client.resources.by_id(id=rid_entity)
        label = resp.content['label']
        return rid_entity, label
    else:
        return rid_entity, "None"


def flatten_query(queries):
    all_queries = []
    for query_structure in queries:
        tmp_queries = list(queries[query_structure])
        all_queries.extend([(query, query_structure) for query in tmp_queries])
    return all_queries


def set_logger(args):
    '''
    Write logs to console and log file
    '''
   
    log_file = os.path.join(args.save_path, 'run.log')

    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S',
        filename=log_file,
        filemode='a+'
    )
    if args.print_on_screen:
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
        console.setFormatter(formatter)
        logging.getLogger('').addHandler(console)


query_name_dict = {('e',('r',)): '1p', 
                    ('e', ('r', 'r')): '2p',
                    ('e', ('r', 'r', 'r')): '3p',
                    (('e', ('r',)), ('e', ('r',))): '2i',
                    (('e', ('r',)), ('e', ('r',)), ('e', ('r',))): '3i',
                    ((('e', ('r',)), ('e', ('r',))), ('r',)): 'ip',
                    (('e', ('r', 'r')), ('e', ('r',))): 'pi',
                    (('e', ('r',)), ('e', ('r',)), ('u')):'2u',
                    (('e', ('r',)), ('e', ('r',)), ('u'),('r')):'up'
                }

test_queries = {(('e', ('r',)), ('e',('r',))): {((2241, (31,)), (2484, (91,)))}}

nentity = 4923
nrelation = 264
hidden_dim = 400
gamma = 12.0
geo = "box"
cuda = True
box_mode = ('none', 0.02)
beta_mode = ('none', 0.02)
checkpoint_path = os.path.join('.','checkpoint')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 

for query_structure in test_queries:
    logging.info(query_name_dict[query_structure]+": "+str(len(test_queries[query_structure])))
    test_queries = flatten_query(test_queries)
    test_dataloader = DataLoader(
        TestDataset(
            test_queries, 
            nentity, 
            nrelation, 
        ), 
        batch_size=1,
        num_workers=1, 
        collate_fn=TestDataset.collate_fn
    )

model = KGReasoning(
    nentity=nentity,
    nrelation=nrelation,
    hidden_dim=hidden_dim,
    gamma=gamma,
    geo="box",
    use_cuda=True,
    box_mode=eval_tuple(box_mode),
    beta_mode = eval_tuple(beta_mode),
    test_batch_size=1,
    query_name_dict = query_name_dict
)
model.to(device)

logging.info('Loading checkpoint %s...' % checkpoint_path)
checkpoint = torch.load(checkpoint_path)
init_step = checkpoint['step']
model.load_state_dict(checkpoint['model_state_dict'])

logging.info('Evaluating on Test Dataset...')

with torch.no_grad():
    for negative_sample, queries, queries_unflatten, query_structures in test_dataloader:
        for query in queries:
            print(query)
        
        print("---")
        
        for query in enumerate(queries_unflatten):
            print(query)
            
        batch_queries_dict = collections.defaultdict(list)
        batch_idxs_dict = collections.defaultdict(list)
        for i, query in enumerate(queries):
            batch_queries_dict[query_structures[i]].append(query)
            batch_idxs_dict[query_structures[i]].append(i)
        
        for query_structure in batch_queries_dict:
            batch_queries_dict[query_structure] = torch.LongTensor(batch_queries_dict[query_structure]).cuda()
        
        negative_sample = negative_sample.cuda()

        _, negative_logit, _, idxs = model(None, negative_sample, None, batch_queries_dict, batch_idxs_dict)
        queries_unflatten = [queries_unflatten[i] for i in idxs]
        query_structures = [query_structures[i] for i in idxs]
        argsort = torch.argsort(negative_logit, dim=1, descending=True)
        ranking = argsort.clone().to(torch.float)
        ranking = ranking.scatter_(1, 
                                    argsort, 
                                    torch.arange(model.nentity).to(torch.float).repeat(argsort.shape[0], 
                                                                                        1).cuda()
                                    ) # achieve the ranking of all entities
        
        
        candidates = ranking[:, :10].tolist()
        print(candidates)
        flat_list = [item for sublist in candidates for item in sublist]

        id2ent = pickle.load(open(os.path.join('.','id2ent.pkl'), 'rb'))
        id2rel = pickle.load(open(os.path.join('.','id2rel.pkl'), 'rb'))
           
        
        rid_entity1 , label1 = getorkglabel(2241, id2ent)
        rid_entity2 , label2 = getorkglabel(2484, id2ent)
        
        print(f"Query ({rid_entity1}:{label1}, is venueof) AND ({rid_entity2}:{label2}, Is reasearch problem) ")

        print("Ranked candidate answer")
        for i in flat_list :
            i = int(i)
            if i in id2ent :
                entity, label = getorkglabel(i, id2ent)
                print(f"{entity} - {label}")

        
