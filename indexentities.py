import pickle
import os.path as osp
import numpy as np
import click
from collections import defaultdict
import random
from copy import deepcopy
import time
import pdb
import logging
import os

import csv

def index_dataset(dataset_name, force=False):
    print('Indexing dataset {0}'.format(dataset_name))
    base_path = './dataset/{0}/'.format(dataset_name)
    base_path = '.'
    files = ['train_dataset.txt', 'valid_dataset.txt', 'test_dataset.txt']
    indexified_files = ['train_indexified.txt', 'valid_indexified.txt', 'test_indexified.txt']
    #files = ['medkg_dataset.txt']
    #indexified_files = ['medkg_indexified.txt']
    return_flag = True
    for i in range(len(indexified_files)):
        if not osp.exists(osp.join('.', indexified_files[i])):
            return_flag = False
            break
    if return_flag and not force:
        print ("index file exists")
        return  

    ent2id, rel2id, id2rel, id2ent = {}, {}, {}, {}

    entid, relid = 0, 0

    lines = []
    
    with open(osp.join(base_path, files[0])) as f:
        lines = f.readlines()
        file_len = len(lines)
    
    print(f"Total triples in {files[0]} {file_len}") 
    for p, indexified_p in zip(files, indexified_files):
        fw = open(osp.join(base_path, indexified_p), "w")
        i = 0
        with open(osp.join(base_path, p), 'r') as f:
            i += 1
            csv_reader = csv.reader(f, delimiter='\t')
            for row in csv_reader:
                print ('[%d/%d]'%(i, file_len), end='\r')
                try:
                    e1 = row[0].strip()
                    rel = row[1].strip()
                    e2 = row[2].strip('\n')
                    rel_reverse = '-' + rel
                    rel = '+' + rel
                    # rel_reverse = rel+ '_reverse'

                    if True:
                        if e1 not in ent2id.keys():
                            ent2id[e1] = entid
                            id2ent[entid] = e1
                            entid += 1

                        if e2 not in ent2id.keys():
                            ent2id[e2] = entid
                            id2ent[entid] = e2
                            entid += 1

                        if not rel in rel2id.keys():
                            rel2id[rel] = relid
                            id2rel[relid] = rel
                            assert relid % 2 == 0
                            relid += 1

                        if not rel_reverse in rel2id.keys():
                            rel2id[rel_reverse] = relid
                            id2rel[relid] = rel_reverse
                            assert relid % 2 == 1
                            relid += 1
                except ValueError:
                        print(f"Unpacking error for line {i} {row}")
                except IndexError:
                    print(f"Index error for line {i} {row}")
                if e1 in ent2id.keys() and e2 in ent2id.keys():
                    fw.write("\t".join([str(ent2id[e1]), str(rel2id[rel]), str(ent2id[e2])]) + "\n")
                    fw.write("\t".join([str(ent2id[e2]), str(rel2id[rel_reverse]), str(ent2id[e1])]) + "\n")
        fw.close()

    with open(osp.join(base_path, "stats.txt"), "w") as fw:
        fw.write("numentity: " + str(len(ent2id)) + "\n")
        fw.write("numrelations: " + str(len(rel2id)))
    with open(osp.join(base_path, 'ent2id.pkl'), 'wb') as handle:
        pickle.dump(ent2id, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(osp.join(base_path, 'rel2id.pkl'), 'wb') as handle:
        pickle.dump(rel2id, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(osp.join(base_path, 'id2ent.pkl'), 'wb') as handle:
        pickle.dump(id2ent, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(osp.join(base_path, 'id2rel.pkl'), 'wb') as handle:
        pickle.dump(id2rel, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print ('num entity: %d, num relation: %d'%(len(ent2id), len(rel2id)))
    print ("indexing finished!!")
            
index_dataset("medkg")