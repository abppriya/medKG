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

tasks = ['1p', '2p', '3p', '2i', '3i', 'pi', 'ip' ,'2u', 'up']

train_queries = defaultdict(set)
train_answers =  defaultdict(set)
valid_queries = defaultdict(set)
valid_easy_answers =  defaultdict(set)
valid_hard_answers =  defaultdict(set)
test_queries = defaultdict(set)
test_easy_answers =  defaultdict(set)
test_hard_answers =  defaultdict(set)

for name in tasks:
    train_queries.update(pickle.load(open(os.path.join('.', f"test-{name}-queries.pkl"), 'rb')))
    train_answers.update(pickle.load(open(os.path.join('.', f"test-{name}-hard-answers.pkl"), 'rb')))    

for name in tasks:
    test_queries.update(pickle.load(open(os.path.join('.', f"test-{name}-queries.pkl"), 'rb')))
    test_hard_answers.update(pickle.load(open(os.path.join('.', f"test-{name}-hard-answers.pkl"), 'rb')))
    test_easy_answers.update(pickle.load(open(os.path.join('.', f"test-{name}-easy-answers.pkl"), 'rb')))
    
for name in tasks:
    valid_queries.update(pickle.load(open(os.path.join('.', f"valid-{name}-queries.pkl"), 'rb')))
    valid_hard_answers.update(pickle.load(open(os.path.join('.', f"valid-{name}-hard-answers.pkl"), 'rb')))
    valid_easy_answers.update(pickle.load(open(os.path.join('.', f"valid-{name}-easy-answers.pkl"), 'rb')))
    

with open('final/test-queries.pkl', 'wb') as f:
        pickle.dump(test_queries, f)
with open('final/test-hard-answers.pkl', 'wb') as f:
        pickle.dump(test_hard_answers, f)
with open('final/test-easy-answers.pkl', 'wb') as f:
        pickle.dump(test_easy_answers, f)

        
with open('final/valid-queries.pkl', 'wb') as f:
        pickle.dump(valid_queries, f)
with open('final/valid-hard-answers.pkl', 'wb') as f:
        pickle.dump(valid_hard_answers, f)
with open('final/valid-easy-answers.pkl', 'wb') as f:
        pickle.dump(valid_easy_answers, f)
        
with open('final/train-queries.pkl', 'wb') as f:
        pickle.dump(train_queries, f)
with open('final/train-answers.pkl', 'wb') as f:
        pickle.dump(train_answers, f)
