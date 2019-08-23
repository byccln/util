#!/bin/env python3

import pandas as pd
import numpy as np
import sys
import codecs

#truncate = 2000

def forward(score, source, target, truncate):
    init_data = pd.read_csv(score, names=['score'], dtype={'score':np.float32})
    fs = codecs.open(source, 'r', 'utf-8')
    ft = codecs.open(target, 'r', 'utf-8')
    sources = fs.readlines()
    targets = ft.readlines()
    assert len(sources) == len(targets)
    scores = init_data['score'].values
    idx = np.argsort(scores)
    idx = idx[::-1]
    idx1 = idx[:truncate]
    idx2 = idx[truncate:]
    fws = codecs.open(source+'.top'+str(truncate), 'w', 'utf-8')
    fwt = codecs.open(target+'.top'+str(truncate), 'w', 'utf-8')
    for i in idx1:
        fws.write(sources[i])
        fwt.write(targets[i])
    
    fo1 = codecs.open(source+'.down'+str(len(idx2)), 'w', 'utf-8')
    fo2 = codecs.open(target+'.down'+str(len(idx2)), 'w', 'utf-8')
    for i in idx2:
        fo1.write(sources[i])
        fo2.write(targets[i])
    
if __name__=="__main__":
    if len(sys.argv) < 5:
        print("./nsort.py score_file source_file target_file top_num")
        exit(0)
    forward(sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]))
