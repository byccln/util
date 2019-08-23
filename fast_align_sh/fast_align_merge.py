#!/bin/env python3

import codecs
import sys

if __name__=="__main__":
    f1 = codecs.open(sys.argv[1], 'r').readlines()
    f2 = codecs.open(sys.argv[2], 'r').readlines()
    fout = codecs.open(sys.argv[3], 'w', 'utf-8')
    len = len(f1)
    for i in range(len):
        fout.write(f1[i].strip()+" ||| "+f2[i].strip()+"\n")
    
