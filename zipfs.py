#!/bin/env python3

import os, sys
from collections import OrderedDict
import numpy as np
from math import log
import matplotlib.pyplot as plt
import json

def _get_frequency_rank(filename, level="word", write=False):
    word_freqs = OrderedDict()
    f = open(filename, "r", encoding="utf-8").readlines()
    for line in f:
        words_in = line.strip().split()
        if level == "char":
             words_in = list(''.join(words_in))
        for w in words_in:
            if w not in word_freqs:
                word_freqs[w] = 0
            word_freqs[w] += 1

    words = list(word_freqs.keys())
    freqs = list(word_freqs.values())
    vfreq = sum(freqs)
    sorted_idx = np.argsort(freqs)

    if write:
        sorted_words = [words[i] for i in sorted_idx[::-1]]
        sorted_freqs = [freqs[i]/vfreq for i in sorted_idx[::-1]]
        worddict = OrderedDict(zip(sorted_words, sorted_freqs))
        with open("/home/bycc/temp/worddict.json", "w", encoding="utf-8") as f:
            json.dump(worddict, f, indent=2, ensure_ascii=False)

    vsize = len(words)
    frequency = [freqs[i] for i in sorted_idx[::-1]]
    rank = [i for i in range(1, vsize+1)]
   
    ref_vsize = pow(10, len(str(frequency[0])))
    ref_x = np.linspace(1, ref_vsize, ref_vsize/10)
    ref_y = ref_vsize/ref_x
 
    fig, ax = plt.subplots()
    ax.loglog(rank, frequency, 'b.')
    ax.loglog(ref_x, ref_y, 'k:')
    ax.set(xlabel="Log Rank", ylabel="Log Frequency")
    ax.set_xlim(1, ref_vsize*10)
    ax.set_ylim(1, ref_vsize*10)
    ax.grid(True, axis='y')
    plt.show()

if __name__=="__main__":
    if len(sys.argv) < 2:
        print("参数不足:\n示例： ./zipfs.py filename")
        exit(1)
    
    write = False if os.path.exists(sys.argv[1]) else True
    level = "word" #["word", "char"]
    _get_frequency_rank(sys.argv[1], level=level, write=write)

