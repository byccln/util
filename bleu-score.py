#!/bin/env python3

import os
import sys
import re
from math import exp, log

def get_bleu(reference_file, candidate_file, tokenize=False):
    if tokenize:
        print("BLEU score (tokenize):")
    else:
        print("BLEU score (no tokenize):")
    reference = open(reference_file, "r", encoding="utf-8")
    candidate = open(candidate_file, "r", encoding="utf-8")
    length_reference, length_candidate = 0, 0
    correct = [0]*4
    total = [0]*4
    #逐行计算bleu
    for ref, can in zip(reference, candidate):
        if tokenize:
            ref = tokenization(ref)
            can = tokenization(can)
        #参考译文句
        reference_dict = {}
        wordslist_ref = ref.strip().split()
        length = len(wordslist_ref)
        length_reference += length
        for n in range(4):
            for i in range(length-n):
                ngram = str(n+1)+" "+" ".join(wordslist_ref[i:i+n+1])
                if ngram not in reference_dict:
                    reference_dict[ngram] = 0
                reference_dict[ngram] += 1
        #候选译文句
        wordslist_can = can.strip().split()
        length = len(wordslist_can)
        length_candidate += length
        for n in range(4):
            candidate_dict = {}
            for i in range(length-n):
                ngram = str(n+1)+" "+" ".join(wordslist_can[i:i+n+1])
                if ngram not in candidate_dict:
                    candidate_dict[ngram] = 0
                candidate_dict[ngram] += 1
        
            for key, value in candidate_dict.items():
                total[n] += value
                if key in reference_dict:
                    correct[n] += min(reference_dict[key], value)      
    #考虑够长惩罚并计算BLEU值
    brevity_penalty = 1
    bleu = [0.]*4
    for n in range(4):
        if total[n]:
            bleu[n] = correct[n]/total[n]
        else:
            bleu[n] = 0.
    if length_reference == 0:
        print("BLEU = 0, 0/0/0/0 (BP=0, ratio=0, hyp_len=0, ref_len=0)")
        exit()
    
    if length_candidate < length_reference:
        brevity_penalty = exp(1-length_reference/length_candidate)
    temp = sum([log(bleu[i]) if bleu[i] else -9999999999 for i in range(4)])/4
    BLEU = brevity_penalty * exp(temp)
    print("BLEU = {0:.2f}, {1:.1f}/{2:.1f}/{3:.1f}/{4:.1f} (BP={5:.3f}, ratio={6:.3f}, hyp_len={7:d}, ref_len={8:d})".format(
            100*BLEU, 100*bleu[0], 100*bleu[1], 100*bleu[2], 100*bleu[3],
            brevity_penalty, length_candidate/length_reference,
            length_candidate, length_reference))            

def tokenization(sentence):
    sent = re.sub(r'<skipped>', '', sentence)
    #language-independent part
    sent = re.sub(r'-\n', '', sent)
    sent = re.sub(r'\n', ' ', sent)
    sent = re.sub(r'&quot;', '"', sent)
    sent = re.sub(r'&amp;', '&', sent)
    sent = re.sub(r'&lt;', '<', sent)
    sent = re.sub(r'&gt;', '>', sent)
    #language-dependent part (assuming Western languages)
    sent = ' '+sent+' '
    #分离标点符号
    sent = re.sub(r'[\{-\~\[-\` -\&\(-\+\:-\@\/]', lambda s:' '+s.group(0)+' ', sent)
    #分离[.,] 除非前面有一个数字
    sent = re.sub(r'([^0-9])([\.,])', lambda s:s.group(1)+' '+s.group(2)+' ', sent)
    #分离[.,] 除非后面有一个数字
    sent = re.sub(r'([\.,])([^0-9])', lambda s:' '+s.group(1)+' '+s.group(2), sent)
    #前面是数字时分离[-]
    sent = re.sub(r'([0-9])(-)', lambda s:s.group(1)+' '+s.group(2)+' ', sent)
    #words间只有一个空格
    sent = re.sub(r'\s+', ' ', sent)
    #消除句首的[\n\t\r\f]+
    sent = re.sub(r'^\s+', '', sent)
    #消除句尾的[\n\t\r\f]+
    sent = re.sub(r'\s+$', '', sent)
    return sent
    
if __name__=="__main__":
    if len(sys.argv) < 3:
        print("参数不足")
        print("Example:\n\t./bleu_score.py reference candidate")
        print("\t./bleu_score.py reference candidate -tok")
        exit(1)
    else:
        reference = sys.argv[1]
        candidate = sys.argv[2]
    if len(sys.argv)>3 and sys.argv[3] == '-tok':
        tokenize = True
    else:
        tokenize = False
    '''
    reference = "D:/Code/perl/newsdev2016.tok.en"
    candidate = "D:/Code/perl/newsdev2016.bpe.ro.output.postprocessed.dev"
    '''
    assert os.access(reference, os.F_OK) and os.access(candidate, os.F_OK)
    get_bleu(reference, candidate, tokenize)
