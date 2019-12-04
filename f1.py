#!/bin/env python3

import codecs
import sys
from collections import OrderedDict
import numpy as np
import json
import matplotlib.pyplot as plt

#文件保存的路径
basepath = "/home/bycc/analysis/f1_score/"

def _getfreqs(datalist, write=False):
    word_freqs = OrderedDict()
    for line in datalist:
        words_in = line.strip().split()
        for w in words_in:
            if w not in word_freqs:
                word_freqs[w] = 0
            word_freqs[w] += 1

    words = list(word_freqs.keys())
    wfreqs = list(word_freqs.values())
    sorted_idx = np.argsort(wfreqs)
    sorted_words = [words[i] for i in sorted_idx[::-1]]

    worddict = OrderedDict()
    sorted_word_freqs = OrderedDict()
    for i, w in enumerate(sorted_words):
        worddict[w] = i
        sorted_word_freqs[w] = word_freqs[w]

    if write:
        with open(basepath + "worddict.json", "w", encoding="utf-8") as f:
            json.dump(worddict, f, indent=2, ensure_ascii=False)
        with open(basepath + "wordfreqs.json", "w", encoding="utf-8") as f:
            json.dump(sorted_word_freqs, f, indent=2, ensure_ascii=False)

    return worddict, sorted_word_freqs

def _get_f1_score(candidateList, referenceList, can_freqs, ref_freqs, train_dict, train_freqs, n=5, write=False):
    """
    candidateList, referenceList: 候选集句子列表，ref句子列表
    can_freqs, ref_freqs: 候选集词与词频的字典，ref词与词频的字典
    train_freqs: 训练集词与词频的字典
    n: 将训练集中词频<=n的词作为rare，默认n=5
    write: 是否将结果写入文件
    """
    correct_dict = {}
    correct_oov_dict = {}
    correct_rare_dict = {}

    #rare: ref中在训练集中的词频<=n，n为阈值
    rare_dict, rare_can_dict = {}, {}
    rarenum = 0

    #oov：ref/can中不在训练集词表中的单词
    oov_dict, oov_can_dict = {}, {}
    oovnum = 0
    for key, value in ref_freqs.items():
        if key not in train_freqs:	#oov
            oov_dict[key] = value
        elif train_freqs[key] <= n:	#rare
            rare_dict[key] = value
    oovnum = sum(oov_dict.values())
    rarenum = sum(rare_dict.values())
    for key, value in can_freqs.items():
        if key in oov_dict or key not in train_freqs:	#oov
            oov_can_dict[key] = value
        if key in rare_dict:		#rare
            rare_can_dict[key] = value
 
    #逐句判断
    for can, ref in zip(candidateList, referenceList):
        #ref句子的词频
        ref_dict = {}
        for word in ref.strip().split():
            if word not in ref_dict:
                ref_dict[word] = 0
            ref_dict[word] += 1
        #can句子的词频
        can_dict = {}
        for word in can.strip().split():
            if word not in can_dict:
                can_dict[word] = 0
            can_dict[word] += 1

        #can中正确的单词数
        for key, value in can_dict.items():
            if key in ref_dict:		#正确命中
                if key not in correct_dict:
                    correct_dict[key] = 0     
                correct_dict[key] += min(value, ref_dict[key])

    #从can中正确的词及词频中抽取oov和rare的正确词与词频
    for key, value in correct_dict.items():
        if key in oov_dict:
            if key not in correct_oov_dict:
                correct_oov_dict[key] = 0
            correct_oov_dict[key] += value
        if key in rare_dict:
            if key not in correct_rare_dict:
                correct_rare_dict[key] = 0
            correct_rare_dict[key] += value    

    #all: p, r, f1
    allnum = sum(ref_freqs.values())
    all_p = sum(correct_dict.values())/sum(can_freqs.values())
    all_r = sum(correct_dict.values())/allnum
    all_f1 = (2*all_p*all_r)/(all_p+all_r)
    print("all:\tnum={},\tp={:.2f},\tr={:.2f},\tf1={:.2f}".format(allnum, all_p, all_r, all_f1))

    #oov: p, r, f1
    if oovnum > 0:
        oov_p = sum(correct_oov_dict.values())/(sum(oov_can_dict.values())+1e-8)
        oov_r = sum(correct_oov_dict.values())/oovnum
        oov_f1 = (2*oov_p*oov_r)/(oov_p+oov_r)
        print("oov:\tnum={},\tp={:.2f},\tr={:.2f},\tf1={:.2f}".format(oovnum, oov_p, oov_r, oov_f1))
    else:
        print("reference中没有oov")

    #rare: p, r, f1
    if rarenum > 0:
        rare_p = sum(correct_rare_dict.values())/(sum(rare_can_dict.values())+1e-8)
        rare_r = sum(correct_rare_dict.values())/rarenum
        rare_f1 = (2*rare_p*rare_r)/(rare_p+rare_r)
        print("rare:\tnum={},\tp={:.2f},\tr={:.2f},\tf1={:.2f}".format(rarenum, rare_p, rare_r, rare_f1))
    else:
        print("reference中没有rare")

    if write:
        with open(basepath + "correct_dict.json", "w", encoding="utf-8") as f:
            json.dump(correct_dict, f, indent=2, ensure_ascii=False)
        with open(basepath + "correct_oov_dict.json", "w", encoding="utf-8") as f:
            json.dump(correct_oov_dict, f, indent=2, ensure_ascii=False)
        with open(basepath + "correct_rare_dict.json", "w", encoding="utf-8") as f:
            json.dump(correct_rare_dict, f, indent=2, ensure_ascii=False)

    #按训练集词频进行分类，计算f1得分
    #词频-词 {625731:[word1, word2], 463449:[word1, word2, word3]}
    freq_words = OrderedDict()
    f1_dict = OrderedDict()

    #按词频聚类
    for key, value in train_freqs.items():
        if value not in freq_words:
            freq_words[value] = [key]
        else:
            freq_words[value].append(key)

    #计算f1
    correct, can_total, ref_total = 0, 0, 0
    for key, words in freq_words.items():
        for word in words:
            if word in ref_freqs: #单词在ref中
                ref_total += ref_freqs[word]
            if word in can_freqs:
                can_total += can_freqs[word]
            if word in correct_dict:
                correct += correct_dict[word]
        if ref_total:
            p = correct/(can_total+1e-8)
            r = correct/ref_total
            f1 = (2*p*r)/(p+r+1e-8)
            f1_dict[key] = f1
  
    #依据training set获取rank
    rank = []
    for k in f1_dict.keys():
        wl = freq_words[k]
        r = train_dict[wl[-1]] + 1 #这是取wl[0]，即第一个，也可考虑是否需要取wl[-1]
        rank.append(r)

    #绘制f1-rank图
    fig, ax = plt.subplots()
    ax.semilogx(rank, list(f1_dict.values()), 'b-')
    ax.set(xlabel="training set frequency rank", ylabel="unigram F1")
    ax.set_xlim(1)
    ax.set_ylim(0, 0.8)
    ax.grid(True)
    plt.show()

    if write:
        with open(basepath + "freq_words.json", "w", encoding="utf-8") as f:
            json.dump(freq_words, f, indent=2, ensure_ascii=False) 
        with open(basepath + "f1_dict.json", "w", encoding="utf-8") as f:
            json.dump(f1_dict, f, indent=2, ensure_ascii=False)
 
if __name__=="__main__":
    
    #训练集: train.en, test.en, ref
    training_set = sys.argv[1]
    test_set = sys.argv[2]
    test_ref = sys.argv[3]

    #训练集有序词表，有序词频
    trainList = codecs.open(training_set, "r", encoding="utf-8").readlines()
    worddict, wordfreqs = _getfreqs(trainList, write=True)
    #测试集有序词表，有序词频
    testList = codecs.open(test_set, "r", encoding="utf-8").readlines()
    test_worddict, test_wordfreqs = _getfreqs(testList)
    #测试集ref有序词表，有序词频
    testrefList = codecs.open(test_ref, "r", encoding="utf-8").readlines()
    ref_worddict, ref_wordfreqs = _getfreqs(testrefList)

    #all, rare, oov的f1得分
    #all: p=候选集正确词数/候选集总词数，r=候选集正确词数/ref总词数
    _get_f1_score(testList, testrefList, test_wordfreqs, ref_wordfreqs, worddict, wordfreqs, n=2, write=True)
