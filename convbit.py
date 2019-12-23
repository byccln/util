#!/bin/env python3

import sys
import re
import json
import numpy as np
import warnings
import time
import codecs

bitlist = {
	'0':'0000', '1':'0001',	'2':'0010', '3':'0011',	'4':'0100', '5':'0101',	'6':'0110', '7':'0111',	'8':'1000', '9':'1001',	'a':'1010', 'b':'1011',	'c':'1100', 'd':'1101',	'e':'1110', 'f':'1111'
}

reversed_bit = dict(zip(bitlist.values(), bitlist.keys()))

#将字符文本转为bit文本，可以选择是否保留空格
def char2bit(filename, keepblank=True):
    print (time.strftime("%a %b %d %H:%M:%S %Y", time.localtime()))
    f = open(filename, encoding="utf-8")
    if keepblank:
        datalist = [line for line in f] #保留空格
    else:
        datalist = [''.join(line.strip().split()) for line in f] #不保留空格
    fwbit = open(filename+".bit", "w", encoding="utf-8")
    for line in datalist:
        u8 = [hex(ord(i))[2:] for i in line]
        bit = [bitlist[bt] for byte in u8 for bt in byte]
        fwbit.write(''.join(bit)+"\n")
    print (time.strftime("%a %b %d %H:%M:%S %Y", time.localtime()))

#这里只转到utf8，utf8-字符由DP实现
def bit2u8(filename):
    f = open(filename, "r", encoding="utf-8").readlines()
    fw = open(filename+".convu8", "w", encoding="utf-8")
    for line in f:
        line = line.strip()
        slist = [line[i:i+4] for i in range(0, len(line), 4)] #4个一组进行分组
        chs = [reversed_bit[i] for i in slist]
        fw.write("".join(chs)+"\n")

def get_information(filename):
    word_cnt, char_cnt = 0, 0
    f = open(filename, "r").readlines()
    for line in f:
        words_in = line.strip().split()
        word_cnt += len(words_in)
        char_cnt += len(''.join(words_in))
    print("words:{}, words_per_sentence:{}".format(word_cnt, word_cnt/len(f)))
    print("chars:{}, chars_per_sentence:{}".format(char_cnt, char_cnt/len(f)))

#将bit序列解码为字符序列
def debug_bit2char(bitfile, vocab_path):
    #待解码的句子，解码用到的bit字符对照表，保存文件的路径
    bitsents = open(bitfile, "r", encoding="utf-8").readlines()
    valid_vocab = json.load(open(vocab_path, "r", encoding="utf-8"))
    #将bit串中的bit片段抽取出来，逐个解码
    pattern1 = re.compile(r"[01]+") #匹配bit串中连续的01序列
    pattern2 = re.compile(r"[^01]+") #匹配bit串中的连续的非01序列
    #逐行解码句子
    unusedlist = [] #统计每句中解码后未使用的bit数
    for idx, bsent in enumerate(bitsents):
        unused = 0
        bsent = bsent.strip().replace('@@ ', '')
        if len(bsent) == 0:
            continue
        res1 = pattern1.findall(bsent) #获得bit串中的连续01序列，得到列表
        res2 = pattern2.findall(bsent) #获得bit串中的连续非01序列，得到列表
        #判断bit串是否以01开头，用于合并01序列和非01序列
        #pos表示非01序列列表中待加入最终序列的当前位置
        len_res2 = len(res2)
        r, pos = "", 0
        if bsent[0] not in ['0', '1']:
            r, pos = res2[0], 1
        #逐个解码01序列列表中的01序列，原则是使解码的字符串占用的bit数最多
        for bs in res1:
            d = {}
            chs, lchs = '', 0 #chs是当前最佳，lchs是chs占用的bit数
            for i in range(0, 8):
                d[i] = [0, '']
            #从第8位开始处理
            for i in range(8, len(bs)+1):
                for j in [8, 12 ,16]:
                    if i-j<0:
                        continue
                    if bs[i-j:i] in valid_vocab:
                        if d[i-j][0]+j>lchs:
                            lchs, chs = d[i-j][0]+j, d[i-j][1]+valid_vocab[bs[i-j:i]]
                    elif d[i-1][0]>lchs:
                        lchs, chs = d[i-1]
                d[i] = [lchs, chs]
            unused += len(bs)-lchs
            r += chs
            if pos < len_res2:
                r += res2[pos]
                pos += 1
        r = r.replace('<>', ' ').replace('０', '0').replace('１', '1')
        print(r)
        if unused:
            unusedlist.append(unused)
    if sum(unusedlist):
        warnings.warn("\nTotal:{}\n{}".format(sum(unusedlist), unusedlist))

#将train.bpe.vi转为train.bpe.vi.re，合并其中可以合并的字符
def convbpebit2charbit(originfile, bpefile):
    print("convbpebit2charbit...")
    orgsents = open(originfile, "r", encoding="utf-8")
    bpesents = open(bpefile, "r", encoding="utf-8")
    fout = open(bpefile+".chbit", "w", encoding="utf-8")
    for osent in orgsents:
        #读取源句子和bpebit句子
        osent = osent.strip()
        bsent = bpesents.readline().strip()
        #将源句子按字符转为bit串,保存在blist中
        charlist = list(osent) #句子的字符列表
        bitslist = [] #字符列表对应的bits列表
        u8list = [hex(ord(i))[2:] for i in charlist]
        for Bytes in u8list:
            bits = ""
            for byte in Bytes:
                bits += bitlist[byte]
            bitslist.append(bits)
        #将源句子转为bit串，并与bpe后的bit串进行比较
        assert "".join(bitslist) == bsent.replace("@@ ", "")
        #利用正则匹配，将bsent中的bit串中可解码的串解码为字符
        pattern = re.compile(r"[01]")
        res, s = "", 0
        for i in range(len(osent)):
            length = len(bitslist[i])
            num = len(pattern.findall(bsent[s:s+length]))
            j = 0
            if num != length:
                while num != length:
                    j += 1
                    num = len(pattern.findall(bsent[s:s+length+j*3]))
                res += bsent[s:s+length+j*3]
            else:
                if charlist[i] == '1':
                    res += '１'
                elif charlist[i] == '0':
                    res += '０'
                elif charlist[i] == ' ':
                    res += '<>'
                else:
                    res += charlist[i]
            s += length+j*3
        #将解码后的句子写入文件
        fout.write(res+"\n")

def build_vocab(path):
    print("build vocab ...")
    filelist = ["train.vi", "tst2012.vi", "tst2013.vi"]
    charset = set()
    for name in filelist:
        orgsents = open(path+name, "r", encoding="utf-8")
        for osent in orgsents:
            charset.update(osent.strip())

    valid_vocab = {}
    for ch in charset:
        u8 = hex(ord(ch))[2:]
        bits = ""
        for _byte in u8:
            bits += bitlist[_byte]
        if bits not in valid_vocab:
            valid_vocab[bits] = ch 

    #将bit串字符词表写入文件
    with open("valid_vocab.json", "w", encoding="utf-8") as f:
        json.dump(valid_vocab, f, indent=2, ensure_ascii=False)

if __name__=="__main__":
    debug = False
    if debug:
        #build_vocab("/media/ntfs-3/EXP/bit2bit/baseExp/data/dev/")
        train_vi = "/media/ntfs-3/EXP/bit2bit/bit2bit/data/tst2012.en"
        convbpebit2charbit(train_vi, sys.argv[1])
        #path = "/media/ntfs-3/EXP/bit2bit/baseExp/data/test/"
        #path = sys.argv[1]
        #vocab_path = "/media/ntfs-3/EXP/bit2bit/baseExp/data/dev/valid_vocab.json"
        #debug_bit2char(path, vocab_path)
    else:
        #get_information(sys.argv[1])
        #英文自带空格，保留空格
        #中文不使用空格
        char2bit(sys.argv[1], keepblank=True)
        #bit2u8(sys.argv[1]+".bit")
