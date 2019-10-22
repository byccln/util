#!/bin/env python3
#每一组的句长差为5

import numpy as np
from bleu_score import get_bleu
import matplotlib.pyplot as plt

reference_file = "/media/ntfs-5/EXP/ENZH/data/test/test2.zh"
candidate_file = "/media/ntfs-5/EXP/ENZH/bpe2bpe/data/test/test2.out4"
bleu_write_file = "bleu"
group_number = 2

reference = open(reference_file, "r", encoding="utf-8").readlines()
candidate = open(candidate_file, "r", encoding="utf-8").readlines()

assert len(reference) == len(candidate)
data_length = len(reference)

sentence_length = [len(line.strip().split()) for line in reference]
sorted_idx = np.argsort(sentence_length)
sorted_sentence_length = [sentence_length[i] for i in sorted_idx[:]]
sorted_reference = [reference[i] for i in sorted_idx[:]]
sorted_candidate = [candidate[i] for i in sorted_idx[:]]

next_points = []
for i in range(1, data_length):
    if sorted_sentence_length[i]%group_number==0 and sorted_sentence_length[i]>sorted_sentence_length[i-1]:
        next_points.append(i)
start_points = [0] + next_points
next_points.append(data_length)

bleudict = {"num":[], "len":[], "bleu":[], "1gram":[], "2gram":[], "3gram":[], "4gram":[], "bp":[]}

for s, n in zip(start_points, next_points):
    bleu, _1g, _2g, _3g, _4g, bp = get_bleu(sorted_reference[s:n], sorted_candidate[s:n])
    bleudict["num"].append(n-s)
    bleudict["len"].append(sorted_sentence_length[s])
    bleudict["bleu"].append(bleu)
    bleudict["1gram"].append(_1g)
    bleudict["2gram"].append(_2g)
    bleudict["3gram"].append(_3g)
    bleudict["4gram"].append(_4g)
    bleudict["bp"].append(bp)

fw = open(bleu_write_file, "w", encoding="utf-8")
for i in range(len(bleudict["num"])):
    wstr = "{:3d} {:3d} {:.2f} {:.2f}/{:.2f}/{:.2f}/{:.2f} {:.2f}\n".format(bleudict["len"][i], bleudict["num"][i], bleudict["bleu"][i], bleudict["1gram"][i], bleudict["2gram"][i], bleudict["3gram"][i], bleudict["4gram"][i], bleudict["bp"][i])
    fw.write(wstr)
fw.close()

fig, ax= plt.subplots()
ax.plot(bleudict["len"], bleudict["bleu"], '.:', label="bleu")
ax.plot(bleudict["len"], [num/5/group_number for num in bleudict["num"]], '.', label="(number of sentences)/{}".format(5*group_number), alpha=0.3)
ax.set(ylabel="BLEU", xlabel="sentence length(word)", title="BLEU analysis")
ax.legend()
plt.show()
