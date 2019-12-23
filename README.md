## fast_align_sh

    用途：使用fast_align工具过滤语料的脚本
    
    需要提前安装fast_align，并添加到/usr/bin等目录下
    
    提取前90%文本，示例：
    
    ./fast_align_sh/run sourcefile targetfile 90

## bleu_score.py

    用途：同multi-bleu-detok.perl/multi-bleu.perl，用于获取candidate和reference的BLEU值
    上述只支持与一个reference比较，需要先reference再candidate, 顺序不可改变
    示例：
    # no tokenize
    ./bleu_score.py reference candidate
    # tokenize
    ./bleu_score.py reference candidate -tok

## sort_bleu_n.py
 
    用途：获取candidate和reference的BLEU值，按句长分组，并plot分组后的句长-BLEU关系图
    依赖：bleu_score.py
    使用时需要修改candidate和reference的路径，及group_number(分组间隔)

## zipfs.py

    用途: 获取语料的词频和排名的关系，验证zipf's law，包括word和char两个level
    示例：
        ./zipfs.py filename

## f1.py

    用途: 根据train.trg test.trg ref.trg，得到模型在测试集上all, oov, rare的f1情况
    示例:
        ./f1.py train.en test.en ref.en

## convbit.py

    用途：实现字符串与bit串的互转
