## fast_align_sh

    用途：使用fast_align工具过滤语料的脚本
    
    需要提前安装fast_align，并添加到/usr/bin等目录下
    
    提取前90%文本，示例：
    
    ./fast_align_sh/run sourcefile targetfile 90

## bleu-score.py

    用途：同multi-bleu-detok.perl/multi-bleu.perl，用于获取candidate和reference的BLEU值
    上述只支持与一个reference比较，需要先reference再candidate, 顺序不可改变
    示例：
    # no tokenize
    ./bleu-score.py reference candidate
    # tokenize
    ./bleu-score.py reference candidate -tok
