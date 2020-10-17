bert-crf ：https://github.com/lonePatient/BERT-NER-Pytorch

此项目让大家学会跑起bert，bert-crf 并提供出自己想要的服务

crf、bert等介绍



3：python3 -m venv venv 创建指定虚拟环境

4： pip install -r requirement.txt 下载python依赖包

5：下载训练数据，进行训练


数据下载：
链接: https://pan.baidu.com/s/1Hjyo27umeforEJD4zw_1UA 提取码: fvff 复制这段内容后打开百度网盘手机App，操作更方便哦

网盘包含了 bert预训练模型，
训练语料

项目目录结构如下：
-rw-r--r--   1 yinxingpan  staff   1068 Sep  9 14:22 LICENSE
-rw-r--r--   1 yinxingpan  staff    673 Oct 18 07:35 README.md
-rw-r--r--   1 yinxingpan  staff      2 Mar 18  2020 __init__.py
drwxr-xr-x  11 yinxingpan  staff    352 Sep  9 14:22 callback/
drwxr-xr-x   4 yinxingpan  staff    128 Mar 18  2020 datasets/
-rw-r--r--   1 yinxingpan  staff   4279 Oct 18 07:05 http_server.py
drwxr-xr-x   7 yinxingpan  staff    224 Sep  9 14:21 losses/
drwxr-xr-x   5 yinxingpan  staff    160 Sep  9 14:21 metrics/
drwxr-xr-x   8 yinxingpan  staff    256 Sep  9 14:22 models/
drwxr-xr-x   4 yinxingpan  staff    128 Sep 10 22:56 outputs/
drwxr-xr-x   4 yinxingpan  staff    128 Sep  9 14:27 prev_trained_model/
drwxr-xr-x   7 yinxingpan  staff    224 Sep  9 14:22 processors/
-rw-r--r--   1 yinxingpan  staff    127 Oct 17 17:23 requirement.txt
-rw-r--r--   1 yinxingpan  staff  27880 Oct 18 07:06 run_ner_crf.py
-rw-r--r--   1 yinxingpan  staff  25736 Sep  9 14:22 run_ner_softmax.py
-rw-r--r--   1 yinxingpan  staff  28129 Sep  9 14:22 run_ner_span.py
drwxr-xr-x   5 yinxingpan  staff    160 Sep  9 14:32 scripts/
drwxr-xr-x   9 yinxingpan  staff    288 Sep  9 14:22 tools/

从网盘下载 prev_trained_model 放入目录
从网盘下载 datasets放入目录


6：进行训练 sh scripts/run_ner_crf.sh

6：训练结束后，查看checkpoint文件是否保存

7：运行服务即可 python http_server.py