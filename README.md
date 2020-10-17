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

6：进行训练 sh scripts/run_ner_crf.sh

6：训练结束后，查看checkpoint文件是否保存

7：运行服务即可 python http_server.py