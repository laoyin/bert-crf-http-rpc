import glob
import logging
import os
import json
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from callback.optimizater.adamw import AdamW
from callback.lr_scheduler import get_linear_schedule_with_warmup
from callback.progressbar import ProgressBar
from tools.common import seed_everything,json_to_text
from tools.common import init_logger, logger

from models.transformers import WEIGHTS_NAME, BertConfig, AlbertConfig
from models.bert_for_ner import BertCrfForNer
from models.albert_for_ner import AlbertCrfForNer
from processors.utils_ner import CNerTokenizer, get_entities
from processors.ner_seq import convert_examples_to_features
from processors.ner_seq import ner_processors as processors
from processors.ner_seq import collate_fn
from metrics.ner_metrics import SeqEntityScore
from tools.finetuning_argparse import get_argparse

MODEL_CLASSES = {
    ## bert ernie bert_wwm bert_wwwm_ext
    'bert': (BertConfig, BertCrfForNer, CNerTokenizer),
    'albert': (AlbertConfig, AlbertCrfForNer, CNerTokenizer)
}

from flask import Flask, request, url_for, send_from_directory

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.getcwd()
# app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024



@app.route("/add_outbound_call", methods=['POST'])
def add_outbound_call():
    task_id = request.args.get("TaskId", "")
    phone = request.args.get("Phone", "")
    application_id = request.args.get("ApplicationId")
    outbound_manage_id = request.args.get("OutboundManageId")

    return '{"code":"200", "msg":"ok"}'




@app.route("/get_court_msg", methods=['POST'])
def get_court_msg():
    model.eval()
    undo_text = request.args.get("undo_text", "")
    sen_code = tokenizer.encode_plus(undo_text)
    print(sen_code)
    print(len(sen_code["token_type_ids"]))
    sen_code["attention_mask"] = [1] * len(sen_code["token_type_ids"])
    print(sen_code)

    with torch.no_grad():
        # inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": None, 'input_lens': batch[4]}
        # inputs["token_type_ids"] = (batch[2] if "bert" in ["bert", "xlnet"] else None)
        inputs = {
            "input_ids": torch.tensor([sen_code["input_ids"]]),
            "token_type_ids": torch.tensor([sen_code["token_type_ids"]]),
            "attention_mask": torch.tensor([sen_code["attention_mask"]])
        }
        outputs = model(**inputs)
        logits = outputs[0]
        tags = model.crf.decode(logits, inputs['attention_mask'])
        tags = tags.squeeze(0).cpu().numpy().tolist()
    preds = tags[0][1:-1]  # [CLS]XXXX[SEP]
    print(preds)
    label_entities = get_entities(preds, id2label)
    print(label_entities)
    return_list = ""
    for index, label in enumerate(preds):
        print("label",label, index)
        if label == 13:
            return_list += undo_text[index]
    return '{"code":"200", "msg":%s}'%return_list

task_name = "cner"
processor = processors[task_name]()
label_list = processor.get_labels()
id2label = {i: label for i, label in enumerate(label_list)}
label2id = {label: i for i, label in enumerate(label_list)}
num_labels = len(label_list)

config_class, model_class, tokenizer_class = MODEL_CLASSES["bert"]
config = config_class.from_pretrained("prev_trained_model/bert-base",
                                      num_labels=num_labels, cache_dir=None)
tokenizer = tokenizer_class.from_pretrained("outputs/cner_output/bert", do_lower_case=True)
# # checkpoints = [args.output_dir]
ck_output_dir = "outputs/cner_output"
checkpoints = ["outputs/cner_output"]
# if args.predict_checkpoints > 0:
checkpoints = list(
    os.path.dirname(c) for c in sorted(glob.glob(ck_output_dir + '/**/' + WEIGHTS_NAME, recursive=True)))
logging.getLogger("transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging

# for checkpoint in checkpoints:
#     print(1)
#     print(checkpoint)
model = model_class.from_pretrained("outputs/cner_output/bert/checkpoint-448", config=config)

model.to("cpu")


results = []

if isinstance(model, nn.DataParallel):
    model = model.module
# for step, batch in enumerate(test_dataloader):
model.eval()

# app.run(port=9909)