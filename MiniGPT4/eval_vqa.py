import os
import re
import json
import argparse
from collections import defaultdict

import numpy as np
from PIL import Image
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader

from minigpt4.datasets.datasets.vqa_datasets import POPEDetectionEvalData,PHDDETEvalData
from minigpt4.common.vqa_tools.VQA.PythonHelperTools.vqaTools.vqa import VQA
from minigpt4.common.vqa_tools.VQA.PythonEvaluationTools.vqaEvaluation.vqaEval import VQAEval

from minigpt4.common.eval_utils import prepare_texts, init_model, eval_parser,prepare_texts_detection,prepare_texts_detection_zero
from minigpt4.conversation.conversation import CONV_VISION_minigptv2
from minigpt4.common.config import Config


def list_of_str(arg):
    return list(map(str, arg.split(',')))

parser = eval_parser()
parser.add_argument("--dataset", type=list_of_str, default='refcoco', help="dataset to evaluate")
parser.add_argument("--prompt_embedding", type=str, help="soft prompt embedding")
args = parser.parse_args()
cfg = Config(args)

model, vis_processor = init_model(args)
conv_temp = CONV_VISION_minigptv2.copy()
conv_temp.system = ""
model.eval()

save_path = cfg.run_cfg.save_path
soft_path = cfg.config.model.prompt_embedding

filename = soft_path.split('/')[-1]
filename_without_ext = filename.split('.')[0]
epoch = filename_without_ext.split('_')[-1]

if 'pope' in args.dataset:
    name = 'adversarial'

    eval_file_path = cfg.evaluation_datasets_cfg["pope"]["eval_file_path"] #数据集
    img_path = cfg.evaluation_datasets_cfg["pope"]["img_path"] #图像路径
    batch_size = cfg.evaluation_datasets_cfg["pope"]["batch_size"]
    max_new_tokens = cfg.evaluation_datasets_cfg["pope"]["max_new_tokens"]

    with open(eval_file_path, 'r') as file1:
        pope_test_split =json.load(file1)
    data = POPEDetectionEvalData(pope_test_split, vis_processor, img_path)
    eval_dataloader = DataLoader(data, batch_size=batch_size, shuffle=False)
    minigpt4_predict = []

    total_time = 0
    num_runs=len(data)

    for imgs, images, questions, question_ids, objects, labels, category, subcategory in tqdm(eval_dataloader):
        # zero-shot inference
        # texts = prepare_texts(questions, conv_temp)  # warp the texts with conversation template
        # answers,time = model.generate(images, texts, max_new_tokens=max_new_tokens, do_sample=False)

        # add det zero-shot inference
        # texts = prepare_texts_detection_zero(questions, conv_temp)  # warp the texts with conversation template
        # answers,time = model.generate_det(images, texts, objects, max_new_tokens=max_new_tokens, do_sample=False)

        # add virtual token inference
        texts = prepare_texts_detection(questions, conv_temp)  # warp the texts with conversation template
        answers,time = model.generate_det_softprompt(soft_path, images, texts, objects, max_new_tokens=max_new_tokens, do_sample=False)
        
        #计算时间
        total_time +=time

        for answer, question, question_id, img_id, label in zip(answers, questions, question_ids, imgs, labels):
            result = dict()
            answer = answer.lower().replace('<unk>','').strip()
            result['answer'] = answer
            result['question'] = question
            result['question_id'] = str(int(question_id))
            result['image'] = img_id.split('/')[-1]
            result['label'] = label
            minigpt4_predict.append(result)

    print('inference time:', total_time/num_runs)
    
    file_save_path= os.path.join(save_path,"pope_{}_epoch{}.json".format(name,epoch))

    if not os.path.exists(file_save_path):
        os.makedirs(os.path.dirname(file_save_path), exist_ok=True)

    with open(file_save_path,'w') as f:
        json.dump(minigpt4_predict, f)
    
    pred_list=[]
    label_list=[]
    q_list=[]
    catlist=[]
    sublist=[]
    for item in minigpt4_predict:
        answer_pope={}
        text = item['answer']
        label = item['label']
        q_id = item['question_id']
        catlist.append(item['category'])
        sublist.append(item['subcategory'])
        if label == 'no':
            label_list.append(0)
        else:
            label_list.append(1)

        if text.find('.') != -1:
            text = text.split('.')[0]
        text = text.replace(',', '')
        words = text.split(' ')
        if 'No' in words or 'not' in words or 'no' in words or 'cannot' in words or 'none' in words:
            answer_pope['answer'] = 'no'
            pred_list.append(0)
        else:
            answer_pope['answer'] = 'yes'
            pred_list.append(1)
        q_list.append(q_id)
    
    pos = 1
    neg = 0
    yes_ratio = pred_list.count(1) / len(pred_list)
    
    wrong=[]
    TP, TN, FP, FN = 0, 0, 0, 0
    
    for pred, label,qid,cats,subs in zip(pred_list, label_list,q_list,catlist,sublist):

        if pred == pos and label == pos:
            TP += 1
        elif pred == pos and label == neg:
            FP += 1
            wrong.append(qid)
        elif pred == neg and label == neg:
            TN += 1
        elif pred == neg and label == pos:
            FN += 1
            wrong.append(qid)

    print('wrong_number：',len(wrong))

    print('TP\tFP\tTN\tFN\t')
    print('{}\t{}\t{}\t{}'.format(TP, FP, TN, FN))

    precision = float(TP) / float(TP + FP)
    recall = float(TP) / float(TP + FN)
    f1 = 2*precision*recall / (precision + recall)
    acc = (TP + TN) / (TP + TN + FP + FN)
    print('Accuracy: {}'.format(acc))
    print('Precision: {}'.format(precision))
    print('Recall: {}'.format(recall))
    print('F1 score: {}'.format(f1))
    print('Yes ratio: {}'.format(yes_ratio))


if 'phd' in args.dataset:

    eval_file_path = cfg.evaluation_datasets_cfg["phd"]["eval_file_path"]
    img_path = cfg.evaluation_datasets_cfg["phd"]["img_path"]
    batch_size = cfg.evaluation_datasets_cfg["phd"]["batch_size"]
    max_new_tokens = cfg.evaluation_datasets_cfg["phd"]["max_new_tokens"]
    
    with open(eval_file_path, 'r') as file1:
        phd_test =json.load(file1)

    data = PHDDETEvalData(phd_test, vis_processor, img_path)
    eval_dataloader = DataLoader(data, batch_size=batch_size, shuffle=False)
    minigpt4_predict = []
    total_time = 0
    num_runs=len(data)

    for imgs, images, questions, question_ids, objects, labels in tqdm(eval_dataloader):
        # zero-shot inference
        # texts = prepare_texts(questions, conv_temp)  # warp the texts with conversation template
        # answers = model.generate(images, texts, max_new_tokens=max_new_tokens, do_sample=False)

        # add det zero-shot inference
        # texts = prepare_texts_detection_zero(questions, conv_temp)  # warp the texts with conversation template
        # answers = model.generate_det(images, texts, objects, max_new_tokens=max_new_tokens, do_sample=False)

        # add virtual token inference
        texts = prepare_texts_detection(questions, conv_temp)  # warp the texts with conversation template
        answers,time = model.generate_det_softprompt(soft_path, images, texts, objects, max_new_tokens=max_new_tokens, do_sample=False)
        total_time +=time

        for answer, question, question_id, img_id, label in zip(answers, questions, question_ids, imgs, labels):
            result = dict()
            answer = answer.lower().replace('<unk>','').strip()
            result['answer'] = answer
            result['question'] = question
            result['question_id'] = str(int(question_id))
            result['image'] = img_id.split('/')[-1]
            result['label'] = label
            minigpt4_predict.append(result)

    print('inference time:', total_time/num_runs)

    file_save_path=os.path.join(save_path,"phd_test_epoch{}.json".format(epoch))

    with open(file_save_path,'w') as f:
        json.dump(minigpt4_predict, f)

    pred_list=[]
    label_list=[]
    q_list=[]
    for item in minigpt4_predict:
        answer_phd={}
        text = item['answer']
        label = item['label']
        q_id = item['question_id']
        if label == 'no':
            label_list.append(0)
        else:
            label_list.append(1)

        if text.find('.') != -1:
            text = text.split('.')[0]
        text = text.replace(',', '')
        words = text.split(' ')
        if 'No' in words or 'not' in words or 'no' in words or 'cannot' in words or 'none' in words:
            answer_phd['answer'] = 'no'
            pred_list.append(0)
        else:
            answer_phd['answer'] = 'yes'
            pred_list.append(1)
        q_list.append(q_id)
    
    pos = 1
    neg = 0
    yes_ratio = pred_list.count(1) / len(pred_list)
    
    wrong=[]
    TP, TN, FP, FN = 0, 0, 0, 0
    for pred, label, qid in zip(pred_list, label_list, q_list):
        if pred == pos and label == pos:
            TP += 1
        elif pred == pos and label == neg:
            FP += 1
            wrong.append(qid)
        elif pred == neg and label == neg:
            TN += 1
        elif pred == neg and label == pos:
            FN += 1
            wrong.append(qid)

    print('wrong_number：',len(wrong))

    print('TP\tFP\tTN\tFN\t')
    print('{}\t{}\t{}\t{}'.format(TP, FP, TN, FN))

    precision = float(TP) / float(TP + FP)
    recall = float(TP) / float(TP + FN)
    f1 = 2*precision*recall / (precision + recall)
    acc = (TP + TN) / (TP + TN + FP + FN)
    print('Accuracy: {}'.format(acc))
    print('Precision: {}'.format(precision))
    print('Recall: {}'.format(recall))
    print('F1 score: {}'.format(f1))
    print('Yes ratio: {}'.format(yes_ratio))
