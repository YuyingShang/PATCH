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
from datasets import load_dataset


from minigpt4.datasets.datasets.vqa_datasets import OKVQAEvalData,VizWizEvalData,IconQAEvalData,GQAEvalData,VSREvalData,HMEvalData,POPEDetectionEvalData,OKVQADETEvalData,PHDDETEvalData,VQADETEvalData
from minigpt4.common.vqa_tools.VQA.PythonHelperTools.vqaTools.vqa import VQA
from minigpt4.common.vqa_tools.VQA.PythonEvaluationTools.vqaEvaluation.vqaEval import VQAEval

from minigpt4.common.eval_utils import prepare_texts, init_model, eval_parser,prepare_texts_detection,prepare_texts_detection_zero
from minigpt4.conversation.conversation import CONV_VISION_LLama2
from minigpt4.common.config import Config


def list_of_str(arg):
    return list(map(str, arg.split(',')))

parser = eval_parser()
parser.add_argument("--dataset", type=list_of_str, default='refcoco', help="dataset to evaluate")
parser.add_argument("--prompt_embedding", type=str, help="soft prompt embedding")
args = parser.parse_args()
cfg = Config(args)

model, vis_processor = init_model(args)
conv_temp = CONV_VISION_LLama2.copy()
conv_temp.system = ""
model.eval()

#修改参数
print(model)
save_path = cfg.run_cfg.save_path
soft_path = cfg.config.model.prompt_embedding


#保存的时候按epoch存
filename = soft_path.split('/')[-1]
# 去掉文件扩展名
filename_without_ext = filename.split('.')[0]
# 拆分文件名并获取最后一个部分的数字
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

    for imgs, images, questions, question_ids, objects, labels in tqdm(eval_dataloader):
        #原始推理
        # texts = prepare_texts(questions, conv_temp)  # warp the texts with conversation template
        # answers,time = model.generate(images, texts, max_new_tokens=max_new_tokens, do_sample=False)

        #添加det进行推理
        texts = prepare_texts_detection_zero(questions, conv_temp)  # warp the texts with conversation template
        answers,time = model.generate_det(images, texts, objects, max_new_tokens=max_new_tokens, do_sample=False)

        #添加softprompt后进行推理
        # texts = prepare_texts_detection(questions, conv_temp)  # warp the texts with conversation template
        # answers,time = model.generate_det_softprompt(soft_path, images, texts, objects, max_new_tokens=max_new_tokens, do_sample=False)
        
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
    
    file_save_path= os.path.join(save_path,"pope_zs_det_{}_epoch{}.json".format(name,epoch))
    # #file_save_path= os.path.join(save_path,"pope_origin_promopt2.json")
    with open(file_save_path,'w') as f:
        json.dump(minigpt4_predict, f)
    
    pred_list=[]
    label_list=[]
    q_list=[]
    for item in minigpt4_predict:
        answer_pope={}
        text = item['answer']
        label = item['label']
        q_id = item['question_id']
        if label == 'no':
            label_list.append(0)
        else:
            label_list.append(1)

        # Only keep the first sentence 检查是否存在句号，如果存在，则保留句号之前的部分；然后去除逗号；最后将处理后的字符串按空格拆分成单词，并将单词存储在words变量中。
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
    for pred, label,qid in zip(pred_list, label_list,q_list):
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

    # print('错误的题数为：',len(wrong))
    # with open('/home/tianyu/syy/essay3/MiniGPT-4-main_origin/output/pope/wrong_id/pope_{}_epoch{}.json'.format(name,epoch),'w') as file:
    #     json.dump(wrong,file)
    # with open('/home/tianyu/syy/essay3/MiniGPT-4-main_origin/output/pope/wrong_id/pope_origin_promopt2.json','w') as file:
    #     json.dump(wrong,file)


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

if 'vqa' in args.dataset:

    eval_file_path = cfg.evaluation_datasets_cfg["vqa"]["eval_file_path"] #数据集
    img_path = cfg.evaluation_datasets_cfg["vqa"]["img_path"] #图像路径
    batch_size = cfg.evaluation_datasets_cfg["vqa"]["batch_size"]
    max_new_tokens = cfg.evaluation_datasets_cfg["vqa"]["max_new_tokens"]

    with open(eval_file_path, 'r') as file1:
        vqa_test_split =json.load(file1)
    data = VQADETEvalData(vqa_test_split, vis_processor, img_path)
    eval_dataloader = DataLoader(data, batch_size=batch_size, shuffle=False)
    minigpt4_predict = []

    total_time = 0
    num_runs=len(data)

    for imgs, images, questions, question_ids, objects in tqdm(eval_dataloader):
        #原始推理
        # texts = prepare_texts(questions, conv_temp)  # warp the texts with conversation template
        # answers,time = model.generate(images, texts, max_new_tokens=max_new_tokens, do_sample=False)

        #添加det进行推理
        # texts = prepare_texts_detection_zero(questions, conv_temp)  # warp the texts with conversation template
        # answers,time = model.generate_det(images, texts, objects, max_new_tokens=max_new_tokens, do_sample=False)

        #添加softprompt后进行推理
        texts = prepare_texts_detection(questions, conv_temp)  # warp the texts with conversation template
        answers,time = model.generate_det_softprompt(soft_path, images, texts, objects, max_new_tokens=max_new_tokens, do_sample=False)
        
        #计算时间
        total_time +=time

        for answer, question, question_id, img_id in zip(answers, questions, question_ids, imgs):
            result = dict()
            answer = answer.lower().replace('<unk>','').strip()
            result['answer'] = answer
            result['question'] = question
            result['question_id'] = str(int(question_id))
            result['image'] = img_id.split('/')[-1]
            minigpt4_predict.append(result)

    print('inference time:', total_time/num_runs)
    
    file_save_path= os.path.join(save_path,"vqa_epoch{}.json".format(epoch))
    with open(file_save_path,'w') as file2:
         json.dump(minigpt4_predict, file2)

    annFile = '/home/tianyu/syy/essay3/dataset/vqa/v2_mscoco_val2014_annotations.json'
    quesFile = '/home/tianyu/syy/essay3/dataset/vqa/v2_OpenEnded_mscoco_val2014_questions.json'

    vqa = VQA(annFile, quesFile)
    vqaRes = vqa.loadRes(file_save_path, quesFile)
    vqaEval = VQAEval(vqa, vqaRes, n=2)
    vqaEval.evaluate()
    print ("Overall VQAv2 Accuracy is: %.02f\n" %(vqaEval.accuracy['overall']), flush=True)
    print ("perQuestionType is:", vqaEval.accuracy['perQuestionType'])
    print ("perAnswerType is:", vqaEval.accuracy['perAnswerType'])

    print('Precision: {}'.format(vqaEval.precision))
    print('Recall: {}'.format(vqaEval.recall))
    print('F1 score: {}'.format(vqaEval.f1))
    

if 'okvqa' in args.dataset:

    eval_file_path = cfg.evaluation_datasets_cfg["okvqa"]["eval_file_path"]
    img_path = cfg.evaluation_datasets_cfg["okvqa"]["img_path"]
    batch_size = cfg.evaluation_datasets_cfg["okvqa"]["batch_size"]
    max_new_tokens = cfg.evaluation_datasets_cfg["okvqa"]["max_new_tokens"]
    
    with open(eval_file_path, 'r') as file1:
        ok_vqa_test_split =json.load(file1)

    data = OKVQADETEvalData(ok_vqa_test_split, vis_processor, img_path)
    eval_dataloader = DataLoader(data, batch_size=batch_size, shuffle=False)
    minigpt4_predict = []

    for imgs, images, questions, question_ids, objects in tqdm(eval_dataloader):
        #原始推理
        texts = prepare_texts(questions, conv_temp)  # warp the texts with conversation template
        answers = model.generate(images, texts, max_new_tokens=max_new_tokens, do_sample=False)

        #添加det进行推理
        # texts = prepare_texts_detection_zero(questions, conv_temp)  # warp the texts with conversation template
        # answers = model.generate_det(images, texts, objects, max_new_tokens=max_new_tokens, do_sample=False)

        #添加softprompt后进行推理
        # texts = prepare_texts_detection(questions, conv_temp)  # warp the texts with conversation template
        # answers = model.generate_det_softprompt(soft_path, vitual_num, images, texts, objects, max_new_tokens=max_new_tokens, do_sample=False)

        for answer, question_id, question, img_id in zip(answers, question_ids, questions, imgs):
            result = dict()
            answer = answer.lower().replace('<unk>','').strip()
            result['answer'] = answer
            result['question_id'] = int(question_id)
            result['question'] = question
            result['image'] = img_id
            minigpt4_predict.append(result)

    file_save_path=os.path.join(save_path,"okvqa_zeroshot_basedon_det.json")
    with open(file_save_path,'w') as f:
        json.dump(minigpt4_predict, f)

    annFile = '/home/tianyu/syy/essay3/dataset/okvqa/mscoco_val2014_annotations_clean.json'
    quesFile = '/home/tianyu/syy/essay3/dataset/okvqa/OpenEnded_mscoco_val2014_questions_clean.json'

    vqa = VQA(annFile, quesFile)
    vqaRes = vqa.loadRes(file_save_path, quesFile)

    vqaEval = VQAEval(vqa, vqaRes, n=2)
    vqaEval.evaluate()
    print ("Overall OKVQA Accuracy is: %.02f\n" %(vqaEval.accuracy['overall']), flush=True)


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
        #原始推理
        # texts = prepare_texts(questions, conv_temp)  # warp the texts with conversation template
        # answers = model.generate(images, texts, max_new_tokens=max_new_tokens, do_sample=False)

        #添加det进行推理
        # texts = prepare_texts_detection_zero(questions, conv_temp)  # warp the texts with conversation template
        # answers = model.generate_det(images, texts, objects, max_new_tokens=max_new_tokens, do_sample=False)

        #添加softprompt后进行推理
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
    #file_save_path=os.path.join(save_path,"phd_det_test_detnew_prompt0.json")
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

        # Only keep the first sentence 检查是否存在句号，如果存在，则保留句号之前的部分；然后去除逗号；最后将处理后的字符串按空格拆分成单词，并将单词存储在words变量中。
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

    print('错误的题数为：',len(wrong))
    

    # wrong_question=[]
    # for qs in tqdm(wrong):
    #     for item in minigpt4_predict:
    #         if qs==item['question_id']:
    #             wrong_question.append(item)
    # with open('/xmnt/mnt_nfs_qynas_v4/tianyu/syy/phd/wrong_id/phd_test_epoch{}.json'.format(epoch),'w') as file:
    # #with open('/xmnt/mnt_nfs_qynas_v4/tianyu/syy/phd/wrong_id/phd_test_detnew_prompt0.json','w') as file:
    #     json.dump(wrong,file)


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

if 'vizwiz' in args.dataset:

    eval_file_path = cfg.evaluation_datasets_cfg["vizwiz"]["eval_file_path"]
    img_path = cfg.evaluation_datasets_cfg["vizwiz"]["img_path"]
    batch_size = cfg.evaluation_datasets_cfg["vizwiz"]["batch_size"]
    max_new_tokens = cfg.evaluation_datasets_cfg["vizwiz"]["max_new_tokens"]

    vizwiz = json.load(open(eval_file_path, 'r'))

    data = VizWizEvalData(vizwiz, vis_processor, img_path)
    eval_dataloader = DataLoader(data, batch_size=batch_size, shuffle=False)
    minigpt4_predict = []
    total_acc = []
    for images, texts, gt_answers in tqdm(eval_dataloader):
        texts = prepare_texts(texts, conv_temp)  # warp the texts with conversation template
        with torch.no_grad():
            answers = model.generate(images, texts, max_new_tokens=max_new_tokens, do_sample=False,repetition_penalty=1.0)

        for answer, gt_answer in zip(answers, gt_answers):
            result = dict()
            result['answer'] = answer.replace('<unk>','').strip()
            minigpt4_predict.append(result)
            count=0
            gt_answer = gt_answer.split('_')
            for gt in gt_answer:
                if gt.lower() == answer.lower():
                    count += 1
            acc = min(count/3.0, 1.0)
            total_acc.append(acc)
        
    file_save_path = os.path.join(save_path, "vizwiz.json")
    with open(file_save_path,'w') as f:
        json.dump(minigpt4_predict, f)
    print('vizwiz Acc: ', np.average(total_acc)* 100.0, flush=True)


if 'iconvqa' in args.dataset:

    eval_file_path = cfg.evaluation_datasets_cfg["iconvqa"]["eval_file_path"]
    img_path = cfg.evaluation_datasets_cfg["iconvqa"]["img_path"]
    batch_size = cfg.evaluation_datasets_cfg["iconvqa"]["batch_size"]
    max_new_tokens = cfg.evaluation_datasets_cfg["iconvqa"]["max_new_tokens"]

    iconqa_text_val = json.load(open(eval_file_path,"r"))

    data = IconQAEvalData(iconqa_text_val, vis_processor, img_path)
    eval_dataloader = DataLoader(data, batch_size=batch_size, shuffle=False)

    count = 0
    for images, texts, candidates, answers in tqdm(eval_dataloader):
        candidates = [candidate.split('_') for candidate in candidates]
        num_cand = [len(candidate) for candidate in candidates]
        for candidate in candidates:
            candidate.extend(['none'] * (max(num_cand) - len(candidate)))
        candidates = [list(x) for x in zip(*candidates)]
        instructions = ["<s>[INST] <Img><ImageHere></Img> {} [/INST]".format(text) for text in texts]
        answer_ranks = model.multi_select(images, instructions, candidates, num_cand=num_cand)
        for idx, answer in enumerate(answers):
            if answer_ranks[idx][0] == answer:
                count += 1

    print('iconqa Acc: ', count / len(iconqa_text_val) * 100.0, flush=True)


if 'gqa' in args.dataset:

    eval_file_path = cfg.evaluation_datasets_cfg["gqa"]["eval_file_path"]
    img_path = cfg.evaluation_datasets_cfg["gqa"]["img_path"]
    batch_size = cfg.evaluation_datasets_cfg["gqa"]["batch_size"]
    max_new_tokens = cfg.evaluation_datasets_cfg["gqa"]["max_new_tokens"]

    gqa = json.load(open(eval_file_path))
    data = GQAEvalData(gqa, vis_processor, img_path)
    eval_dataloader = DataLoader(data, batch_size=batch_size, shuffle=False)
    count=0
    total=0
    minigpt4_predict = []
    for images, texts, labels in tqdm(eval_dataloader):
        texts = prepare_texts(texts, conv_temp)  # warp the texts with conversation template
        answers = model.generate(images, texts, max_new_tokens=max_new_tokens, do_sample=False)

        for answer, label in zip(answers, labels):
            result = dict()
            result['pred'] = answer.lower().replace('<unk>','').strip()
            result['gt'] = label
            minigpt4_predict.append(result)
            if answer.lower() == label:
                count+=1
            total+=1
    print('gqa val:', count / total * 100, flush=True)

    file_save_path = os.path.join(save_path, "gqa.json")
    with open(file_save_path,'w') as f:
        json.dump(minigpt4_predict, f)

if 'vsr' in args.dataset:

    img_path = cfg.evaluation_datasets_cfg["vsr"]["img_path"]
    batch_size = cfg.evaluation_datasets_cfg["vsr"]["batch_size"]
    max_new_tokens = cfg.evaluation_datasets_cfg["vsr"]["max_new_tokens"]

    annotation = load_dataset("cambridgeltl/vsr_zeroshot", split='test')
    data = VSREvalData(annotation, vis_processor, img_path)
    eval_dataloader = DataLoader(data, batch_size=batch_size, shuffle=False)
    count=0
    total=0

    minigpt4_predict = []

    for images, texts, labels in tqdm(eval_dataloader):
        texts = prepare_texts(texts, conv_temp)  # warp the texts with conversation template
        answers = model.generate(images, texts, max_new_tokens=max_new_tokens, do_sample=False)

        for answer, label in zip(answers, labels):
            result = dict()
            result['pred'] = answer.replace('<unk>','').strip()
            result['gt'] = label
            minigpt4_predict.append(result)
            if answer.lower() ==  label.lower():
                count+=1
            total+=1
    print('vsr test:', count / total * 100, flush=True)
    file_save_path = os.path.join(save_path,"vsr.json")
    with open(file_save_path,'w') as f:
        json.dump(minigpt4_predict, f)

if 'hm' in args.dataset:

    eval_file_path = cfg.evaluation_datasets_cfg["hm"]["eval_file_path"]
    img_path = cfg.evaluation_datasets_cfg["hm"]["img_path"]
    batch_size = cfg.evaluation_datasets_cfg["hm"]["batch_size"]
    max_new_tokens = cfg.evaluation_datasets_cfg["hm"]["max_new_tokens"]

    annotation = []
    with open(eval_file_path, 'r') as jsonl_file:
        for line in jsonl_file:
            json_obj = json.loads(line)
            annotation.append(json_obj)

    data = HMEvalData(annotation, vis_processor, img_path)
    eval_dataloader = DataLoader(data, batch_size=batch_size, shuffle=False)
    count=0
    total=0

    minigpt4_predict = []

    for images, texts, labels in tqdm(eval_dataloader):
        texts = prepare_texts(texts, conv_temp)  # warp the texts with conversation template
        
        answers = model.generate(images, texts, max_new_tokens=max_new_tokens, do_sample=False)

        for answer, label in zip(answers, labels):
            result = dict()
            if answer.lower().strip() =="yes":
                answer=1
            elif answer.lower().strip()=="no":
                answer=0
            else:
                print("non-matching answer",answer)

            result['pred'] = answer
            result['gt'] = int(label)
            minigpt4_predict.append(result)
            if answer == label:
                count+=1
            total+=1

    print('hm val:', count / total * 100, flush=True)
    file_save_path = os.path.join(save_path, "hm.json")
    with open(file_save_path,'w') as f:
        json.dump(minigpt4_predict, f)
