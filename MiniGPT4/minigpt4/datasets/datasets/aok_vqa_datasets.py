"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

from collections import OrderedDict
import json
import os
import random
import torch

from PIL import Image

from minigpt4.datasets.datasets.vqa_datasets import VQADataset  #, VQAEvalDataset


class __DisplMixin:
    def displ_item(self, index):
        sample, ann = self.__getitem__(index), self.annotation[index]
        return OrderedDict(
            {
                "file": ann["image"],
                "question": ann["question"],
                "question_id": ann["question_id"],
                "direct_answers": "; ".join(ann["direct_answers"]),
                "choices": "; ".join(ann["choices"]),
                "correct_choice": ann["choices"][ann["correct_choice_idx"]],
                "image": sample["image"],
            }
        )
    

class __PromptPosMixin:
    def set_position(self, pos='middle'):
        assert pos in ['middle', 'late','nono'], "pos must be in ['middle', 'late']"
        self.pos = pos
        #self.pos = 'nono'
    
    def set_info(self, info='obj+position'):
        assert info in ['obj', 'obj+position'], "info must be in ['obj', 'obj+position']"
        self.info = info


class AOKVQADataset(VQADataset, __DisplMixin):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)

        self.instruction_pool =[
            "[vqa] {}",
            "[vqa] Based on the image, respond to this question with a short answer: {}"
        ]

        exist_annotation = []
        for ann in self.annotation:
            image_path = os.path.join(self.vis_root, ann["image"].split('/')[-1])
            if os.path.exists(image_path):
                exist_annotation.append(ann)
        self.annotation = exist_annotation

    def get_data(self, index):
        ann = self.annotation[index]

        image_path = os.path.join(self.vis_root, ann["image"].split('/')[-1])
        image = Image.open(image_path).convert("RGB")

        image = self.vis_processor(image)
        question = self.text_processor(ann["question"])

        answer_key = "direct_answers"

        answer_weight = {}
        for answer in ann[answer_key]:
            if answer in answer_weight.keys():
                answer_weight[answer] += 1 / len(ann[answer_key])
            else:
                answer_weight[answer] = 1 / len(ann[answer_key])

        answers = list(answer_weight.keys())
        weights = list(answer_weight.values())

        answer = random.choices(answers, weights=weights, k=1)[0]  # random sample an answer according to weights

        return {
            "image": image,
            "question": question,
            "answer": answer,
        }

    def __getitem__(self, index):
        data = self.get_data(index)
        question = self.text_processor(data["question"])
        instruction = random.choice(self.instruction_pool).format(question)

        instruction = "<Img><ImageHere></Img> {} ".format(instruction)
        answer = self.text_processor(data['answer'])

        return {
            "image": data['image'],
            "instruction_input": instruction,
            "answer": answer,
        }


class OKVQADataset(VQADataset, __DisplMixin):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)

        self.instruction_pool =[
            "[vqa] {}",
            "[vqa] Based on the image, respond to this question with a short answer: {}"
        ]

        exist_annotation = []
        for ann in self.annotation:
            image_path = os.path.join(self.vis_root, ann["image"].split('/')[-1])
            if os.path.exists(image_path):
                exist_annotation.append(ann)
        self.annotation = exist_annotation

    def get_data(self, index):
        ann = self.annotation[index]

        image_path = os.path.join(self.vis_root, ann["image"].split('/')[-1])
        image = Image.open(image_path).convert("RGB")

        image = self.vis_processor(image)
        question = self.text_processor(ann["question"])

        answer_key = "direct_answers"

        answer_weight = {}
        for answer in ann[answer_key]:
            if answer in answer_weight.keys():
                answer_weight[answer] += 1 / len(ann[answer_key])
            else:
                answer_weight[answer] = 1 / len(ann[answer_key])

        answers = list(answer_weight.keys())
        weights = list(answer_weight.values())

        answer = random.choices(answers, weights=weights, k=1)[0]  # random sample an answer according to weights

        return {
            "image": image,
            "question": question,
            "answer": answer,
        }

    def __getitem__(self, index):
        data = self.get_data(index)
        question = self.text_processor(data["question"])
        instruction = random.choice(self.instruction_pool).format(question)

        instruction = "<Img><ImageHere></Img> {} ".format(instruction)
        answer = self.text_processor(data['answer'])

        return {
            "image": data['image'],
            "instruction_input": instruction,
            "answer": answer,
        }



class AOKVQAPOPEDataset(VQADataset, __DisplMixin, __PromptPosMixin):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths, pos='middle', info='obj+position'):
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)
        
        print("pos: {}, info: {}".format(pos, info))
        self.instruction_pool =[
            "[vqa] {}",
            "[vqa] Based on the image, respond to this question with a short answer: {}"
        ]

        exist_annotation = []
        for ann in self.annotation:
            image_path = os.path.join(self.vis_root, ann["image"].split('/')[-1]) #得到图像路径
            if os.path.exists(image_path):
                exist_annotation.append(ann)
        self.annotation = exist_annotation
        #self.data = self.get_data(self.annotation)

        self.set_info(info)
        self.set_position(pos)

    def get_data(self, index):
        ann = self.annotation[index]

        image_path = os.path.join(self.vis_root, ann["image"].split('/')[-1])
        image = Image.open(image_path).convert("RGB") #获得图像

        image = self.vis_processor(image)
        question = self.text_processor(ann["question"])
        question_id = ann["question_id"]

        obtext=[]
        if not ann["objects"]:
            objects=''
        else:
            for ob in ann["objects"]:
                #bbox = [int(x) for x in ob['bbox']]
                bbox = [int(x) for x in ob['resized_bbox']]
                bbox = "{{<{}><{}><{}><{}>}}".format(*bbox)
                if self.info == 'obj':
                    info = str(ob['category'])
                elif self.info == 'obj+position':
                    info = str(ob['category'] + bbox)
                else:
                    raise NotImplementedError 
                obtext.append(info)
                objects=';'.join(obtext)

        answer_weight = {}
        for answer in ann["answer"]:
            if answer in answer_weight.keys():
                answer_weight[answer] += 1 / len(ann["answer"])
            else:
                answer_weight[answer] = 1 / len(ann["answer"])

        answers = list(answer_weight.keys())
        weights = list(answer_weight.values())

        answer = random.choices(answers, weights=weights, k=1)[0]  # random sample an answer according to weights

        return {
            "image": image,
            "question": question,
            "question_id": question_id,
            "answer": answer,
            'objects': objects
        }

    def __getitem__(self, index):
        data = self.get_data(index)
        instruction = random.choice(self.instruction_pool).format(data['question'])
        #instruction = "<Img><ImageHere></Img><ImageHere>{} ".format(instruction)

        # second <ImageHere> for inserting softprompt.
        if self.pos == 'middle':
            instruction = "<Img><ImageHere></Img><ImageHere>Objects:<ObjHere>{} ".format(instruction)
        elif self.pos == 'late':
            instruction = "<Img><ImageHere></Img>Objects:<ObjHere><ImageHere>{} ".format(instruction)
        else:
            instruction = "<Img><ImageHere></Img><ImageHere>{} ".format(instruction)

        return {
            "image": data['image'],
            "question_id": data["question_id"],
            "instruction_input": instruction,
            "answer": self.text_processor(data['answer']),
            'object': data['objects']
        }

class AOKVQAPHDDataset(VQADataset, __DisplMixin, __PromptPosMixin):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths, pos='middle', info='obj+position'):
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)

        print("pos: {}, info: {}".format(pos, info))
        self.instruction_pool =[
            "[vqa] {}",
            "Based on the image, respond to this question with a short answer: [vqa] {}",
            "Based on the image, if the question is answerable, respond to this question with 'yes' or 'no' : [vqa] {}"
        ]

        exist_annotation = []
        for ann in self.annotation:
            image_path = os.path.join(self.vis_root, ann["image"].split('/')[-1]) #得到图像路径
            if os.path.exists(image_path):
                exist_annotation.append(ann)
        self.annotation = exist_annotation
        #self.data = self.get_data(self.annotation)
        self.set_info(info)
        self.set_position(pos)

    def get_data(self, index):
        ann = self.annotation[index]

        image_path = os.path.join(self.vis_root, ann["image"].split('/')[-1])
        image = Image.open(image_path).convert("RGB") #获得图像

        image = self.vis_processor(image)
        question = self.text_processor(ann["question"])
        question_id = ann["question_id"]

        obtext=[]
        if not ann["objects"]:
            objects=''
        else:
            for ob in ann["objects"]:
                #bbox = [int(x) for x in ob['bbox']]
                bbox = [int(x) for x in ob['resized_bbox']]
                bbox = "{{<{}><{}><{}><{}>}}".format(*bbox)
                if self.info == 'obj':
                    info = str(ob['category'])
                elif self.info == 'obj+position':
                    info = str(ob['category'] + bbox)
                else:
                    raise NotImplementedError 
                obtext.append(info)
                objects=';'.join(obtext)

        answer_weight = {}
        answers_list=[]
        answers_list.append(ann["answer"])
        for answer in answers_list:
            answer_weight[answer] = 1 
        # for answer in answers_list:
        #     if answer in answer_weight.keys():
        #         answer_weight[answer] += 1 / len(ann["answer"])
        #     else:
        #         answer_weight[answer] = 1 / len(ann["answer"])

        answers = list(answer_weight.keys())
        weights = list(answer_weight.values())

        answer = random.choices(answers, weights=weights, k=1)[0]  # random sample an answer according to weights

        return {
            "image": image,
            "question": question,
            "question_id": question_id,
            "answer": answer,
            'objects': objects
        }

    def __getitem__(self, index):
        data = self.get_data(index)
        instruction = random.choice(self.instruction_pool).format(data['question'])
        #instruction = "<Img><ImageHere></Img><ImageHere>{} ".format(instruction)
        instruction = "<Img><ImageHere></Img><ImageHere>Objects:<ObjHere>{} ".format(instruction)
        #instruction = "<Img><ImageHere></Img><ImageHere><Objects>:<ObjHere>{} ".format(instruction)


        return {
            "image": data['image'],
            "question_id": data["question_id"],
            "instruction_input": instruction,
            "answer": self.text_processor(data['answer']),
            'object': data['objects']
        }


class AOKVQAVQADataset(VQADataset, __DisplMixin, __PromptPosMixin):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths, pos='middle', info='obj+position'):
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)
        
        print("pos: {}, info: {}".format(pos, info))
        self.instruction_pool =[
            "[vqa] {}",
            "[vqa] Based on the image, respond to this question with a short answer: {}"
        ]

        exist_annotation = []
        for ann in self.annotation:
            image_path = os.path.join(self.vis_root, ann["image"].split('/')[-1]) #得到图像路径
            if os.path.exists(image_path):
                exist_annotation.append(ann)
        self.annotation = exist_annotation
        #self.data = self.get_data(self.annotation)

        self.set_info(info)
        self.set_position(pos)

    def get_data(self, index):
        ann = self.annotation[index]

        image_path = os.path.join(self.vis_root, ann["image"].split('/')[-1])
        image = Image.open(image_path).convert("RGB") #获得图像

        image = self.vis_processor(image)
        question = self.text_processor(ann["question"])
        question_id = ann["question_id"]

        obtext=[]
        if not ann["objects"]:
            objects=''
        else:
            for ob in ann["objects"]:
                #bbox = [int(x) for x in ob['bbox']]
                bbox = [int(x) for x in ob['resized_bbox']]
                bbox = "{{<{}><{}><{}><{}>}}".format(*bbox)
                if self.info == 'obj':
                    info = str(ob['category'])
                elif self.info == 'obj+position':
                    info = str(ob['category'] + bbox)
                else:
                    raise NotImplementedError 
                obtext.append(info)
                objects=';'.join(obtext)

        answer_weight = {}
        for answer in ann["answer"]:
            if answer in answer_weight.keys():
                answer_weight[answer] += 1 / len(ann["answer"])
            else:
                answer_weight[answer] = 1 / len(ann["answer"])

        answers = list(answer_weight.keys())
        weights = list(answer_weight.values())

        answer = random.choices(answers, weights=weights, k=1)[0]  # random sample an answer according to weights

        return {
            "image": image,
            "question": question,
            "question_id": question_id,
            "answer": answer,
            'objects': objects
        }

    def __getitem__(self, index):
        data = self.get_data(index)
        instruction = random.choice(self.instruction_pool).format(data['question'])
        #instruction = "<Img><ImageHere></Img><ImageHere>{} ".format(instruction)

        # second <ImageHere> for inserting softprompt.
        if self.pos == 'middle':
            instruction = "<Img><ImageHere></Img><ImageHere>Objects:<ObjHere>{} ".format(instruction)
        elif self.pos == 'late':
            instruction = "<Img><ImageHere></Img>Objects:<ObjHere><ImageHere>{} ".format(instruction)
        else:
            instruction = "<Img><ImageHere></Img><ImageHere>{} ".format(instruction)

        return {
            "image": data['image'],
            "question_id": data["question_id"],
            "instruction_input": instruction,
            "answer": self.text_processor(data['answer']),
            'object': data['objects']
        }


class AOKVQGDataset(AOKVQADataset):

    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)
        self.instruction_pool = [
            'Given the image, generate a question whose answer is: {}',
            'Based on the image, provide a question with the answer: {}',
            'Given the visual representation, create a question for which the answer is "{}"',
            'From the image provided, craft a question that leads to the reply: {}',
            'Considering the picture, come up with a question where the answer is: {}',
            'Taking the image into account, generate an question that has the answer: {}'
        ]

    def __getitem__(self, index):
        data = self.get_data(index)
        instruction = random.choice(self.instruction_pool).format(data['answer'])

        return {
            "image": data['image'],
            "instruction_input": instruction,
            "answer": data['question'],
        }
