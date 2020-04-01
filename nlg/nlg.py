import json
import os

import torch
import numpy
from tqdm import tqdm
# from transformers import *
import warnings

## input format
# List [{
#         "question": "xxxx",
#         "data": 
#         {
#             "answer": ["answer1", "answer2", ...],
#             "confidence": [confidence1, confidence2, ...],
#             "title": [title1, title2, ...],
#             "doi": [doi1, doi2, ...]
#             "sha": [sha1, sha2, ...]
#         }
# }]



## output format:
# List [{
#         "question": "xxxx",
#         "text": "xxx"      
# }]


class Nlg_example(object):

    def __init__(self,
                question,
                answer,
                confidence,
                title,
                doi,
                sha
                ):
        self.question = question
        self.answer = answer
        self.confidence = confidence
        self.title = title
        self.doi = doi
        self.sha = sha 

# call ths function to generate the output 
def nlg(data_list_in):
    data_in = data_list_in
    
    output_list = []

    for each in data_in:
        nlg_list = []
        input = each
        question = input['question']
        data = input['data']
        ans_list = data['answer']
        confidence_list = data['confidence']
        title_list = data['title']
        doi_list = data['doi']
        sha_list = data['sha']

        
        for i in range(len(ans_list)):
            answer = ans_list[i]
            confidence = confidence_list[i]
            title = title_list[i]
            doi = doi_list[i]
            sha = sha_list[i]

            example = Nlg_example(
                question = question,
                answer = answer,
                confidence = confidence,
                title = title,
                doi = doi,
                sha = sha
            )
            nlg_list.append(example)
        
        nlg_out = model(nlg_list)
        print(nlg_out)

        output_list.append(nlg_out)
    
    return output_list

def model(nlg_example_list):
    ##### 
    output_text = "hello world. I hate corona virus. Hope everything goes well. We love peace"
    question = nlg_example_list[0].question

    output = {}
    output['question'] = question
    output['text'] = output_text
    
    return output


if __name__ == "__main__":

    input_list = []
    input_json = {}
    input_json['question'] = "what is the risk factors of covid-19?"
    input_json['data'] = {}
    input_json['data']['answer'] = ["hello world"]
    input_json['data']['confidence'] = ['0.9']
    input_json['data']['title'] = ['xxxx']
    input_json['data']['doi'] = ['1234']
    input_json['data']['sha'] = ['1234']

    for i in range(10):
        print(input_json)
        input_list.append(input_json)

    out_list = nlg(input_list)
    







