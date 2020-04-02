import os
import sys
from collections import namedtuple
import tensorflow as tf
from nltk.tokenize import sent_tokenize

from mrqa.predictor_kaggle import mrqa_predictor
from biobert.predictor_biobert import biobert_predictor


class QaModule():
    def __init__(self, model_name):
        # init QA models
        self.model_name = model_name
        self.predict_fn = self.getPredictors()

    def readIR(self, data):
        synthetic = []

        idx = 0
        for data_item in data:
            question = data_item["question"]
            answer = data_item["data"]["answer"]
            contexts = data_item["data"]["context"]
            dois = data_item["data"]["doi"]
            titles = data_item["data"]["titles"]
            
            for (context, doi, title) in zip(contexts, dois, titles):
                data_sample = {
                    "context": context,
                    "qas": []
                }

                qas_item = {
                    "id": idx,
                    "question": question,
                    "answer": answer,
                    "doi": doi,
                    "title": title,
                }
                data_sample["qas"].append(qas_item)
                synthetic.append(data_sample)

                idx += 1
        return synthetic

    def mrqaPredictor(self, data):
        return mrqa_predictor(self.mrqaFLAGS, self.mrqa_predict_fn, data)
    
    def biobertPredictor(self, data):
        return biobert_predictor(self.bioFLAGS, self.bio_predict_fn, data)

    def getPredictors(self):
        if "mrqa" in self.model_name:
            self.mrqa_predict_fn = self.getPredictor("mrqa")
        if "biobert" in self.model_name:
            self.bio_predict_fn = self.getPredictor("biobert")

    def getPredictor(self, model_name):
        if model_name == 'mrqa':
            d = {
                "uncased": False,
                "start_n_top": 5,
                "end_n_top": 5,
                "use_tpu": False,
                "train_batch_size": 1,
                "predict_batch_size": 1,
                "shuffle_buffer": 2048,
                "spiece_model_file": "./mrqa/model/spiece.model",
                "max_seq_length": 512,
                "doc_stride": 128,
                "max_query_length": 64,
                "n_best_size": 5,
                "max_answer_length": 64,
            }
            self.mrqaFLAGS = namedtuple("FLAGS", d.keys())(*d.values())
            return tf.contrib.predictor.from_saved_model("/kaggle/input/pretrained-qa-models/mrqa/1564469515")
        elif model_name == 'biobert':
            d = {
                "version_2_with_negative": False,
                "null_score_diff_threshold": 0.0,
                "verbose_logging": False,
                "init_checkpoint": None,
                "do_lower_case": False,
                "bert_config_file": "./biobert/model/bert_config.json",
                "vocab_file": "./biobert/model/vocab.txt",
                "train_batch_size": 1,
                "predict_batch_size": 1,
                "max_seq_length": 384,
                "doc_stride": 128,
                "max_query_length": 64,
                "n_best_size": 5,
                "max_answer_length": 30,
            }
            self.bioFLAGS = namedtuple("FLAGS", d.keys())(*d.values())
            return tf.contrib.predictor.from_saved_model("/kaggle/input/pretrained-qa-models/biobert/1585470591")
        else:
            raise ValueError("invalid model name")

    def getAnswers(self, data):
        """
        Output:
            List [{
                "question": "xxxx",
                "data": 
                    {
                        "answer": ["answer1", "answer2", ...],
                        "confidence": [1,2, ...],
                        "context": ["paragraph1", "paragraph2", ...],
                    }
            }]
        """
        answers = []
        qas = self.readIR(data)
        for qa in qas:
            question = qa["qas"][0]["question"]
            if len(answers)==0 or answers[-1]["question"]!=question:
                answer_sample = {}
                answer_sample["question"] = question
                answer_sample["data"] = {
                    "answer": [],
                    "context": [],
                    "title": [],
                    "doi": [],
                }
                answers.append(answer_sample)

            context = qa["context"]
            doi = qa["qas"][0]["doi"]
            title = qa["qas"][0]["title"] 

            answers[-1]["data"]["context"].append(context)

            sents = sent_tokenize(context)
            spans = self.convert_idx(context, sents)

            if "mrqa" in self.model_name:
                raw_mrqa = self.mrqaPredictor([qa])
                # get sentence from MRQA
                raw = raw_mrqa[qa["qas"][0]["id"]]            
                # question answering one by one
                answer_start = context.find(raw, 0)
                answer_end = answer_start + len(raw)
                answer_span = []
                for idx, span in enumerate(spans):
                    if not (answer_end <= span[0] or answer_start >= span[1]):
                        answer_span.append(idx)

                y1, y2 = answer_span[0], answer_span[-1]
                if not y1 == y2:
                    # context tokens in index y1 and y2 should be merged together
                    # print("Merge knowledge sentence")
                    answer_sent_mrqa = " ".join(sents[y1:y2+1])
                else:
                    answer_sent_mrqa = sents[y1]
                assert raw in answer_sent_mrqa
            else:
                answer_sent_mrqa = ""
            
            
            if "biobert" in self.model_name:
                raw_bio = self.biobertPredictor([qa])
                # get sentence from BioBERT
                raw = raw_bio[qa["qas"][0]["id"]]            
                # question answering one by one
                answer_start = context.find(raw, 0)
                answer_end = answer_start + len(raw)
                answer_span = []
                for idx, span in enumerate(spans):
                    if not (answer_end <= span[0] or answer_start >= span[1]):
                        answer_span.append(idx)

                y1, y2 = answer_span[0], answer_span[-1]
                if not y1 == y2:
                    # context tokens in index y1 and y2 should be merged together
                    # print("Merge knowledge sentence")
                    answer_sent_bio = " ".join(sents[y1:y2+1])
                else:
                    answer_sent_bio = sents[y1]
                
                # if raw not in answer_sent_bio:
                #     print("RAW", raw)
                #     print("BIO", answer_sent_bio)
                # assert raw in answer_sent_bio
            else:
                answer_sent_bio = ""

            if answer_sent_mrqa == answer_sent_bio or answer_sent_mrqa in answer_sent_bio:
                # print("SAME OR QA < BIO")
                answer_sent = answer_sent_bio
            elif answer_sent_bio in answer_sent_mrqa:
                # print("BIO < QA")
                answer_sent = answer_sent_mrqa
            else:
                # print("DIFFERENT ANSWERS")
                answer_sent= " ".join([answer_sent_mrqa, answer_sent_bio])
            
            answers[-1]["data"]["answer"].append(answer_sent)


            # print("context:", context)
            # print("-"*80)
            # print("query:", question)
            # print("-"*80)
            # print("answer:", answer_sent)
            # input()
            # break
        return answers
    
    def convert_idx(self, text, tokens):
        current = 0
        spans = []
        for token in tokens:
            current = text.find(token, current)
            if current < 0:
                print("Token {} cannot be found".format(token))
                raise Exception()
            spans.append((current, current + len(token)))
            current += len(token)
        return spans

def print_answers_in_file(answers, filepath="./answers.txt"):
    """
        Input:
            List [{
                "question": "xxxx",
                "data": 
                    {
                        "answer": ["answer1", "answer2", ...],
                        "confidence": [1,2, ...],
                        "context": ["paragraph1", "paragraph2", ...],
                    }
            }]
        """
    with open(filepath, "w") as f:
        print("WRITE ANSWERS IN FILES ...")
        for item in answers:
            question = item["question"]
            cas = item["data"]
            for (answer, context) in zip(cas["answer"], cas["context"]):
                f.write("-"*80+"\n")
                f.write("context: "+context+"\n")
                f.write("-"*80+"\n")
                f.write("question: "+question+"\n")
                f.write("-"*80+"\n")
                f.write("answer: "+answer+"\n")
            f.write("="*80+"\n")




    
