import os
import sys
from collections import namedtuple
import tensorflow as tf
from nltk.tokenize import sent_tokenize

from .mrqa.predictor_kaggle import mrqa_predictor
from .biobert.predictor_biobert import biobert_predictor


class QaModule():
    def __init__(self, model_name, model_path, spiece_model, bert_config, bert_vocab):
        # init QA models
        self.model_name = model_name
        self.model_path = model_path
        self.spiece_model = spiece_model
        self.bert_config = bert_config
        self.bert_vocab = bert_vocab
        self.getPredictors()

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
        modelpath = self.getModelPath(model_name)
        if model_name == 'mrqa':
            d = {
                "uncased": False,
                "start_n_top": 5,
                "end_n_top": 5,
                "use_tpu": False,
                "train_batch_size": 1,
                "predict_batch_size": 1,
                "shuffle_buffer": 2048,
                "spiece_model_file": self.spiece_model,
                "max_seq_length": 512,
                "doc_stride": 128,
                "max_query_length": 64,
                "n_best_size": 5,
                "max_answer_length": 64,
            }
            self.mrqaFLAGS = namedtuple("FLAGS", d.keys())(*d.values())
            return tf.contrib.predictor.from_saved_model(modelpath)
        elif model_name == 'biobert':
            d = {
                "version_2_with_negative": False,
                "null_score_diff_threshold": 0.0,
                "verbose_logging": False,
                "init_checkpoint": None,
                "do_lower_case": False,
                "bert_config_file": self.bert_config,
                "vocab_file": self.bert_vocab,
                "train_batch_size": 1,
                "predict_batch_size": 1,
                "max_seq_length": 384,
                "doc_stride": 128,
                "max_query_length": 64,
                "n_best_size": 5,
                "max_answer_length": 30,
            }
            self.bioFLAGS = namedtuple("FLAGS", d.keys())(*d.values())
            return tf.contrib.predictor.from_saved_model(modelpath)
        else:
            raise ValueError("invalid model name")
    
    def getModelPath(self, model_name):
        index = self.model_name.index(model_name)
        return self.model_path[index]

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
                if len(answers) > 0:
                    scores = answers[-1]["data"]["confidence"]
                    answers[-1]["data"]["confidence"] = self._compute_softmax(scores)

                answer_sample = {}
                answer_sample["question"] = question
                answer_sample["data"] = {
                    "answer": [],
                    "context": [],
                    "title": [],
                    "doi": [],
                    "confidence": [],
                    "raw": [],
                }
                answers.append(answer_sample)

            context = qa["context"]
            doi = qa["qas"][0]["doi"]
            title = qa["qas"][0]["title"] 

            answers[-1]["data"]["context"].append(context)
            answers[-1]["data"]["doi"].append(doi)
            answers[-1]["data"]["title"].append(title)

            sents = sent_tokenize(context)
            spans = self.convert_idx(context, sents)
            
            raw_score_mrqa = 0
            raw_score_bio = 0

            if "mrqa" in self.model_name:
                raw_mrqa = self.mrqaPredictor([qa])
                # get sentence from MRQA
                raw = raw_mrqa[qa["qas"][0]["id"]]   
                raw_answer_mrqa = raw[0]
                raw_score_mrqa = raw[1]

                # question answering one by one
                answer_start = context.find(raw_answer_mrqa, 0)
                answer_end = answer_start + len(raw_answer_mrqa)
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
                assert raw_answer_mrqa in answer_sent_mrqa
            else:
                answer_sent_mrqa = ""
            
            
            if "biobert" in self.model_name:
                raw_bio = self.biobertPredictor([qa])
                # get sentence from BioBERT
                raw = raw_bio[qa["qas"][0]["id"]]
                raw_answer_bio = raw[0]
                raw_score_bio = raw[1] 

                if raw_answer_bio == "empty" or "":
                    answer_sent_bio = ""
                    raw_score_bio = 0
                else:
                    # question answering one by one
                    answer_start = context.find(raw_answer_bio, 0)
                    answer_end = answer_start + len(raw_answer_bio)
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
                    assert raw_answer_bio in answer_sent_bio
            else:
                answer_sent_bio = ""

            if answer_sent_mrqa == answer_sent_bio or answer_sent_mrqa in answer_sent_bio:
                # print("SAME OR QA < BIO")
                answer_sent = answer_sent_bio
                if raw_score_mrqa < 0 and raw_score_bio < 0:
                    if abs(raw_score_mrqa) < abs(raw_score_bio):
                        score = abs(raw_score_mrqa) * 0.5 + raw_score_bio
                    else:
                        score = raw_score_mrqa + abs(raw_score_bio) * 0.5
                else:
                    score = raw_score_mrqa + raw_score_bio
            elif answer_sent_bio in answer_sent_mrqa:
                # print("BIO < QA")
                answer_sent = answer_sent_mrqa
                if raw_score_mrqa < 0 and raw_score_bio < 0:
                    if abs(raw_score_mrqa) < abs(raw_score_bio):
                        score = abs(raw_score_mrqa) * 0.5 + raw_score_bio
                    else:
                        score = raw_score_mrqa + abs(raw_score_bio) * 0.5
                else:
                    score = raw_score_mrqa + raw_score_bio
            else:
                # print("DIFFERENT ANSWERS")
                answer_sent= " ".join([answer_sent_mrqa, answer_sent_bio])
                score = 0.5 * raw_score_mrqa + 0.5 * raw_score_bio
            
            if raw_answer_mrqa == raw_answer_bio or raw_answer_mrqa in raw_answer_bio:
                # print("SAME OR QA < BIO")
                answer = [raw_answer_bio]
            elif raw_answer_bio in raw_answer_mrqa:
                # print("BIO < QA")
                answer = [answer_sent_mrqa]
            else:
                # print("DIFFERENT ANSWERS")
                answer = [raw_answer_mrqa, raw_answer_bio]
            
            answers[-1]["data"]["answer"].append(answer_sent)
            answers[-1]["data"]["raw"].append(answer)
            answers[-1]["data"]["confidence"].append(score)
        
        # rerank the answers
        score_qa = get_rank_score(answers)
        return score_qa
    
    def _compute_softmax(self, scores):
        """Compute softmax probability over scores."""
        if not scores:
            return []

        max_score = None
        for score in scores:
            if max_score is None or score > max_score:
                max_score = score

        exp_scores = []
        total_sum = 0.0
        for score in scores:
            x = math.exp(score - max_score)
            exp_scores.append(x)
            total_sum += x

        probs = []
        for score in exp_scores:
            probs.append(score / total_sum)
        return probs
    
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

def get_rank_score(qa_output):
    for item in qa_output:
        query = item["question"]
        context = item['data']['context']
        item['data']['matching_score'] = []
        item['data']['rerank_score'] = []
        # make new query with only n. and adj.
        tokens = word_tokenize(query.lower())
        tokens = [word for word in tokens if word not in stop_words]
        tagged = pos_tag(tokens)
        query_token = [tag[0] for tag in tagged if 'NN' in tag[1] or 'JJ' in tag[1] or 'VB' in tag[1]]

        for i in range(len(context)):
            text = context[i].lower()
            count = 0
            text_words = word_tokenize(text)
            for word in text_words:
                if word in query_token:
                    count += 1
            
            # matching_score = count/len(text_words)*10 if len(text_words)>50 else count/len(text_words)   # short sentence penalty
            matching_score = count / (1 + math.exp(-len(text_words)+50)) / 5
            item['data']['matching_score'].append(matching_score)
            item['data']['rerank_score'].append(matching_score + item['data']['confidence'][i])
        
        # sort QA results
        c = list(zip(item['data']['rerank_score'], item['data']['context'], item['data']['answer'], item['data']['confidence'], item['data']['doi'], item['data']['title'], item['data']['matching_score'], item['data']['raw']))
        c.sort(reverse = True)
        item['data']['rerank_score'], item['data']['context'], item['data']['answer'], item['data']['confidence'], item['data']['doi'], item['data']['title'], item['data']['matching_score'], item['data']['raw'] = zip(*c)
    return qa_output


    
