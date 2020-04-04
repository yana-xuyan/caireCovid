import os
import sys

import json
from retrieval import information_retrieval
from caireCovid import QaModule, print_answers_in_file

all_results, data_for_qa = information_retrieval("question_generation/task1_question.json")

qa_model = QaModule(["mrqa", "biobert"], ["/home/xuyan/mrqa/xlnet-qa/experiment/multiqa-1e-5-tpu/1564469515", "/home/xuyan/kaggle/bioasq-biobert/model/1585470591"], \
    "./mrqa/model/spiece.model", "./biobert/model/bert_config.json", "./biobert/model/vocab.txt"
)

answers = qa_model.getAnswers(data_for_qa)

# print_answers_in_file(answers)
format_answer = {}
for item in answers:
    format_answer[item["question"]] = item["data"]

with open("data.json", "w") as f:
    json.dump(format_answer, f)

# Final output for synthesis
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
