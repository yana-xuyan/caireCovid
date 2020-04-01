import requests
import json
import os
def get_ir_result(query):
    post = {}
    post['text'] = query
    r = requests.post('http://hlt027.ece.ust.hk:5000/query_paragraph', json=post)
    paragraphs = []
    for i in range(len(r.json())):
        if 'body_text' in r.json()[i].keys():
            oen_article = ""
            for j in range(len(r.json()[i]['body_text'])):
                oen_article += r.json()[i]['body_text'][j]['text']
            paragraphs.append(oen_article)
    return paragraphs

def main(paragraphs):
     # write data with json
    with open('./data/data.json', 'w') as f:
        for paragraph in paragraphs:
            one_line = {}
            one_line["src"] = paragraph
            one_line["tgt"] = ""
            json.dump(one_line, f)
            f.write('\n')
    
    os.system("sh ./unilm/test.sh")
    
def get_result():
    with open('./unilm/checkpoint/ckpt-30000.validation', 'w') as f:
        data = f.readlines()
        result = ""
        for item in data:
            result += item.replace('\n', ' ')
        print(result)
        return result
    
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='unilm summary')
    parser.add_argument('-q','--query', help='input query', required=False, default='What is the range of incubation periods for COVID-19 in humans?')
    args = vars(parser.parse_args())
    
    query = args['query']
    paragraphs = get_ir_result(query)
    main(paragraphs)
    get_result()