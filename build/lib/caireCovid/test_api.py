from flask import Flask, request, jsonify
import json
from retrieval import retrieve_paragraph
from qa import QaModule

def get_qa_result(query):
	temp_json = retrieve_paragraph(query)
	qa_item = {'question': query}
	contexts = []
	titles = []
	doi = []
	count = 1
	for item in temp_json:
		if count>10:
			break
		if 'abstract' in item and len(item['abstract']) > 0:
			contexts.append(item['abstract'])
		if 'paragraphs' in item:
			contexts.append(item['paragraphs'][0]['text'])
		doi.append(item["doi"])
		titles.append(item["title"])
		count+=1
	#print(len(doi), len(titles))
	qa_item['data'] = {'answer': '', 'context':contexts, 'doi': doi, 'titles': titles}
	data_for_qa = [qa_item]
	qa_model = QaModule(['mrqa', 'biobert'])
	answers = qa_model.getAnswers(data_for_qa)
	output_list = []
	for i in range(len(answers[0]['data']['answer'])):
		outJson = {}
		outJson['question'] = answers[0]['question']
		outJson['answer'] = answers[0]['data']['answer'][i]
		outJson['context'] = answers[0]['data']['context'][i]
		outJson['doi'] = doi[i]
		outJson['title'] = titles[i]
		output_list.append(outJson)
	#print(len(output_list))
	return output_list

#print(json.dumps(get_qa_result('incubation period of covid-19 in humans'), indent=4))

app = Flask(__name__)

@app.route('/query_qa', methods=['POST'])
def return_matches():
	content = request.json
	out = get_qa_result(content['text'])
	return jsonify(out)

if __name__ == '__main__':
	app.run(host= '0.0.0.0',debug=True)
