from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration
from clipscore_helper import get_all_metrics
import contextlib
import pandas as pd
import torch
import json
from PIL import Image
from evaluate import load
import requests

bertscore = load("bertscore")

import numpy as np

model = InstructBlipForConditionalGeneration.from_pretrained("Salesforce/instructblip-vicuna-7b")
processor = InstructBlipProcessor.from_pretrained("Salesforce/instructblip-vicuna-7b")

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

f = open('../dataset_annotations.json')
data = json.load(f)
results = []

refs = []
hyps = []

for eval_datapoint in data:
    print("Datapoint ", eval_datapoint)
    image_path = eval_datapoint['image']

    question = eval_datapoint['question']

    question = 'Assume someone is browsing a ' + eval_datapoint['context'] + ' website when they encounter this image. They cannot see the image directly, but they can access this image description: ' + eval_datapoint['description'] + ' Based on this description, they asked this follow-up question. Please answer based on the image. In your answer, prioritize details relevant to this person. Question: ' + eval_datapoint['question']
    image = Image.open('../' + image_path).convert("RGB")
    inputs = processor(images=image, text=question, return_tensors="pt").to(device)

    outputs = model.generate(
        **inputs,
        do_sample=False,
        max_length=84,
        min_length=1,
    )

    generated_answer = processor.batch_decode(outputs, skip_special_tokens=True)[0].strip()
    print("Question: ", question)
    print("Answer: ", generated_answer)

    results.append({
            'image': eval_datapoint['image'],
            'description': eval_datapoint['description'],
            'context': eval_datapoint['context'],
            'question': eval_datapoint['question'],
            'generated_answer': generated_answer,
            'answers': eval_datapoint['answers']
        })
    if (len(eval_datapoint['answers']) == 3):
        refs.append([eval_datapoint['answers'][0].lower().strip(), eval_datapoint['answers'][1].lower().strip(), eval_datapoint['answers'][2].lower().strip()])
    else:
        refs.append([eval_datapoint['answers'][0].lower().strip(), eval_datapoint['answers'][1].lower().strip()])
    hyps.append(generated_answer.lower())

results_per_context = {}
refs_per_context = {}
hyps_per_context = {}

for datapoint in results:
    if (datapoint['context'] not in refs_per_context):
        refs_per_context[datapoint['context']] = []
        hyps_per_context[datapoint['context']] = []

    hyps_per_context[datapoint['context']].append(datapoint['generated_answer'].lower().strip())
    eval_datapoint = datapoint

    if (len(datapoint['answers']) == 3):
        refs_per_context[datapoint['context']].append([eval_datapoint['answers'][0].lower().strip(), eval_datapoint['answers'][1].lower().strip(), eval_datapoint['answers'][2].lower().strip()])
    else:
        refs_per_context[datapoint['context']].append([eval_datapoint['answers'][0].lower().strip(), eval_datapoint['answers'][1].lower().strip()])

metrics_per_context = {}

for context in refs_per_context:
    with contextlib.redirect_stdout(None):
        metrics_dict = get_all_metrics(refs_per_context[context], hyps_per_context[context])
    metrics_per_context[context] = metrics_dict

# Dump each metric per context here!
with open('instructblip_context_description_percontext_metrics.json', 'w') as fp:
    json.dump(metrics_per_context, fp)

with contextlib.redirect_stdout(None):
    metrics_dict = get_all_metrics(refs, hyps)

bertscore_results = bertscore.compute(predictions=hyps, references=refs, lang="en")

metrics_dict['bertscore_avg_precision'] = np.mean(bertscore_results['precision'])
metrics_dict['bertscore_avg_recall'] = np.mean(bertscore_results['recall'])
metrics_dict['bertscore_avg_f1'] = np.mean(bertscore_results['f1'])

print("Metrics dict for Instruct BLIP evaluation: ", metrics_dict)

metrics_dict['full_bertscore'] = bertscore_results
idx = list(range(0, len(results)))
df = pd.DataFrame(results, index=idx)
df.to_csv('instructblip_context_description_results.csv')

with open('instructblip_context_description_metrics_dict.json', 'w') as fp:
        json.dump(metrics_dict, fp)
