from clipscore_helper import get_all_metrics, get_clip_score, get_refonlyclipscore
import contextlib
import pandas as pd
import json
import clip

from PIL import Image
import requests
from transformers import Blip2Processor, Blip2ForConditionalGeneration
import torch
from evaluate import load
import requests

bertscore = load("bertscore")
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"

processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
model = Blip2ForConditionalGeneration.from_pretrained(
    "Salesforce/blip2-opt-2.7b", load_in_8bit=True, device_map={"": 0}, torch_dtype=torch.float16
)

f = open('../dataset_annotations_final.json')
data = json.load(f)

results = []

refs = []
hyps = []

images_for_clipscore = []

for eval_datapoint in data:
    print("Datapoint ", eval_datapoint)
    image_path = eval_datapoint['image']

    question = eval_datapoint['question']
#    question = 'Assume someone is browsing a ' + eval_datapoint['context'] + ' website when they encounter this image. They cannot see the image directly, but they can access this image description: ' + eval_datapoint['description'] + ' Based on this description, they asked this follow-up question. Please answer based on the image. In your answer, prioritize details relevant to this person. Question: ' + eval_datapoint['question']
    image = Image.open('../' + image_path).convert("RGB")

    images_for_clipscore.append('../' + image_path)
    
    inputs = processor(images=image, text=question, return_tensors="pt").to(device="cuda", dtype=torch.float16)

    generated_ids = model.generate(**inputs,
            do_sample=False,
            max_length=256)

    generated_answer = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

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
images_per_context = {}

for datapoint in results:
    if (datapoint['context'] not in refs_per_context):
        refs_per_context[datapoint['context']] = []
        hyps_per_context[datapoint['context']] = []
        images_per_context[datapoint['context']] = []

    hyps_per_context[datapoint['context']].append(datapoint['generated_answer'].lower().strip())
    eval_datapoint = datapoint

    images_per_context[datapoint['context']].append('../' + datapoint['image'])

    if (len(datapoint['answers']) == 3):
        refs_per_context[datapoint['context']].append([eval_datapoint['answers'][0].lower().strip(), eval_datapoint['answers'][1].lower().strip(), eval_datapoint['answers'][2].lower().strip()])
    else:
        refs_per_context[datapoint['context']].append([eval_datapoint['answers'][0].lower().strip(), eval_datapoint['answers'][1].lower().strip()])

metrics_per_context = {}

device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, transform = clip.load("ViT-B/32", device=device, jit=False)
clip_model.eval()
for context in refs_per_context:
    with contextlib.redirect_stdout(None):
        metrics_per_context[context] = get_all_metrics(refs_per_context[context], hyps_per_context[context])

        _, per_instance_image_text, candidate_feats = get_clip_score(clip_model, images_per_context[context], hyps_per_context[context], device)

        _, per_instance_text_text = get_refonlyclipscore(
            clip_model, refs_per_context[context], candidate_feats, device)
# F-score
        refclipscores = 2 * per_instance_image_text * per_instance_text_text / (per_instance_image_text + per_instance_text_text)

        scores = {image_id: {'CLIPScore': float(clipscore), 'RefCLIPScore': float(refclipscore)}
            for image_id, clipscore, refclipscore in
            zip(images_per_context[context], per_instance_image_text, refclipscores)}

        metrics_per_context[context]['clipscore'] = np.mean([s['CLIPScore'] for s in scores.values()])
        metrics_per_context[context]['clipscore_std'] = np.std([s['CLIPScore'] for s in scores.values()])

        metrics_per_context[context]['ref_clipscore'] = np.mean([s['RefCLIPScore'] for s in scores.values()])
        metrics_per_context[context]['ref_clipscore_std'] = np.std([s['RefCLIPScore'] for s in scores.values()])

writefile = "blip2_baseline"

with open(writefile + '_percontext_metrics.json', 'w') as fp:
    json.dump(metrics_per_context, fp)

with contextlib.redirect_stdout(None):
    metrics_dict = get_all_metrics(refs, hyps)

bertscore_results = bertscore.compute(predictions=hyps, references=refs, lang="en")

metrics_dict['bertscore_avg_precision'] = np.mean(bertscore_results['precision'])
metrics_dict['bertscore_avg_recall'] = np.mean(bertscore_results['recall'])
metrics_dict['bertscore_avg_f1'] = np.mean(bertscore_results['f1'])

metrics_dict['full_bertscore'] = bertscore_results

_, per_instance_image_text, candidate_feats = get_clip_score(clip_model, images_for_clipscore, hyps, device)

_, per_instance_text_text = get_refonlyclipscore(
            clip_model, refs, candidate_feats, device)
# F-score
refclipscores = 2 * per_instance_image_text * per_instance_text_text / (per_instance_image_text + per_instance_text_text)

scores = {image_id: {'CLIPScore': float(clipscore), 'RefCLIPScore': float(refclipscore)}
            for image_id, clipscore, refclipscore in
            zip(images_for_clipscore, per_instance_image_text, refclipscores)}

print("Metrics dict for BLIP-2 evaluation: ", metrics_dict)
clipscore = np.mean([s['CLIPScore'] for s in scores.values()])
clipscore_std = np.std([s['CLIPScore'] for s in scores.values()])
ref_clipscore = np.mean([s['RefCLIPScore'] for s in scores.values()])
ref_clipscore_std = np.std([s['RefCLIPScore'] for s in scores.values()])
print("CLIPScore: ", clipscore)
print("CLIPScore standard deviation: ", clipscore_std)

metrics_dict['clipscore'] = clipscore
metrics_dict['clipscore_std_dev'] = clipscore_std
metrics_dict['ref_clipscore'] = ref_clipscore
metrics_dict['ref_clipscore_std_dev'] = ref_clipscore_std

metrics_dict['full_clipscore'] = [s['CLIPScore'] for s in scores.values()]
idx = list(range(0, len(results)))
df = pd.DataFrame(results, index=idx)
df.to_csv(writefile + '_results.csv')

with open(writefile + '_metrics_dict.json', 'w') as fp:
    json.dump(metrics_dict, fp)

