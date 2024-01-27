import torch
from clipscore_helper import get_all_metrics, get_clip_score, get_refonlyclipscore
import clip
import contextlib
import pandas as pd
import json
import numpy as np
import os
import torch
from transformers import IdeficsForVisionText2Text, AutoProcessor
from PIL import Image
import requests
import torch
from evaluate import load
import requests
import argparse

bertscore = load("bertscore")

device = "cuda" if torch.cuda.is_available() else "cpu"

f = open('../dataset_annotations_cleaned.json')
data = json.load(f)

results = []

parser = argparse.ArgumentParser(description='Optional app description')
parser.add_argument('--writefile', type=str,
                    required=True)

parser.add_argument('--generation_length', type=int,
                    help='A required integer positional argument',
                    default=21)

parser.add_argument('--per_context_scores', action='store_true')

parser.add_argument('--context_description', action='store_true')
parser.add_argument('--description_only', action='store_true')
parser.add_argument('--context_only', action='store_true')

args = parser.parse_args()

refs = []
hyps = []

device = "cuda" if torch.cuda.is_available() else "cpu"

checkpoint = "HuggingFaceM4/idefics-9b-instruct"
model = IdeficsForVisionText2Text.from_pretrained(checkpoint, torch_dtype=torch.bfloat16).to(device)
processor = AutoProcessor.from_pretrained(checkpoint)

formatted_contexts = {'social_media': 'social media', 'science_journals': 'science magazine'}

context_goals = {
    'social media': 'learning more about their connections',
    'health': 'learning how to live a healthier lifestyle',
    'shopping': 'purchasing an item or experience',
    'travel': 'traveling to a new location',
    'science magazine': 'learning more about recent science developments',
    'news': 'learning more about recent news developments'
}

images = []

for eval_datapoint in data:
  #  print("Datapoint ", eval_datapoint)
    image_path = eval_datapoint['image']

    images.append('../' + image_path)

    question = eval_datapoint['question']

    context = eval_datapoint['context']

    if (eval_datapoint['context'] in formatted_contexts):
        context = formatted_contexts[eval_datapoint['context']]

    # Context + description condition
    if (args.context_description):
        question = 'Assume someone is browsing a ' + context + ' website when they encounter this image. They cannot see the image directly, but they can access this image description: ' + eval_datapoint['description'] + ' Based on this description, they asked this follow-up question. Please answer based on the image. In your answer, prioritize details relevant to this person. Question: ' + eval_datapoint['question']

    # Description-only condition
    if (args.description_only):
        question = 'Assume someone is browsing a website when they encounter this image. They cannot see the image directly, but they can access this image description: ' + eval_datapoint['description'] + ' Based on this description, they asked this follow-up question. Please answer based on the image. In your answer, prioritize details relevant to this person. Question: ' + eval_datapoint['question']

    if (args.context_only):
        question = 'Assume someone is browsing a ' + context + ' website when they encounter this image. They cannot see the image directly, but they can access an image description. Based on this description, they asked this follow-up question. Please answer based on the image. In your answer, prioritize details relevant to this person. Question: ' + eval_datapoint['question']

    image = Image.open('../' + image_path).convert("RGB")

    prompts = [
    [
        "User: " + question,
        image,
        "<end_of_utterance>",
        "\nAssistant: ",
    ],
    ]

    inputs = processor(prompts, add_end_of_utterance_token=False, return_tensors="pt").to(device)
    exit_condition = processor.tokenizer("<end_of_utterance>", add_special_tokens=False).input_ids
    bad_words_ids = processor.tokenizer(["<image>", "<fake_token_around_image>"], add_special_tokens=False).input_ids

    generated_ids = model.generate(**inputs, eos_token_id=exit_condition, bad_words_ids=bad_words_ids, max_new_tokens=256, do_sample=False, num_beams=1)
    generated_answer = processor.batch_decode(generated_ids, skip_special_tokens=True)

    generated_answer[0] = generated_answer[0].replace(prompts[0][0], '')
    generated_answer[0] = generated_answer[0].replace(' \nAssistant: ', '')
    generated_answer[0] = generated_answer[0].replace('Assistant:', '').strip()

    print("Question: ", question)
    print("Answer: ", generated_answer)

    results.append({
            'image': eval_datapoint['image'],
            'description': eval_datapoint['description'],
            'context': eval_datapoint['context'],
            'prompt': prompts[0][0],
            'question': eval_datapoint['question'],
            'answers': eval_datapoint['answers'],
            'generated_answer': generated_answer[0]
            })

    if (len(eval_datapoint['answers']) == 3):
        refs.append([eval_datapoint['answers'][0].lower().strip(), eval_datapoint['answers'][1].lower().strip(), eval_datapoint['answers'][2].lower().strip()])
    else:
        refs.append([eval_datapoint['answers'][0].lower().strip(), eval_datapoint['answers'][1].lower().strip()])
    hyps.append(generated_answer[0].lower().strip())

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

device = "cuda" if torch.cuda.is_available() else "cpu"
model, transform = clip.load("ViT-B/32", device=device, jit=False)
model.eval()

_, per_instance_image_text, candidate_feats = get_clip_score(model, images, hyps, device)

_, per_instance_text_text = get_refonlyclipscore(
            model, refs, candidate_feats, device)
# F-score
refclipscores = 2 * per_instance_image_text * per_instance_text_text / (per_instance_image_text + per_instance_text_text)

scores = {image_id: {'CLIPScore': float(clipscore), 'RefCLIPScore': float(refclipscore)}
            for image_id, clipscore, refclipscore in
            zip(images, per_instance_image_text, refclipscores)}

#scores = {image_id: {'CLIPScore': float(clipscore)}
 #                   for image_id, clipscore in zip(images, per_instance_image_text)}

clipscore = np.mean([s['CLIPScore'] for s in scores.values()])
clipscore_std = np.std([s['CLIPScore'] for s in scores.values()])

ref_clipscore = np.mean([s['RefCLIPScore'] for s in scores.values()])
ref_clipscore_std = np.std([s['RefCLIPScore'] for s in scores.values()])

print("CLIPScore: ", clipscore)
print("CLIPScore standard deviation: ", clipscore_std)

writeFile = args.writefile

folder_path = "results/" + writeFile + "/" + writeFile

os.mkdir("results/" + writeFile)
metrics_per_context = {}
metrics_overall_contexts = {}

if (args.per_context_scores):
    for context in refs_per_context:
        with contextlib.redirect_stdout(None):
            metrics_per_context[context] = get_all_metrics(refs_per_context[context], hyps_per_context[context])

        print("Metrics overall contexts ", metrics_overall_contexts)

        _, per_instance_image_text, candidate_feats = get_clip_score(model, images_per_context[context], hyps_per_context[context], device)

        _, per_instance_text_text = get_refonlyclipscore(
            model, refs_per_context[context], candidate_feats, device)
# F-score
        refclipscores = 2 * per_instance_image_text * per_instance_text_text / (per_instance_image_text + per_instance_text_text)

        scores = {image_id: {'CLIPScore': float(clipscore), 'RefCLIPScore': float(refclipscore)}
            for image_id, clipscore, refclipscore in
            zip(images, per_instance_image_text, refclipscores)}

        metrics_per_context[context]['clipscore'] = np.mean([s['CLIPScore'] for s in scores.values()])
        metrics_per_context[context]['clipscore_std'] = np.std([s['CLIPScore'] for s in scores.values()])

        metrics_per_context[context]['ref_clipscore'] = np.mean([s['RefCLIPScore'] for s in scores.values()])
        metrics_per_context[context]['ref_clipscore_std'] = np.std([s['RefCLIPScore'] for s in scores.values()])

        metrics_overall_contexts[context] = metrics_per_context[context]
    print("Metrics per context ", metrics_per_context)

    with open(folder_path + '_percontext_metrics.json', 'w') as fp:
        json.dump(metrics_per_context, fp)

with contextlib.redirect_stdout(None):
    metrics_dict = get_all_metrics(refs, hyps)

metrics_dict['clipscore'] = clipscore
metrics_dict['clipscore_std_dev'] = clipscore_std

metrics_dict['ref_clipscore'] = ref_clipscore
metrics_dict['ref_clipscore_std_dev'] = ref_clipscore_std

metrics_dict['clipscores'] = [s['CLIPScore'] for s in scores.values()]
metrics_dict['ref_clipscore'] = [s['RefCLIPScore'] for s in scores.values()]

metrics_per_context = {}


bertscore_results = bertscore.compute(predictions=hyps, references=refs, lang="en")

metrics_dict['bertscore_avg_precision'] = np.mean(bertscore_results['precision'])
metrics_dict['bertscore_avg_recall'] = np.mean(bertscore_results['recall'])
metrics_dict['bertscore_avg_f1'] = np.mean(bertscore_results['f1'])

metrics_dict['full_bertscore'] = bertscore_results

print("Metrics dict for IDEFICS evaluation: ", metrics_dict)

idx = list(range(0, len(results)))
df = pd.DataFrame(results, index=idx)
df.to_csv(folder_path + '_results.csv')

with open(folder_path + '_metrics_dict.json', 'w') as fp:
    json.dump(metrics_dict, fp)
