from transformers import AutoTokenizer, BitsAndBytesConfig
import sys
from clipscore_helper import get_all_metrics, get_clip_score, get_refonlyclipscore
import random
import argparse

import clip
from llava.model import LlavaLlamaForCausalLM
import torch
import os
import json
import pandas as pd 
import contextlib
import numpy as np

from tqdm import tqdm

from evaluate import load
import requests

bertscore = load("bertscore")
import os
import requests
from PIL import Image
from io import BytesIO
from llava.conversation import conv_templates, SeparatorStyle
from llava.utils import disable_torch_init
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria
from transformers import TextStreamer
model_path = "4bit/llava-v1.5-13b-3GB"
kwargs = {"device_map": "auto"}
kwargs['load_in_4bit'] = True
kwargs['quantization_config'] = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type='nf4'
)
model = LlavaLlamaForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)
tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)

vision_tower = model.get_vision_tower()
if not vision_tower.is_loaded:
    vision_tower.load_model()
vision_tower.to(device='cuda')
image_processor = vision_tower.image_processor

def caption_image(image_file, prompt):
    if image_file.startswith('http') or image_file.startswith('https'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    disable_torch_init()
    conv_mode = "llava_v0"
    conv = conv_templates[conv_mode].copy()
    roles = conv.roles
    image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'].half().cuda()
    inp = f"{roles[0]}: {prompt}"
    inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + inp
    conv.append_message(conv.roles[0], inp)
    conv.append_message(conv.roles[1], None)
    raw_prompt = conv.get_prompt()
    input_ids = tokenizer_image_token(raw_prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
    with torch.inference_mode():
      output_ids = model.generate(input_ids, images=image_tensor, do_sample=False, max_new_tokens=256, use_cache=True, stopping_criteria=[stopping_criteria])
    outputs = tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()
    conv.messages[-1][-1] = outputs
    output = outputs.rsplit('</s>', 1)[0]
    return image, output

results = []
f = open('../CommVQA_dataset/annotations.json')
data = json.load(f)

parser = argparse.ArgumentParser(description='Optional app description')
parser.add_argument('--writefile', type=str,
                    required=True)

parser.add_argument('--generation_length', type=int,
                    help='A required integer positional argument',
                    default=21)

parser.add_argument('--context_description', action='store_true')

args = parser.parse_args()

fname = 'results/' + args.writefile

if (not os.path.exists(fname)):
    os.mkdir(fname)
else:
    print("Run ID already exists; please choose a different one")
    exit()

results = []

refs = []
hyps = []

answers_by_context = {}

refs_per_answer = {}

images = []

images_per_clipscore_context = []

for eval_datapoint in tqdm(data):
    #print("Datapoint ", eval_datapoint)
    image_path = '../' + eval_datapoint['image']

    images.append(image_path)

    contexts = ['science magazine', 'health', 'travel', 'shopping', 'news', 'social media']

    context = eval_datapoint['context']

    question = "USER: <image>\n" + eval_datapoint['question'] + "ASSISTANT:"

#    question = eval_datapoint['question']

    formatted_contexts = {
        'science_journals': 'science magazines',
        'social_media': 'social media'
    }

    if (context in formatted_contexts):
        context = formatted_contexts[context]

    if (args.context_description):
        question = "USER: <image>\n" +  'Assume someone is browsing a ' + context + ' website when they encounter this image. They cannot see the image directly, but they can access this image description: ' + eval_datapoint['description'] + ' Based on this description, they asked this follow-up question. Please answer based on the image. In your answer, prioritize details relevant to this person. Question: ' + eval_datapoint['question'] + "ASSISTANT:"
        #question = 'Assume someone is browsing a ' + context + ' website when they encounter this image. They cannot see the image directly, but they can access this image description: ' + eval_datapoint['description'] + ' Based on this description, they asked this follow-up question. Please answer based on the image. In your answer, prioritize details relevant to this person. Question: ' + eval_datapoint['question']

    print("Question: ", question)
   # question = 'Context: This image appears on a ' + context + ' website. Description: ' + eval_datapoint['description'] + ' Question: ' + eval_datapoint['question']
    image, generated_answer = caption_image(image_path, question)
    print("Generated answer: ", generated_answer)

    results.append({
            'image': eval_datapoint['image'],
            'description': eval_datapoint['description'],
            'context': eval_datapoint['context'],
            'question': eval_datapoint['question'],
            'prompt': question,
            'generated_answer': generated_answer,
            'answers': eval_datapoint['answers']
        })

    if (eval_datapoint['context'] not in answers_by_context):
        answers_by_context[eval_datapoint['context']] = []

    if (len(eval_datapoint['answers']) == 3):
        refs.append([eval_datapoint['answers'][0].lower().strip(), eval_datapoint['answers'][1].lower().strip(), eval_datapoint['answers'][2].lower().strip()])
    else:
        refs.append([eval_datapoint['answers'][0].lower().strip(), eval_datapoint['answers'][1].lower().strip()])
    hyps.append(generated_answer.lower().strip())

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

folder_path = fname + '/'

device = "cuda" if torch.cuda.is_available() else "cpu"
model, transform = clip.load("ViT-B/32", device=device, jit=False)
model.eval()
if (args.per_context_scores):
    for context in refs_per_context:
        with contextlib.redirect_stdout(None):
            metrics_per_context[context] = get_all_metrics(refs_per_context[context], hyps_per_context[context])
        _, per_instance_image_text, candidate_feats = get_clip_score(model, images_per_context[context], hyps_per_context[context], device)

        _, per_instance_text_text = get_refonlyclipscore(
            model, refs_per_context[context], candidate_feats, device)

        refclipscores = 2 * per_instance_image_text * per_instance_text_text / (per_instance_image_text + per_instance_text_text)

        scores = {image_id: {'CLIPScore': float(clipscore), 'RefCLIPScore': float(refclipscore)}
            for image_id, clipscore, refclipscore in
            zip(images, per_instance_image_text, refclipscores)}

        metrics_per_context[context]['clipscore'] = np.mean([s['CLIPScore'] for s in scores.values()])
        metrics_per_context[context]['clipscore_std'] = np.std([s['CLIPScore'] for s in scores.values()])

        metrics_per_context[context]['ref_clipscore'] = np.mean([s['RefCLIPScore'] for s in scores.values()])
        metrics_per_context[context]['ref_clipscore_std'] = np.std([s['RefCLIPScore'] for s in scores.values()])

    with open(folder_path + args.writefile + '_percontext_metrics.json', 'w') as fp:
        json.dump(metrics_per_context, fp)

print("Full metrics suite: ")

bertscore_results = bertscore.compute(predictions=hyps, references=refs, lang="en")

_, per_instance_image_text, candidate_feats = get_clip_score(model, images, hyps, device)

_, per_instance_text_text = get_refonlyclipscore(
            model, refs, candidate_feats, device)
# F-score
refclipscores = 2 * per_instance_image_text * per_instance_text_text / (per_instance_image_text + per_instance_text_text)

scores = {image_id: {'CLIPScore': float(clipscore), 'RefCLIPScore': float(refclipscore)}
            for image_id, clipscore, refclipscore in
            zip(images, per_instance_image_text, refclipscores)}

with contextlib.redirect_stdout(None):
    metrics_dict = get_all_metrics(refs, hyps)

clipscore = np.mean([s['CLIPScore'] for s in scores.values()])
clipscore_std = np.std([s['CLIPScore'] for s in scores.values()])

ref_clipscore = np.mean([s['RefCLIPScore'] for s in scores.values()])
ref_clipscore_std = np.std([s['RefCLIPScore'] for s in scores.values()])

print("CLIPScore: ", clipscore)
print("CLIPScore standard deviation: ", clipscore_std)

print("Ref-CLIPScore: ", ref_clipscore)
print("Ref-CLIPScore standard deviation: ", ref_clipscore_std)

metrics_dict['clipscore'] = clipscore
metrics_dict['clipscore_std_dev'] = clipscore_std

metrics_dict['ref_clipscore'] = ref_clipscore
metrics_dict['ref_clipscore_std_dev'] = ref_clipscore_std

metrics_dict['bertscore_avg_precision'] = np.mean(bertscore_results['precision'])
metrics_dict['bertscore_avg_recall'] = np.mean(bertscore_results['recall'])
metrics_dict['bertscore_avg_f1'] = np.mean(bertscore_results['f1'])

metrics_dict['full_bertscore'] = bertscore_results

print("Metrics dict for LLaVA evaluation: ", metrics_dict)

idx = list(range(0, len(results)))
df = pd.DataFrame(results, index=idx)

df.to_csv(folder_path + args.writefile + '_results.csv')

with open(folder_path + args.writefile + '_metrics_dict.json', 'w') as fp:
    json.dump(metrics_dict, fp)
