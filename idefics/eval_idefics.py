import torch, contextlib, wandb, json, os
from clipscore_helper import get_all_metrics, get_clip_score, get_refonlyclipscore
import pandas as pd
from transformers import IdeficsForVisionText2Text, AutoProcessor
from PIL import Image
from evaluate import load
from tqdm import tqdm
import clip
import random
import numpy as np

wandb.init(project="comm_vqa")

device = "cuda" if torch.cuda.is_available() else "cpu"

f = open('../dataset_annotations_cleaned.json')
data = json.load(f)

results = []

clip_model, transform = clip.load("ViT-B/32", device=device, jit=False)
clip_model.eval()

def evaluate_metrics(refs, cands, images, num_confidence_intervals):
    if (num_confidence_intervals > 1):
        metrics_dict_list = []

        for i in range(0, num_confidence_intervals):
            random.seed(i)
            indices = list(np.arange(0, len(refs)))
            print("Indices ", indices)

            sample = random.sample(indices, int(0.8 * len(refs)))

            refs_sample, cands_sample, images_sample = [refs[i] for i in indices], [cands[i] for i in indices], [images[i] for i in indices]

            print("Refs sample ", refs_sample)
            print("Cands sample ", cands_sample)

            with contextlib.redirect_stdout(None):
                metrics_dict = get_all_metrics(refs_sample, cands_sample)

            _, per_instance_image_text, candidate_feats = get_clip_score(
                clip_model, images_sample, cands_sample, device)

            _, per_instance_text_text = get_refonlyclipscore(
                clip_model, refs_sample, candidate_feats, device)

            refclipscores = 2 * per_instance_image_text * per_instance_text_text / (per_instance_image_text + per_instance_text_text)

            scores = {image_id: {'CLIPScore': float(clipscore), 'RefCLIPScore': float(refclipscore)}
                    for image_id, clipscore, refclipscore in
                    zip(images, per_instance_image_text, refclipscores)}

            metrics_dict['clipscore'] = np.mean([s['CLIPScore'] for s in scores.values()][0])
            metrics_dict['refclipscore'] = np.mean([s['RefCLIPScore'] for s in scores.values()][0])
            metrics_dict['full_clipscore'] = [s['CLIPScore'] for s in scores.values()][0]
            metrics_dict['full_refclipscore'] = [s['RefCLIPScore'] for s in scores.values()][0]

            print("Metrics dict: ", metrics_dict)
            
            metrics_dict_list.append(metrics_dict)
        
        mean_metrics_dict = {}

        print("Metrics dict list ", metrics_dict_list)

        for metric in metrics_dict_list[0]:
            print("Metric ", metric)
            if (metric == 'bleu'):
                for i in range(0, 4):
                    metrics_dict_scores = [confidence_interval['bleu'][i] for confidence_interval in metrics_dict_list][0]
                    print("Metrics dict scores ", metrics_dict_scores)
                    mean_metrics_dict['bleu' + str(i + 1)] = np.mean(metrics_dict_scores)
            else:
                metrics_dict_scores = [confidence_interval[metric] for confidence_interval in metrics_dict_list]
                print("Metrics dict scores ", metrics_dict_scores)
                mean_metrics_dict[metric] = np.mean(metrics_dict_scores)
        return mean_metrics_dict
    else:
        print("Refs ", [refs])
        print("Cands ", [cands])

        with contextlib.redirect_stdout(None):
            metrics_dict = get_all_metrics(refs, cands)
        _, per_instance_image_text, candidate_feats = get_clip_score(
                clip_model, images, cands, device)
        _, per_instance_text_text = get_refonlyclipscore(
                clip_model, refs, candidate_feats, device)
        refclipscores = 2 * per_instance_image_text * per_instance_text_text / (per_instance_image_text + per_instance_text_text)
        scores = {image_id: {'CLIPScore': float(clipscore), 'RefCLIPScore': float(refclipscore)}
                    for image_id, clipscore, refclipscore in
                    zip(images, per_instance_image_text, refclipscores)}

        metrics_dict['clipscore'] = np.mean([s['CLIPScore'] for s in scores.values()][0])
        metrics_dict['refclipscore'] = np.mean([s['RefCLIPScore'] for s in scores.values()][0])
        metrics_dict['full_clipscore'] = [s['CLIPScore'] for s in scores.values()][0]
        metrics_dict['full_refclipscore'] = [s['RefCLIPScore'] for s in scores.values()][0]
        return metrics_dict

refs = []
hyps = []

checkpoint = "HuggingFaceM4/idefics-9b-instruct"
model = IdeficsForVisionText2Text.from_pretrained(checkpoint, torch_dtype=torch.bfloat16).to(device)
processor = AutoProcessor.from_pretrained(checkpoint)

formatted_contexts = {'social_media': 'social media', 'science_journals': 'science magazine'}

images = []

condition = "baseline"

index = 0

for eval_datapoint in tqdm(data):
    if (index == 10):
        break
    
    image_path, question, context = eval_datapoint['image'], eval_datapoint['question'], eval_datapoint['context']

    images.append('../' + image_path)

    index += 1

    if (eval_datapoint['context'] in formatted_contexts):
        context = formatted_contexts[eval_datapoint['context']]

    # Context + description condition
    if (condition == "context_description"):
        question = 'Assume someone is browsing a ' + context + ' website when they encounter this image. They cannot see the image directly, but they can access this image description: ' + eval_datapoint['description'] + ' Based on this description, they asked this follow-up question. Please answer based on the image. In your answer, prioritize details relevant to this person. Question: ' + eval_datapoint['question']
    # Description-only condition
    if (condition == "description_only"):
        question = 'Assume someone is browsing a website when they encounter this image. They cannot see the image directly, but they can access this image description: ' + eval_datapoint['description'] + ' Based on this description, they asked this follow-up question. Please answer based on the image. In your answer, prioritize details relevant to this person. Question: ' + eval_datapoint['question']
    if (condition == "context_only"):
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
    generated_answer = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].replace(prompts[0][0], '').replace(' \nAssistant: ', '').replace('Assistant:', '').strip()

    if (len(eval_datapoint['answers']) == 3):
        refs.append([eval_datapoint['answers'][0].lower().strip(), eval_datapoint['answers'][1].lower().strip(), eval_datapoint['answers'][2].lower().strip()])
    else:
        refs.append([eval_datapoint['answers'][0].lower().strip(), eval_datapoint['answers'][1].lower().strip()])

    # Note: I'm trying too hard with this!!
#    metrics_dict = evaluate_metrics(refs, [generated_answer], ['../' + image_path], 1)

    hyps.append(generated_answer.lower().strip())

    results.append({
            'image': eval_datapoint['image'],
            'description': eval_datapoint['description'],
            'context': eval_datapoint['context'],
            'prompt': prompts[0][0],
            'question': eval_datapoint['question'],
            'answers': eval_datapoint['answers'],
            'generated_answer': generated_answer
#            'metrics': metrics_dict
    })

results_per_context = {}
refs_per_context = {}
hyps_per_context = {}
images_per_context = {}

index = 0 

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

writeFile = "idefics_baseline"

folder_path = "results/" + writeFile + "/" + writeFile

os.mkdir("results/" + writeFile)

metrics_per_context = {}

for context in refs_per_context:
    metrics_per_context[context] = evaluate_metrics(
        refs_per_context[context],
        hyps_per_context[context],
        images_per_context[context],
        5)

with open(folder_path + '_percontext_metrics.json', 'w') as fp:
        json.dump(metrics_per_context, fp)

print("Metrics dict for IDEFICS evaluation: ", metrics_dict)

idx = list(range(0, len(results)))
df = pd.DataFrame(results, index=idx)
df.to_csv(folder_path + '_results.csv')

evaluate_metrics(refs, cands, images, 5)
wandb.log(metrics_dict)

with open(folder_path + '_metrics_dict.json', 'w') as fp:
    json.dump(metrics_dict, fp)