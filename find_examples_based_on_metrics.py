import pandas as pd
import matplotlib.pyplot as plt 
from clipscore_helper import get_all_metrics, get_clip_score
import contextlib
import json
import clip
import torch

files = [
        'LLaVA/results/llava_context_description/llava_context_description_results.csv']

# Save the top-10 examples from LLaVA which scored high on CLIPScore
# And low on CIDEr metrics
# Save the top-10 examples from LLaVA which scored low on CLIPScore
# And high on CIDEr metrics

device = "cuda" if torch.cuda.is_available() else "cpu"

model, transform = clip.load("ViT-B/32", device=device, jit=False)
model.eval()

results = []

for current_file in files:
    df = pd.read_csv(current_file)
    current_file = current_file.replace('_results.csv', '')

    f = open(current_file + '_metrics_dict.json')
    metrics_dict = json.load(f)
    metrics_dict_file = current_file.replace('.csv', '')

    candidates = []
    references = []
    hyps = []

    for idx, row in df.iterrows():
        ground_truth_answers = row['answers'].replace(']','').replace('[','').split(',')

        answer1 = ground_truth_answers[0].replace("'", '').replace('"', '').lower().strip()
        answer2 = ground_truth_answers[1].replace("'", '').replace('"', '').lower().strip()
        
        if (len(ground_truth_answers) == 3):
            answer3 = ground_truth_answers[2].replace("'", '').replace('"', '').lower().strip()
            references = [answer1, answer2, answer3]
        else:
            references = [answer1, answer2]
        
        if (row['generated_answer'] != row['generated_answer']):
            generated_answer = ''
        else:
            generated_answer = row['generated_answer'].lower().strip()

        candidates.append(generated_answer)

        print("Referneces ", references)
        print("Generatd answer ", generated_answer)

        with contextlib.redirect_stdout(None):
            metrics_dict = get_all_metrics([references], [generated_answer])
        # Evaluate the CLIPScore here

        image_paths = [row['image']]

        # get image-text clipscore
        _, per_instance_image_text, candidate_feats = get_clip_score(
            model, image_paths, candidates, device)
   
        scores = {image_id: {'CLIPScore': float(clipscore)}
                  for image_id, clipscore in
                  zip(image_paths, per_instance_image_text)}

        clipscore = [s['CLIPScore'] for s in scores.values()][0]

        metrics_dict['clipscore'] = clipscore
        metrics_dict['image'] = row['image']
        metrics_dict['generated_answer'] = row['generated_answer']
        metrics_dict['refs'] = references
        metrics_dict['question'] = row['question']
        metrics_dict['context'] = row['context']
        metrics_dict['description'] = row['description']
        results.append(metrics_dict)

        with open('per_datapoint_scores.json', 'w') as f:
            f.write(json.dumps(results, indent = 4))
