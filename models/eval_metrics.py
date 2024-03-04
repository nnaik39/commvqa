import pandas as pd
from metrics_helper import get_all_metrics, get_clip_score
import contextlib
import json
import random
import numpy as np
import random
import clip

files = [
        'idefics/results/idefics_baseline_results.csv',
        'mPLUG-Owl/results/mplug_owl_baseline_results.csv',
        'LLaVA/results/llava_baseline_results.csv',
        'blip2/results/blip2_baseline_results.csv',
        'idefics/results/idefics_contextual_results.csv',
        'mPLUG-Owl/results/mplug_owl_context_description_results.csv',
        'LLaVA/results/llava_contextual_results.csv',
        'blip2/results/blip2_contextual_results.csv'
    ]

formatted_files = ['IDEFICS', 'mPLUG-Owl', 'LLaVA', 'BLIP-2', 'IDEFICS (C+D)', 'mPLUG-Owl (C+D)', 'LLaVA (C+D)', 'BLIP-2 (C+D)']

metrics = ['meteor', 'rouge', 'cider', 'bleu', 'clipscore']

model_perf = {}

device = "cuda"

model, transform = clip.load("ViT-B/32", device=device, jit=False)
model.eval()

index = 0

f = open('../CommVQA_dataset/annotations.json')
data = json.load(f)

for current_file in files:
    index += 1
    df = pd.read_csv(current_file)

    candidates = []
    references = []
    hyps = []
    images = []

    for idx, row in df.iterrows():
        ground_truth_answers = row['answers'].replace(']','').replace('[','').split(',')

        answer1 = ground_truth_answers[0].replace("'", '').replace('"', '').lower().strip()
        answer2 = ground_truth_answers[1].replace("'", '').replace('"', '').lower().strip()
        
        if (len(ground_truth_answers) == 3):
            answer3 = ground_truth_answers[2].replace("'", '').replace('"', '').lower().strip()
            references.append([answer1, answer2, answer3])
        else:
            references.append([answer1, answer2])
            
        images.append('../CommVQA_dataset/' + row['image'])

        print("Generated answer: ", row['generated_answer'])

        if (row['generated_answer'] != row['generated_answer']):
            row['generated_answer'] = ''
        
        generated_answer = row['generated_answer'].lower().strip()
        candidates.append(generated_answer)

    model_perf_for_file = {}

    for i in range(0, 3):
        num_samples = int(0.8 * len(references))
        random.seed(i)
        refs, hyps, img = zip(*random.sample(list(zip(references, candidates, images)), num_samples))

        with contextlib.redirect_stdout(None):
            ci_metrics_dict = get_all_metrics(refs, hyps)

            print("Confidence interval metrics dict: ", ci_metrics_dict)
            
            _, per_instance_image_text, candidate_feats = get_clip_score(model,
                                                                     list(img),
                                                                      hyps,
                                                                     device)

        scores = {image_id: {'CLIPScore': float(clipscore)}
            for image_id, clipscore in
            zip(img, per_instance_image_text)}

        ci_metrics_dict['clipscore'] = np.mean([s['CLIPScore'] for s in scores.values()])

        for metric in metrics:
            if (metric not in model_perf_for_file):
                model_perf_for_file[metric] = []

                if (metric == 'bleu'):
                    model_perf_for_file['bleu1'] = []
                    model_perf_for_file['bleu2'] = []
                    model_perf_for_file['bleu3'] = []
                    model_perf_for_file['bleu4'] = []
    
            if (metric == 'bleu'):
                for i in range(0, len(ci_metrics_dict['bleu'])):
                    model_perf_for_file[metric + str(i + 1)].append(ci_metrics_dict[metric][i])
            else:
                model_perf_for_file[metric].append(ci_metrics_dict[metric])

    for metric in metrics:
        if (metric not in model_perf):
            model_perf[metric] = []

            if (metric == 'bleu'):
                for i in range(0, 4):
                    model_perf[metric + str(i + 1)] = []

        if (metric == 'bleu'):
            for i in range(0, 4):
                model_perf[metric + str(i + 1)].append(np.mean(model_perf_for_file[metric + str(i + 1)]))
        else:
            model_perf[metric].append(np.mean(model_perf_for_file[metric]))

with open("model_performance.json", "w") as f:
    f.write(json.dumps(model_perf, indent = 4))