import pandas as pd
import matplotlib.pyplot as plt 
from clipscore_helper import get_all_metrics, get_clip_score, get_refonlyclipscore
import contextlib
import json
import random
import numpy as np
import random
import clip
from tqdm import tqdm

# TODO: Add in BLIP-2 reruns here as well,
# just to see what's happening!
files = [
#    'llava_baseline_rerun_results.csv',
 #   'llava_context_description_rerun_results.csv',
     'blip2/blip2_baseline_updated_prompt_results.csv',
     'blip2/blip2_context_description_results_previous_prompt.csv'
]

#files = [
#        'idefics/results/idefics_baseline/idefics_baseline_results.csv',
#        'mPLUG-Owl/mPLUG-Owl/mplug_owl_baseline_results.csv',
 #       'LLaVA/results/llava_baseline/llava_baseline_results.csv',
 #       'blip2/blip2_baseline_results.csv',
 #       'idefics/results/idefics_context_description/idefics_context_description_results.csv',
 #       'mPLUG-Owl/mPLUG-Owl/mplug_owl_context_description_results.csv',
 #       'LLaVA/results/llava_context_description/llava_context_description_results.csv',
 #       'blip2/blip2_context_description_results.csv'
  #  ]

metrics = ['meteor', 'rouge', 'cider', 'clipscore', 'bleu1', 'bleu2', 'bleu3', 'bleu4']

#formatted_files = ['LLaVA Baseline', 'BLIP-2 Baseline', 'LLaVA C+D']
#formatted_files = ['IDEFICS', 'mPLUG-Owl', 'LLaVA', 'BLIP-2', 'IDEFICS (C+D)', 'mPLUG-Owl (C+D)', 'LLaVA (C+D)', 'BLIP-2 (C+D)']
formatted_files = ['BLIP-2 Baseline', 'BLIP-2 C+D']

print("Length of formatted files ", len(formatted_files))
print("Length of all files ", len(files))

assert(len(formatted_files) == len(files))

metrics = ['meteor', 'rouge', 'cider', 'clipscore', 'bleu', 'ref_clipscore']

model_perf = {}
stddev_model_perf = {}

device = "cuda"

model, transform = clip.load("ViT-B/32", device=device, jit=False)
model.eval()

index = 0

f = open('dataset_annotations_more_than_three.json')
data = json.load(f)

# Get human evaluation metrics here!

candidates = []
references = []
hyps = []
images = []

for i in range(0, len(data)):
    # Seeds: 42, 39, 27
    random.seed(39)
    random.shuffle(data[i]['answers'])
    generated_answer = data[i]['answers'][0]

    print("Selected answer to compare: ", generated_answer)

    references.append([data[i]['answers'][1], data[i]['answers'][2],
                       data[i]['answers'][3]])
#    if (len(data[i]['answers']) == 3):
 #       references.append([data[i]['answers'][1].lower().strip(),
  #                     data[i]['answers'][2].lower().strip()])
  #  else:
   #     references.append([data[i]['answers'][1].lower().strip()])
            
    images.append(data[i]['image'])
        
    generated_answer = generated_answer.lower().strip()

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

    _, per_instance_text_text = get_refonlyclipscore(model,
                                                         refs,
                                                         candidate_feats,
                                                         device)
        
    refclipscores = 2 * per_instance_image_text * per_instance_text_text / (per_instance_image_text + per_instance_text_text)

    scores = {image_id: {'CLIPScore': float(clipscore), 'RefCLIPScore': float(refclipscore)}
            for image_id, clipscore, refclipscore in
            zip(img, per_instance_image_text, refclipscores)}

    ci_metrics_dict['clipscore'] = np.mean([s['CLIPScore'] for s in scores.values()])
    ci_metrics_dict['ref_clipscore'] = np.mean([s['RefCLIPScore'] for s in scores.values()])

    print("CLIPScore: ", ci_metrics_dict['clipscore'])

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
        stddev_model_perf[metric] = []

        if (metric == 'bleu'):
            model_perf['bleu1'] = []
            model_perf['bleu2'] = []
            model_perf['bleu3'] = []
            model_perf['bleu4'] = []

            stddev_model_perf['bleu1'] = []
            stddev_model_perf['bleu2'] = []
            stddev_model_perf['bleu3'] = []
            stddev_model_perf['bleu4'] = []

    if (metric == 'bleu'):
        for i in range(0, 4):
            model_perf[metric + str(i + 1)].append(np.mean(model_perf_for_file[metric + str(i + 1)]))
            stddev_model_perf[metric + str(i + 1)].append(np.std(model_perf_for_file[metric + str(i +  1)])/2)
    else:
        model_perf[metric].append(np.mean(model_perf_for_file[metric]))
        stddev_model_perf[metric].append(np.std(model_perf_for_file[metric])/2)

print("Model perf: ", model_perf)

fontsize = 17

colors = ['#332288', '#88CCEE', '#44AA99', '#117733', '#999933', '#DDCC77', '#CC6677', '#882255', '#AA4499']
colors = ['#88CCEE', '#44AA99', '#117733', '#999933', '#88CCEE', '#44AA99', '#117733', '#999933']

with open("model_perf_ci_overall.json", "w") as f:
    f.write(json.dumps(model_perf, indent = 4))

with open("stddev_perf_ci_overall.json", "w") as f:
    f.write(json.dumps(stddev_model_perf, indent = 4))

exit()

for current_file in files:
    print("File ", current_file)

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
            
        images.append(row['image'])

        print("Generated answer: ", row['generated_answer'])

        if (row['generated_answer'] != row['generated_answer']):
            row['generated_answer'] = ''
        
#        generated_answer = row['generated_answer'].replace(row['prompt'], '').lower().strip()

#        message_len = row['generated_answer'].find('ASSISTANT: ')
  #      print("message len ", message_len)
 #       generated_answer = row['generated_answer'][message_len + len('Assistant: '):].lower().strip()
        generated_answer = row['generated_answer'].lower().strip()
        print("File: ", current_file, " Generated answer: ", generated_answer)
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

            _, per_instance_text_text = get_refonlyclipscore(model,
                                                         refs,
                                                         candidate_feats,
                                                         device)
        
            refclipscores = 2 * per_instance_image_text * per_instance_text_text / (per_instance_image_text + per_instance_text_text)

        scores = {image_id: {'CLIPScore': float(clipscore), 'RefCLIPScore': float(refclipscore)}
            for image_id, clipscore, refclipscore in
            zip(img, per_instance_image_text, refclipscores)}

        ci_metrics_dict['clipscore'] = np.mean([s['CLIPScore'] for s in scores.values()])
        ci_metrics_dict['ref_clipscore'] = np.mean([s['RefCLIPScore'] for s in scores.values()])

        print("CLIPScore: ", ci_metrics_dict['clipscore'])

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
            stddev_model_perf[metric] = []

            if (metric == 'bleu'):
                model_perf['bleu1'] = []
                model_perf['bleu2'] = []
                model_perf['bleu3'] = []
                model_perf['bleu4'] = []

                stddev_model_perf['bleu1'] = []
                stddev_model_perf['bleu2'] = []
                stddev_model_perf['bleu3'] = []
                stddev_model_perf['bleu4'] = []

        if (metric == 'bleu'):
            for i in range(0, 4):
                model_perf[metric + str(i + 1)].append(np.mean(model_perf_for_file[metric + str(i + 1)]))
                stddev_model_perf[metric + str(i + 1)].append(np.std(model_perf_for_file[metric + str(i +  1)])/2)
        else:
            model_perf[metric].append(np.mean(model_perf_for_file[metric]))
            stddev_model_perf[metric].append(np.std(model_perf_for_file[metric])/2)

print("Model perf: ", model_perf)

fontsize = 17

colors = ['#332288', '#88CCEE', '#44AA99', '#117733', '#999933', '#DDCC77', '#CC6677', '#882255', '#AA4499']
colors = ['#88CCEE', '#44AA99', '#117733', '#999933', '#88CCEE', '#44AA99', '#117733', '#999933']

with open("model_perf_ci_overall.json", "w") as f:
    f.write(json.dumps(model_perf, indent = 4))

with open("stddev_perf_ci_overall.json", "w") as f:
    f.write(json.dumps(stddev_model_perf, indent = 4))

index = 0

metrics = ['meteor', 'rouge', 'cider', 'clipscore', 'bleu1', 'bleu4']

patterns = ['', '', '', '', '\\', '\\', '\\', '\\']

fig, ax = plt.subplots(2, 3, sharex=True, dpi=100, figsize=(20, 10))

for i,row in enumerate(ax):
    for j,col in enumerate(row):
        col.set_title('{}'.format(metrics[index]))
  
        if (metrics[index] == 'bleu'):
            index += 1
            continue

        bars = col.bar(formatted_files,
                model_perf[metrics[index]],
                color = colors,
                edgecolor='black',
                capsize=2
            )
        
        for bar, pattern in zip(bars, patterns):
            bar.set_hatch(pattern)

        col.errorbar(formatted_files,
            model_perf[metrics[index]],
            yerr=stddev_model_perf[metrics[index]],
            capsize=2)
        col.tick_params(axis='x', labelsize='10')

        index += 1

fig.tight_layout()
fig.autofmt_xdate(rotation=45)
plt.savefig('all_models_performance_plot.png')
