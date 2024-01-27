import pandas as pd
import matplotlib.pyplot as plt 
from clipscore_helper import get_all_metrics
import contextlib
import json

files = [
        'idefics/results/idefics_baseline/idefics_baseline_results.csv',
        'mPLUG-Owl/mPLUG-Owl/mplug_owl_baseline_results.csv',
        'LLaVA/results/llava_baseline/llava_baseline_results.csv',
        'blip2/blip2_baseline_results.csv',
        'idefics/results/idefics_context_description/idefics_context_description_results.csv',
        'mPLUG-Owl/mPLUG-Owl/mplug_owl_context_description_results.csv',
        'LLaVA/results/llava_context_description/llava_context_description_results.csv',
        'blip2/blip2_context_description_results.csv'
    ]

metrics = ['meteor', 'rouge', 'cider', 'clipscore', 'bleu1', 'bleu2', 'bleu3', 'bleu4']

formatted_files = ['IDEFICS', 'mPLUG-Owl', 'LLaVA', 'BLIP-2', 'IDEFICS (C+D)', 'mPLUG-Owl (C+D)', 'LLaVA (C+D)', 'BLIP-2 (C+D)']
#formatted_files = ['IDEFICS', 'mPLUG-OWL', 'LLaVA', 'BLIP-2']

#files = [
#        'idefics/results/idefics_baseline/idefics_baseline_results.csv',
#        'idefics/results/idefics_description_only/idefics_description_only_results.csv',
 #       'idefics/results/idefics_context_only/idefics_context_only_results.csv',
#        'mPLUG-Owl/mPLUG-Owl/mplug_owl_baseline_results.csv',
#        'LLaVA/results/llava_baseline/llava_baseline_results.csv',
#        'blip2/blip2_baseline_results.csv',
  #      'idefics/results/idefics_context_description/idefics_context_description_results.csv',
 #       'mPLUG-Owl/mPLUG-Owl/mplug_owl_context_description_results.csv',
#        'LLaVA/results/llava_context_description/llava_context_description_results.csv',
 #       'blip2/blip2_context_description_results.csv'
   # ]

#formatted_files = ['IDEFICS', 'IDEFICS (D)', 'IDEFICS (C)', 'IDEFICS (C+D)']

assert(len(formatted_files) == len(files))

metrics = ['meteor', 'rouge', 'cider', 'clipscore', 'bleu']

model_perf = {}

for current_file in files:
    print("File ", current_file)

    csv_file = current_file
    df = pd.read_csv(current_file)
    current_file = current_file.replace('_results.csv', '')

    f = open(current_file + '_metrics_dict.json')
    metrics_dict = json.load(f)

    print("Initial metrics dict: ", metrics_dict)

    metrics_dict_file = current_file.replace('.csv', '')

    candidates = []
    references = []
    hyps = []

    for idx, row in df.iterrows():
        # Lowercase and strip away white space here!
        ground_truth_answers = row['answers'].replace(']','').replace('[','').split(',')

        answer1 = ground_truth_answers[0].replace("'", '').replace('"', '').lower().strip()
        answer2 = ground_truth_answers[1].replace("'", '').replace('"', '').lower().strip()
        
        if (len(ground_truth_answers) == 3):
            answer3 = ground_truth_answers[2].replace("'", '').replace('"', '').lower().strip()
            references.append([answer1, answer2, answer3])
        else:
            references.append([answer1, answer2])
        
        generated_answer = row['generated_answer'].lower().strip()

        candidates.append(generated_answer)

    with contextlib.redirect_stdout(None):
        metrics_dict = get_all_metrics(references, candidates)

    print("Candidates\n ", candidates)

    print("References\n ", references)

    with open('candidates.txt', 'w') as f:
        f.write(json.dumps(candidates, indent=4))

    with open('candidates.txt', 'w') as f:
        f.write(json.dumps(candidates, indent=4))

    print("Pycocoevalcap metrics dict for model: ", current_file)
    print(metrics_dict)
    print("Length of references ", len(references))
    print("Length of candidates ", len(candidates))

    exit()

    for metric in metrics:
        if (metric not in model_perf):
            model_perf[metric] = []

            if (metric == 'bleu'):
                model_perf['bleu1'] = []
                model_perf['bleu2'] = []
                model_perf['bleu3'] = []
                model_perf['bleu4'] = []
    
        if (metric == 'bleu'):
            for i in range(0, len(metrics_dict['bleu'])):
                model_perf[metric + str(i + 1)].append(metrics_dict[metric][i])
        else:
            model_perf[metric].append(metrics_dict[metric])

fig, ax = plt.subplots(1, 4, sharex=True, sharey=True, dpi=100, figsize=(20, 10))

fontsize = 17

colors = ['#332288', '#88CCEE', '#44AA99', '#117733', '#999933', '#DDCC77', '#CC6677', '#882255', '#AA4499']

colors = ['#88CCEE', '#44AA99', '#117733', '#999933', '#88CCEE', '#44AA99', '#117733', '#999933']

print("Files ", files)
print("Model perf ", model_perf)

index = 0

metrics = ['meteor', 'rouge', 'cider', 'clipscore']

patterns = ['', '', '', '', '\\', '\\', '\\', '\\']

for i,row in enumerate(ax):
   # for j,col in enumerate(row):
        row.set_title('{}'.format(metrics[index]))
  
        if (metrics[index] == 'bleu'):
            index += 1
            continue

        bars = row.bar(formatted_files,
                model_perf[metrics[index]],
                color = colors,
                edgecolor='black',
                capsize=2
            )
        
        for bar, pattern in zip(bars, patterns):
            bar.set_hatch(pattern);

        row.tick_params(axis='x', labelsize='10')

        index += 1

fig.tight_layout()
fig.autofmt_xdate(rotation=45)
plt.savefig('/sailhome/nanditan/all_models_performance_plot.png')
