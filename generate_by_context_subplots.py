import pandas as pd
import matplotlib.pyplot as plt 
from clipscore_helper import get_all_metrics
import contextlib
import json
import random
import numpy as np

files = [
        'idefics/results/idefics_baseline_rerun_context_scores/idefics_baseline_rerun_context_scores_results.csv',
        'mPLUG-Owl/mPLUG-Owl/mplug_owl_baseline_results.csv',
        'LLaVA/results/llava_baseline/llava_baseline_results.csv',
        'blip2/blip2_baseline_results.csv',
        'idefics/results/idefics_context_description_rerun/idefics_context_description_rerun_results.csv',
        'mPLUG-Owl/mPLUG-Owl/mplug_owl_context_description_rerun_results.csv',
        'LLaVA/results/llava_context_description/llava_context_description_results.csv',
        'blip2/blip2_context_description_results.csv'
    ]

formatted_files = ['IDEFICS', 'mPLUG-Owl', 'LLaVA', 'BLIP-2', 'IDEFICS C+D', 'mPLUG-Owl C+D', 'LLaVA C+D', 'BLIP-2 C+D']

assert(len(formatted_files) == len(files))

metrics = ['meteor', 'rouge', 'cider', 'spice', 'bleu']

model_perf = {}

current_metric = "cider"
random.seed(42)

contexts = ['science_journals', 'social_media', 'health', 'travel', 'news', 'shopping']

for index in range(0, len(files)):
    current_file = files[index]
    print("File ", current_file)

    df = pd.read_csv(current_file)
 #   current_file = current_file.replace('_results.csv', '_percontext_metrics.json')

  #  f = open(current_file)

   # metrics_dict = json.load(f)
    #metrics_dict_file = current_file.replace('.csv', '')

    candidates = []
    references = []
    hyps = []

    refs_by_context = {}
    hyps_by_context = {}

    for idx, row in df.iterrows():
        if (row['context'] not in refs_by_context):
            refs_by_context[row['context']] = []
            hyps_by_context[row['context']] = []

        ground_truth_answers = row['answers'].replace(']','').replace('[','').split(',')

        answer1 = ground_truth_answers[0].replace("'", '').replace('"', '').lower().strip()
        answer2 = ground_truth_answers[1].replace("'", '').replace('"', '').lower().strip()
        
        if (len(ground_truth_answers) == 3):
            answer3 = ground_truth_answers[2].replace("'", '').replace('"', '').lower().strip()
            refs_by_context[row['context']].append([answer1, answer2, answer3])
        else:
            refs_by_context[row['context']].append([answer1, answer2])

        if (row['generated_answer'] != row['generated_answer']):
            generated_answer = ''
        else:
            generated_answer = row['generated_answer'].lower().strip()

        hyps_by_context[row['context']].append(generated_answer)
        candidates.append(generated_answer)
        hyps.append(generated_answer)

    cider_per_model = {}

    full_metrics_dict = []

    for context in refs_by_context:
        cider_per_model[context] = []

        for i in range(0, 3):
            num_samples = int(0.8 * len(refs_by_context[context]))

            random.seed(i)
            refs, hyps = zip(*random.sample(list(zip(refs_by_context[context], hyps_by_context[context])), num_samples))

            with contextlib.redirect_stdout(None):
                metrics_dict = get_all_metrics(refs, hyps)
            cider_per_model[context].append(metrics_dict['cider'])

    print("Pycocoevalcap metrics dict for model: ", current_file)
    print(metrics_dict)

    print("Cider per model ", cider_per_model)

    model = formatted_files[index]

    for context in contexts:
        if (model not in model_perf):
            model_perf[model] = []
        model_perf[model].append(cider_per_model[context])

fig, ax = plt.subplots(1, 8, sharex=True, sharey=True, dpi=100, figsize=(20, 10))

fontsize = 17

colors = ['#88CCEE', '#44AA99', '#117733', '#999933', '#88CCEE', '#44AA99', '#117733', '#999933']

colors = ['#332288', '#88CCEE', '#44AA99', '#117733', '#999933', '#DDCC77', '#CC6677', '#882255', '#AA4499']

print("Files ", files)
print("Model perf ", model_perf)

index = 0

metrics = ['meteor', 'rouge', 'cider', 'spice', 'bleu1', 'bleu2', 'bleu3', 'bleu4']

models = ['IDEFICS', 'mPLUG-Owl', 'LLaVA', 'BLIP-2', 'IDEFICS C+D', 'mPLUG-Owl C+D', 'LLaVA C+D', 'BLIP-2 C+D']

#plt.set_title('{}'.format(models[index]))

means = {}
stddevs = {}

with open("by_context_scores_model_perf.json", "w") as f:
    f.write(json.dumps(model_perf, indent = 4))

for model in model_perf:
    means[model] = []
    stddevs[model] = []

    for by_ctxt_scores in model_perf[model]:
        means[model].append(np.mean(by_ctxt_scores))
        stddevs[model].append(np.std(by_ctxt_scores)/2)

print("Means ", means)
print("Stddevs ", stddevs)

#plt.bar(contexts,
#            means['IDEFICS'],
#            color = colors,
#            edgecolor='black',
#            yerr=stddevs['IDEFICS'],
#            capsize=2
#)

#plt.tick_params(axis='x', labelsize='6')

   # index += 1
for i,col in enumerate(ax):
    col.set_title('{}'.format(models[index]))
 
    bars = col.bar(contexts,
            means[models[index]],
            color = colors,
            edgecolor='black',
            yerr=stddevs[models[index]],
            capsize=2
        )

    col.tick_params(axis='x', labelsize='6')

    index += 1

fig.tight_layout()
fig.autofmt_xdate(rotation=45)
plt.savefig('/sailhome/nanditan/' + current_metric + '_by_context_performance_plot.png')
