import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import json 

import pandas as pd
import textdistance
from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer('all-MiniLM-L6-v2')

files = [
        ('dataset_annotations_cleaned.json', 'Ground-Truth Answers'),
        ('idefics/results/idefics_baseline/idefics_baseline_results.csv', 'IDEFICS'),
        ('idefics/results/idefics_context_description/idefics_context_description_results.csv', 'IDEFICS C+D'),
        ('mPLUG-Owl/mPLUG-Owl/mplug_owl_baseline_results.csv', 'mPLUG-Owl'),
        ('mPLUG-Owl/mPLUG-Owl/mplug_owl_context_description_results.csv', 'mPLUG-Owl C+D'),
        ('LLaVA/results/llava_baseline/llava_baseline_results.csv', 'LLaVA'),
        ('LLaVA/results/llava_context_description/llava_context_description_results.csv', 'LLaVA C+D'),
        ('blip2/blip2_baseline_updated_prompt_results.csv', 'BLIP-2'),
        ('blip2/blip2_context_description_results_previous_prompt.csv', 'BLIP-2 C+D')
        ]

categories = []
sbert_similarity_values = []
levenshtein_similarity_values = []

across_models_p_values = []

for f, formatted_file_name in files:
    num_responses = 0

    avg_sbert_sim = 0
    avg_lev_sim = 0

    if (formatted_file_name == 'Ground-Truth Answers'):
        data = json.load(open(f))

        for i in data:
            for answer in i['answers']:
                answer_embeddings = model.encode(answer, convert_to_tensor=True)
                ground_truth_embeddings = model.encode(i['description'], convert_to_tensor=True)

                cos_sim = util.cos_sim(answer_embeddings, ground_truth_embeddings).item()
                levenshtein_similarity = textdistance.levenshtein.normalized_similarity(answer, i['description'])

                across_models_p_values.append({
                    'f': formatted_file_name,
                    'cosine_similarity': cos_sim
                })

                sbert_similarity_values.append(cos_sim)
                levenshtein_similarity_values.append(levenshtein_similarity)
                avg_sbert_sim += cos_sim
                avg_lev_sim += levenshtein_similarity
                num_responses += 1

                categories.append(formatted_file_name)
    else:
        df = pd.read_csv(f)
    
        for idx, row in df.iterrows():
            if (row['generated_answer'] != row['generated_answer']):
                continue

            answer_embeddings = model.encode(row['generated_answer'], convert_to_tensor=True)
            ground_truth_embeddings = model.encode(row['description'], convert_to_tensor=True)

            cos_sim = util.cos_sim(answer_embeddings, ground_truth_embeddings).item()
            levenshtein_similarity = textdistance.levenshtein.normalized_similarity(row['generated_answer'], row['description'])

            across_models_p_values.append({
                'f': f,
                'cosine_similarity': cos_sim
            })
            sbert_similarity_values.append(cos_sim)
            levenshtein_similarity_values.append(levenshtein_similarity)

            avg_sbert_sim += cos_sim
            avg_lev_sim += levenshtein_similarity
            num_responses += 1

            categories.append(formatted_file_name)

#    avg_sbert_sim /= num_responses
 #   avg_lev_sim /= num_responses
    print("Avg SBert similarity for ", formatted_file_name, " :", avg_sbert_sim)
    print("Avg Levenshtein similarity: ", formatted_file_name, " :", avg_lev_sim)

categories = pd.Series(categories)
levenshtein_values = pd.Series(levenshtein_similarity_values)
cosine_values = pd.Series(sbert_similarity_values)

color_codes = {"IDEFICS": "#009988",
        "IDEFICS C+D": "#EE7733",
        "mPLUG-Owl": "#009988",
        "mPLUG-Owl C+D": "#EE7733",
        "LLaVA": "#009988",
        "LLaVA C+D": "#EE7733",
        "BLIP-2": "#009988",
        "BLIP-2 C+D": "#EE7733",
        "Ground-Truth Answers": "#33BBEE"
        }

models = color_codes.keys()

print("Categories: ", categories)
print("SBert similarity values: ", sbert_similarity_values)

idx = list(range(0, len(across_models_p_values)))
df = pd.DataFrame(across_models_p_values, index=idx)
df.to_csv('cosine_similarity_values.csv')

# TODO: Fix this plot here!
width = 0.25  # the width of the bars
multiplier = 0

x = np.arange(len(contexts))  # the label locations

plt.title('{}'.format("NER Analysis Across Contexts"), fontsize=fontsize)

color_map = {
    'PERSON': '#228833',
    'ORG': '#66CCEE'
}

# x-axis is the contexts
sns.violinplot(x=categories, y=sbert_similarity_values, palette=color_codes)
plt.xticks(rotation=30, ha='right')

plt.savefig('description_cosine_similarity.png')

plt.clf()

sns.violinplot(x=categories, y=levenshtein_similarity_values, palette=color_codes)
plt.xticks(rotation=30, ha='right')

#plt.figure(figsize=(10,6))
plt.savefig('description_levenshtein_similarity.png')
