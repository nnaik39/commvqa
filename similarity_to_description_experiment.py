import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import pandas as pd
import textdistance
from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer('all-MiniLM-L6-v2')

# Run this experiment with IDEFICS baseline, IDEFICS description-only, and IDEFICS context + description

files = [
        ('idefics/results/idefics_baseline/idefics_baseline_results.csv', 'IDEFICS'),
        ('idefics/results/idefics_context_description/idefics_context_description_results.csv', 'IDEFICS C+D'),
        ('mPLUG-Owl/mPLUG-Owl/mplug_owl_baseline_results.csv', 'mPLUG-Owl'),
        ('mPLUG-Owl/mPLUG-Owl/mplug_owl_context_description_results.csv', 'mPLUG-Owl C+D'),
        ('LLaVA/results/llava_baseline/llava_baseline_results.csv', 'LLaVA'),
        ('LLaVA/results/llava_context_description/llava_context_description_results.csv', 'LLaVA C+D'),
        ('blip2/blip2_baseline_results.csv', 'BLIP-2'),
        ('blip2/blip2_context_description_results.csv', 'BLIP-2 C+D')
        ]

categories = []
sbert_similarity_values = []
levenshtein_similarity_values = []

for f, formatted_file_name in files:
    df = pd.read_csv(f)

    num_responses = 0

    avg_sbert_sim = 0
    avg_lev_sim = 0

    for idx, row in df.iterrows():
        if (row['generated_answer'] != row['generated_answer']):
            continue

        answer_embeddings = model.encode(row['generated_answer'], convert_to_tensor=True)
        ground_truth_embeddings = model.encode(row['description'], convert_to_tensor=True)

        cos_sim = util.cos_sim(answer_embeddings, ground_truth_embeddings).item()
        levenshtein_similarity = textdistance.levenshtein.normalized_similarity(row['generated_answer'], row['description'])

        sbert_similarity_values.append(cos_sim)
        levenshtein_similarity_values.append(levenshtein_similarity)

        avg_sbert_sim += cos_sim
        avg_lev_sim += levenshtein_similarity
        num_responses += 1

        categories.append(formatted_file_name)

    print("Avg SBert similarity for ", formatted_file_name, " :", avg_sbert_sim)
    print("Avg Levenshtein similarity: ", formatted_file_name, " :", avg_lev_sim)

categories = pd.Series(categories)
levenshtein_values = pd.Series(levenshtein_similarity_values)
cosine_values = pd.Series(sbert_similarity_values)

color_codes = {"IDEFICS": "#0077BB",
        "IDEFICS C+D": "#0077BB",
        "mPLUG-Owl": "#33BBEE",
        "mPLUG-Owl C+D": "#33BBEE",
        "LLaVA": "#009988",
        "LLaVA C+D": "#009988",
        "BLIP-2": "#EE7733",
        "BLIP-2 C+D": "#EE7733"
        }

sns.violinplot(x=categories, y=sbert_similarity_values, palette=color_codes)
plt.xticks(rotation=30, ha='right')

#plt.figure(figsize=(10,6))

plt.savefig('/sailhome/nanditan/description_cosine_similarity.png')

plt.clf()

sns.violinplot(x=categories, y=levenshtein_similarity_values, palette=color_codes)
plt.xticks(rotation=30, ha='right')

#plt.figure(figsize=(10,6))
plt.savefig('/sailhome/nanditan/description_levenshtein_similarity.png')
