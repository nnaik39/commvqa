import json
import pandas as pd

# Load two different results CSVs
# If the answers are different, log the image, context, answer, and description

df1 = pd.read_csv('results/idefics_context_description_340_questions/idefics_context_description_340_questions_results.csv')

df2 = pd.read_csv('results/idefics_baseline_340_questions/idefics_baseline_340_questions_results.csv')

answer_differences = []

for idx, row in df1.iterrows():
    answer1 = row['generated_answer']
    row2 = df2.iloc[int(idx)]
    answer2 = row2['generated_answer']

    if (answer1 != answer2):
        print("Baseline answer: ", answer2)
        print("Context+description answer: ", answer1)

        answer_differences.append({
            'image': row['image'],
            'context': row['context'],
            'description': row['description'],
            'question': row['question'],
            'baseline_answer': answer2,
            'context_description_answer': answer1
        })

idx = list(range(0, len(answer_differences)))
df = pd.DataFrame(answer_differences, index=idx)

df.to_csv('answer_differences_results.csv')
