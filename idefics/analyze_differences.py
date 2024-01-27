import pandas as pd

baseline = pd.read_csv('idefics_description_only_results.csv')
context_description = pd.read_csv('idefics_context_description_results.csv')

baseline_answers = []
cd_answers = []

for idx, row in baseline.iterrows():
    # If they're not equal, then print out the differences here!
    baseline_answers.append(row['generated_answer'])

for idx, row in context_description.iterrows():
    cd_answers.append(row['generated_answer'])

for i in range(0, len(baseline_answers)):
    if (baseline_answers[i] != cd_answers[i]):
        print("\n")
        print("Description-only IDEFICS answer: ", baseline_answers[i])
        print("Context-description IDEFICS anwer: ", cd_answers[i])

