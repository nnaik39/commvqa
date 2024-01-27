import json
import numpy as np

f = open('dataset_annotations_cleaned.json')
data = json.load(f)

avg_answer_length = 0
num_answers = 0

answers = []

num_word_length = []

for datapoint in data:
    for answer in datapoint['answers']:
        print("Answer ", answer)
        answers.append(len(answer))
        avg_answer_length += len(answer)
        num_answers += 1

        num_word_length.append(len(answer.split(' ')))

print("Avg answer length ", np.mean(answers))
print("Stddev answer length ", np.std(answers))
# Take standard deviation here?

print("Avg word length ", np.mean(num_word_length))
print("Standard deviation word length ", np.std(num_word_length))
