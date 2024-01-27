import json

f = open('per_datapoint_scores.json')

data = json.load(f)

# Rank the top-10 examples with high CLIPScore
# Rank the top-10 examples with low CLIPScore
# But is that the same??

sorted_list = sorted(data, key=lambda x: x['clipscore'])
#print("Top 10 lowest CLIPScores ", sorted_list[:10])
print("Top 10 highest CLIPScores ", sorted_list[-10:])
