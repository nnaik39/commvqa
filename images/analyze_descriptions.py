import json
f = open('descriptions.json')

data = json.load(f)

# Count number of unique images
images = []

count_filenames = {}

for i in data:
    images.append(i['filename'])
    if (i['filename'] not in count_filenames):
        count_filenames[i['filename']] = 0
    count_filenames[i['filename']] += 1

for filename in count_filenames:
    if (count_filenames[filename] != 2):
        print("Filename ", filename)