import contextlib
import pandas as pd
import json
import clip
from tqdm import tqdm

from PIL import Image
import requests
from transformers import Blip2Processor, Blip2ForConditionalGeneration
import torch
from evaluate import load
import requests

bertscore = load("bertscore")
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"

processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
model = Blip2ForConditionalGeneration.from_pretrained(
    "Salesforce/blip2-opt-2.7b", load_in_8bit=True, device_map={"": 0}, torch_dtype=torch.float16
)

f = open('../dataset_annotations_cleaned.json')
data = json.load(f)

results = []

refs = []
hyps = []

images_for_clipscore = []

writefile = "blip2_baseline_updated_prompt"

for eval_datapoint in data:
    image_path = eval_datapoint['image']

    question = "Question: " + eval_datapoint['question'] + " Answer:"
#    question = 'Assume someone is browsing a ' + eval_datapoint['context'] + ' website when they encounter this image. They cannot see the image directly, but they can access this image description: ' + eval_datapoint['description'] + ' Based on this description, they asked this follow-up question. Please answer based on the image. In your answer, prioritize details relevant to this person. Question: ' + eval_datapoint['question']

    image = Image.open('../' + image_path).convert("RGB")

    images_for_clipscore.append('../' + image_path)
    
    inputs = processor(images=image, text=question, return_tensors="pt").to(device="cuda", dtype=torch.float16)

    generated_ids = model.generate(**inputs,
            do_sample=False,
            max_length=256)

    generated_answer = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

    print("Question: ", question)
    print("Answer: ", generated_answer)

    idx = list(range(0, len(results)))
    df = pd.DataFrame(results, index=idx)
    df.to_csv(writefile + '_results.csv')
    results.append({
            'image': eval_datapoint['image'],
            'description': eval_datapoint['description'],
            'prompt': question,
            'context': eval_datapoint['context'],
            'question': eval_datapoint['question'],
            'generated_answer': generated_answer,
            'answers': eval_datapoint['answers']
        })

    if (len(eval_datapoint['answers']) == 3):
        refs.append([eval_datapoint['answers'][0].lower().strip(), eval_datapoint['answers'][1].lower().strip(), eval_datapoint['answers'][2].lower().strip()])
    else:
        refs.append([eval_datapoint['answers'][0].lower().strip(), eval_datapoint['answers'][1].lower().strip()])
    hyps.append(generated_answer.lower())

results_per_context = {}
refs_per_context = {}
hyps_per_context = {}
images_per_context = {}

idx = list(range(0, len(results)))
df = pd.DataFrame(results, index=idx)
df.to_csv(writefile + '_results.csv')