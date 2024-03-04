import pandas as pd
import json

from PIL import Image
from transformers import Blip2Processor, Blip2ForConditionalGeneration
import torch
import argparse

device = "cuda" if torch.cuda.is_available() else "cpu"

processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
model = Blip2ForConditionalGeneration.from_pretrained(
    "Salesforce/blip2-opt-2.7b", load_in_8bit=True, device_map={"": 0}, torch_dtype=torch.float16
)

parser = argparse.ArgumentParser(description='Optional app description')
parser.add_argument('--writefile', type=str,
                    required=True)
parser.add_argument('--contextual', action='store_true')
args = parser.parse_args()

f = open('../../CommVQA_dataset/annotations.json')
data = json.load(f)

results = []

writefile = args.writefile

for eval_datapoint in data:
    image_path = eval_datapoint['image']

    question = "Question: " + eval_datapoint['question'] + " Answer:"

    if (args.contextual):
        question = 'Assume someone is browsing a ' + eval_datapoint['context'] + ' website when they encounter this image. They cannot see the image directly, but they can access this image description: ' + eval_datapoint['description'] + ' Based on this description, they asked this follow-up question. Please answer based on the image. In your answer, prioritize details relevant to this person. ' + eval_datapoint['question']

    image = Image.open('../../' + image_path).convert("RGB")
    
    inputs = processor(images=image, text=question, return_tensors="pt").to(device="cuda", dtype=torch.float16)

    generated_ids = model.generate(**inputs,
            do_sample=False,
            max_length=256)

    generated_answer = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

    print("Question: ", question)
    print("Answer: ", generated_answer)

    results.append({
            'image': eval_datapoint['image'],
            'description': eval_datapoint['description'],
            'prompt': question,
            'context': eval_datapoint['context'],
            'question': eval_datapoint['question'],
            'generated_answer': generated_answer,
            'answers': eval_datapoint['answers']
        })

idx = list(range(0, len(results)))
df = pd.DataFrame(results, index=idx)
df.to_csv('results/' + args.writefile + '_results.csv')