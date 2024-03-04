import torch
import pandas as pd
import json
from transformers import IdeficsForVisionText2Text, AutoProcessor
from PIL import Image
import argparse

device = "cuda" if torch.cuda.is_available() else "cpu"

f = open('../../CommVQA_dataset/annotations.json')
data = json.load(f)

results = []

parser = argparse.ArgumentParser(description='Optional app description')
parser.add_argument('--writefile', type=str,
                    required=True)
parser.add_argument('--contextual', action='store_true')

args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"

checkpoint = "HuggingFaceM4/idefics-9b-instruct"
model = IdeficsForVisionText2Text.from_pretrained(checkpoint, torch_dtype=torch.bfloat16).to(device)
processor = AutoProcessor.from_pretrained(checkpoint)

formatted_contexts = {'social_media': 'social media', 'science_journals': 'science magazine'}

for eval_datapoint in data:
    image_path = eval_datapoint['image']
    question = eval_datapoint['question']
    context = eval_datapoint['context']

    if (eval_datapoint['context'] in formatted_contexts):
        context = formatted_contexts[eval_datapoint['context']]

    if (args.contextual):
        question = 'Assume someone is browsing a ' + context + ' website when they encounter this image. They cannot see the image directly, but they can access this image description: ' + eval_datapoint['description'] + ' Based on this description, they asked this follow-up question. Please answer based on the image. In your answer, prioritize details relevant to this person. Question: ' + eval_datapoint['question']

    image = Image.open('../' + image_path).convert("RGB")

    prompts = [
    [
        "User: " + question,
        image,
        "<end_of_utterance>",
        "\nAssistant: ",
    ],
    ]

    inputs = processor(prompts, add_end_of_utterance_token=False, return_tensors="pt").to(device)
    exit_condition = processor.tokenizer("<end_of_utterance>", add_special_tokens=False).input_ids
    bad_words_ids = processor.tokenizer(["<image>", "<fake_token_around_image>"], add_special_tokens=False).input_ids

    generated_ids = model.generate(**inputs,
        eos_token_id=exit_condition,
        bad_words_ids=bad_words_ids,
        max_new_tokens=256,
        do_sample=False,
        num_beams=1)
    
    generated_answer = processor.batch_decode(generated_ids, skip_special_tokens=True)    
    generated_answer[0] = generated_answer[0].replace(prompts[0][0], '')
    generated_answer[0] = generated_answer[0].replace(' \nAssistant: ', '')
    generated_answer[0] = generated_answer[0].replace('Assistant:', '').strip()

    print("Question: ", question)
    print("Answer: ", generated_answer)

    results.append({
            'image': eval_datapoint['image'],
            'description': eval_datapoint['description'],
            'context': eval_datapoint['context'],
            'prompt': prompts[0][0],
            'question': eval_datapoint['question'],
            'answers': eval_datapoint['answers'],
            'generated_answer': generated_answer[0]
            })

folder_path = "results/" + args.writefile

idx = list(range(0, len(results)))
df = pd.DataFrame(results, index=idx)
df.to_csv(folder_path + '_results.csv')