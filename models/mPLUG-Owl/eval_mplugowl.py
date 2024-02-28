from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration
from PIL import Image
from clipscore_helper import get_all_metrics, get_clip_score, get_refonlyclipscore
import contextlib
import pandas as pd
from mplug_owl.modeling_mplug_owl import MplugOwlForConditionalGeneration
from mplug_owl.tokenization_mplug_owl import MplugOwlTokenizer
from mplug_owl.processing_mplug_owl import MplugOwlImageProcessor, MplugOwlProcessor
import torch
import json
from evaluate import load
import requests
import numpy as np
import argparse
import clip
from tqdm import tqdm

pretrained_ckpt = 'MAGAer13/mplug-owl-llama-7b'

model = MplugOwlForConditionalGeneration.from_pretrained(
    pretrained_ckpt,
    torch_dtype=torch.bfloat16,
)
image_processor = MplugOwlImageProcessor.from_pretrained(pretrained_ckpt)
tokenizer = MplugOwlTokenizer.from_pretrained(pretrained_ckpt)
processor = MplugOwlProcessor(image_processor, tokenizer)

f = open('../../annotations.json')
data = json.load(f)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

parser = argparse.ArgumentParser(description='Optional app description')

parser.add_argument('--writefile', type=str,
                    help='A required integer positional argument',
                    required=True)

parser.add_argument('--generation_length', type=int,
                    help='A required integer positional argument',
                    default=512)
parser.add_argument('--per_context_scores', action='store_true')
parser.add_argument('--context_description', action='store_true')

args = parser.parse_args()

results = []

refs = []
hyps = []

for eval_datapoint in tqdm(data):
    image_path = '../../' + eval_datapoint['image']
    images_for_clipscore.append(image_path)

    question = eval_datapoint['question']

    if (args.context_description):
        question = 'Assume someone is browsing a ' + eval_datapoint['context'] + ' website when they encounter this image. They cannot see the image directly, but they can access this image description: ' + eval_datapoint['description'] + ' Based on this description, they asked this follow-up question. Please answer based on the image. In your answer, prioritize details relevant to this person. Question: ' + eval_datapoint['question']

    prompts = [
        '''The following is a conversation between a curious human and AI assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. \n Human: <image>
Human: ''' + question]
    prompts[0] += '\nAI:'

    image_list = [image_path]
    
    generate_kwargs = {
    'do_sample': False,
    'max_length': 256
    }

    images = [Image.open(_) for _ in image_list]
    inputs = processor(text=prompts, images=images, return_tensors='pt')
    inputs = {k: v.bfloat16() if v.dtype == torch.float else v for k, v in inputs.items()}
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        res = model.generate(**inputs, **generate_kwargs)
    generated_answer = tokenizer.decode(res.tolist()[0], skip_special_tokens=True)

    results.append({
            'image': eval_datapoint['image'],
            'description': eval_datapoint['description'],
            'context': eval_datapoint['context'],
            'question': eval_datapoint['question'],
            'prompt': question,
            'generated_answer': generated_answer,
            'answers': eval_datapoint['answers']
        })

    if (len(eval_datapoint['answers']) == 3):
        refs.append([eval_datapoint['answers'][0].lower().strip(), eval_datapoint['answers'][1].lower().strip(), eval_datapoint['answers'][2].lower().strip()])
    else:
        refs.append([eval_datapoint['answers'][0].lower().strip(), eval_datapoint['answers'][1].lower().strip()])
    hyps.append(generated_answer.lower())

idx = list(range(0, len(results)))
df = pd.DataFrame(results, index=idx)

df.to_csv(args.writefile + '_results.csv')
