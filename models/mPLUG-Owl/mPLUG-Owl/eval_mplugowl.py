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

bertscore = load("bertscore")
pretrained_ckpt = 'MAGAer13/mplug-owl-llama-7b'

model = MplugOwlForConditionalGeneration.from_pretrained(
    pretrained_ckpt,
    torch_dtype=torch.bfloat16,
)
image_processor = MplugOwlImageProcessor.from_pretrained(pretrained_ckpt)
tokenizer = MplugOwlTokenizer.from_pretrained(pretrained_ckpt)
processor = MplugOwlProcessor(image_processor, tokenizer)

f = open('../../dataset_annotations_cleaned.json')
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

images_for_clipscore = []

for eval_datapoint in tqdm(data):
    #print("Datapoint ", eval_datapoint)
    image_path = '../../' + eval_datapoint['image']
    images_for_clipscore.append(image_path)

    question = eval_datapoint['question']

    #print("Question: ", question)

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

print("Length of images ", len(images_for_clipscore))

results_per_context = {}
refs_per_context = {}
hyps_per_context = {}
images_per_context = {}

for datapoint in results:
    if (datapoint['context'] not in refs_per_context):
        refs_per_context[datapoint['context']] = []
        hyps_per_context[datapoint['context']] = []
        images_per_context[datapoint['context']] = []

    hyps_per_context[datapoint['context']].append(datapoint['generated_answer'].lower().strip())
    eval_datapoint = datapoint

    images_per_context[datapoint['context']].append('../../' + datapoint['image'])
    if (len(datapoint['answers']) == 3):
        refs_per_context[datapoint['context']].append([eval_datapoint['answers'][0].lower().strip(), eval_datapoint['answers'][1].lower().strip(), eval_datapoint['answers'][2].lower().strip()])
    else:
        refs_per_context[datapoint['context']].append([eval_datapoint['answers'][0].lower().strip(), eval_datapoint['answers'][1].lower().strip()])

metrics_per_context = {}

device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, transform = clip.load("ViT-B/32", device=device, jit=False)
clip_model.eval()
model, transform = clip.load("ViT-B/32", device=device, jit=False)
model.eval()
if (args.per_context_scores):
    for context in refs_per_context:
        with contextlib.redirect_stdout(None):
            metrics_per_context[context] = get_all_metrics(refs_per_context[context], hyps_per_context[context])

        _, per_instance_image_text, candidate_feats = get_clip_score(model, images_per_context[context], hyps_per_context[context], device)

        _, per_instance_text_text = get_refonlyclipscore(
            model, refs_per_context[context], candidate_feats, device)
# F-score
        refclipscores = 2 * per_instance_image_text * per_instance_text_text / (per_instance_image_text + per_instance_text_text)

        scores = {image_id: {'CLIPScore': float(clipscore), 'RefCLIPScore': float(refclipscore)}
            for image_id, clipscore, refclipscore in
            zip(images_per_context[context], per_instance_image_text, refclipscores)}

        metrics_per_context[context]['clipscore'] = np.mean([s['CLIPScore'] for s in scores.values()])
        metrics_per_context[context]['clipscore_std'] = np.std([s['CLIPScore'] for s in scores.values()])

        metrics_per_context[context]['ref_clipscore'] = np.mean([s['RefCLIPScore'] for s in scores.values()])
        metrics_per_context[context]['ref_clipscore_std'] = np.std([s['RefCLIPScore'] for s in scores.values()])
#scores = {image_id: {'CLIPScore': float(clipscore)}
 #                   for image_id, clipscore in zip(images, per_instance_image_text)}

    print("Metrics per context ", metrics_per_context)

    with open(args.writefile + '_percontext_metrics.json', 'w') as fp:
        json.dump(metrics_per_context, fp)

_, per_instance_image_text, candidate_feats = get_clip_score(model, images_for_clipscore, hyps, device)

_, per_instance_text_text = get_refonlyclipscore(
            model, refs, candidate_feats, device)
# F-score
refclipscores = 2 * per_instance_image_text * per_instance_text_text / (per_instance_image_text + per_instance_text_text)

scores = {image_id: {'CLIPScore': float(clipscore), 'RefCLIPScore': float(refclipscore)}
            for image_id, clipscore, refclipscore in
            zip(images_for_clipscore, per_instance_image_text, refclipscores)}

with contextlib.redirect_stdout(None):
    metrics_dict = get_all_metrics(refs, hyps)
clipscore = np.mean([s['CLIPScore'] for s in scores.values()])
clipscore_std = np.std([s['CLIPScore'] for s in scores.values()])
ref_clipscore = np.mean([s['RefCLIPScore'] for s in scores.values()])
ref_clipscore_std = np.std([s['RefCLIPScore'] for s in scores.values()])
print("CLIPScore: ", clipscore)
print("CLIPScore standard deviation: ", clipscore_std)

metrics_dict['clipscore'] = clipscore
metrics_dict['clipscore_std_dev'] = clipscore_std
metrics_dict['ref_clipscore'] = ref_clipscore
metrics_dict['ref_clipscore_std_dev'] = ref_clipscore_std

metrics_dict['full_clipscore'] = [s['CLIPScore'] for s in scores.values()]
metrics_dict['full_ref_clipscore'] = [s['RefCLIPScore'] for s in scores.values()]

bertscore_results = bertscore.compute(predictions=hyps, references=refs, lang="en")

metrics_dict['bertscore_avg_precision'] = np.mean(bertscore_results['precision'])
metrics_dict['bertscore_avg_recall'] = np.mean(bertscore_results['recall'])
metrics_dict['bertscore_avg_f1'] = np.mean(bertscore_results['f1'])

#metrics_dict['full_bertscore'] = bertscore_results
print("Metrics dict for mPLUG-OWL evaluation: ", metrics_dict)

print("Length of images here ", len(images_for_clipscore))
idx = list(range(0, len(results)))
df = pd.DataFrame(results, index=idx)

df.to_csv(args.writefile + '_results.csv')

with open(args.writefile + '_metrics.json', 'w') as fp:
        json.dump(metrics_dict, fp)
