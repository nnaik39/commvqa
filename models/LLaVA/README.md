This folder is adapted from the LLaVA repo. Please remember to adjust the file paths to point to the downloaded dataset within your directory path.

To evaluate LLaVA on the CommVQA dataset with the baseline condition, run the following command:
```
python3 eval_llava.py --writefile {YOUR WRITEFILE HERE}
```

To evaluate LLaVA with the contextual condition, run the following command:
```
python3 eval_llava.py --context_description --writefile {YOUR WRITEFILE HERE}
```
