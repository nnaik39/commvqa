# Model Experiments

This folder contains all the models and evaluations necessary to reproduce the results in our paper. To re-run for any model in particular, please navigate to the corresponding models' folder.

All post-processing and evaluation is done in the evalmetrics.py file.

To reproduce Table 1, please run:

```
python3 eval_metrics.py
```

For each model in the corresponding folder, to evaluate the model on the dataset with the baseline condition, run the following command:
```
python3 eval_{MODEL_NAME}.py --writefile {YOUR WRITEFILE HERE}
```

To evaluate each model with the contextual condition, run the following command:
```
python3 eval_{MODEL_NAME}.py --contextual --writefile {YOUR WRITEFILE HERE}
```
