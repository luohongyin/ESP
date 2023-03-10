# Logic against bias

Here we provide the evaluation code for stereotypical bias on different tasks, including
- StereoSet
- Profession bias test
- Emotion bias test

The scripts in this folder are
- `eval_stereo.py`: evaluation on StereoSet.
- `eval_gender.py`: evaluation on profession/emotion tests.

## Data and pretrained models
Download the `test.json` file from the `bias-bench` Github repo with this [link](https://github.com/McGill-NLP/bias-bench/blob/main/data/stereoset/test.json), and put the `test.json` file under `data/StereoSet/`.

All pretrained models are hosted on [Huggingface model hub](https://huggingface.co/luohy).

## Fairness experiments

### StereoSet

Run the following command to reproduce the experiments on StereoSet
```
python eval_stereo.py MODEL_TYPE EVAL_SPLIT CLS_HEAD MODEL_NAME_STR SCORE_TYPE
```
The parameters have the following valid values:
- `MODEL_TYPE`: 
    - `sc` for sequence classification (entailment models);
    - `nsp` for next sentence prediction (pretrained models)
- `EVAL_SPLIT`:
    - `intrasentence`
    - `intersentence`
- `CLS_HEAD`: the logit of the entailment model for final scoring
    - `0` for the entailment score
    - `1` for the neutral score
    - `2` for the contradictory score
    - Note - this parameter is ignored with `MODEL_TYPE = nsp` or `SCORE_TYPE = pred`
- `MODEL_NAME_STR`: the name of selected models
    - `bert, deberta, roberta` for both `sc` and `nsp`
    - `bert-simcse, bert-simcse-unsup, roberta-simcse, roberta-simcse-unsup` for `nsp` only.
- `SCORE_TYPE`: Specify scoring strategy
    - `score`: continous scoring
    - `pred`: discrete scoring
    - Note - this parameter is only used when `MODEL_TYPE = sc`

For example, the following command runs the intrasentence task with the `ESP-deberta-large` entailment model.
```
python eval_stereo.py sc intrasentence 0 deberta score
```

### Gender-profession-emotion test

Run the following command to reproduce the experiments on gender-profession-emotion test
```
python eval_gender.py MODEL_TYPE CLS_HEAD MODEL_NAME_STR SCORE_TYPE
```
The only difference is that we no longer specify the evaluation split here and the code will run both profession and emotion tests. For example,
```
python eval_gender.py nsp 0 roberta-simcse score
python eval_gender.py sc 0 roberta pred
```

## General language understanding tasks (GLUE)

We construct suppositions for each test sample and predict `True / Neutral / False` truth values of the suppositions. Here we list some example suppositions. Our `ESP` models can be zero-shot text classifiers, including
- `luohy/ESP-bert-large`
- `luohy/ESP-roberta-large`
- `luohy/ESP-deberta-large`

For classification, we associate the task-specific labels with the truth values of the constructed suppositions as follows.

- QNLI
```
The answer to {question} is entailed by {context}.
```
- QQP
```
The answer to {question 1} is entailed by the answer to {question 2}.
```
- RTE
```
{hypothesis} is entailed by {premise}
```
- CoLA
```
The sentence is not fluent is entailed by the sentence is "{sentence}".
```
- SST2
```
The movie is good is entailed by {movie review}.
```
Note that the the relation between the final predictions and truth values might vary. For example, one can use the following supposition for SST2
```
The movie is bad is entailed by {movie review}.
```
If the supposition is `True`, the review would be `negative`.


## Citation

If you used our code or model, please cite the paper. If there is any question, please feel free to contact the first author HL. Thank you!