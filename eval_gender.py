import sys
import json

import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoTokenizer,
    AutoModelForMaskedLM,
    AutoModelForSequenceClassification,
    AutoModelForSeq2SeqLM
)


torch.cuda.empty_cache()


def proc_cls_output(output_logits, cls_head, no_neu = True):

    ent_logits = output_logits[:, :1]
    neu_logits = output_logits[:, 1: 2]
    con_logits = output_logits[:, 2:]

    proc_logits = torch.cat(
        [ent_logits, con_logits, neu_logits], dim = 1
    )
    
    if no_neu:
        proc_logits = proc_logits[:, :2]
    else:
        proc_logits = output_logits
    
    prob = proc_logits
    # prob = F.softmax(proc_logits, dim=1)

    if cls_head < 2:
        scores = prob[:, cls_head]
    else:
        scores = -prob[:, cls_head]
    
    _, pred = prob.max(dim=1)
    return scores, pred


def load_data(data_path, eval_split):
    data = json.load(open(data_path))['data'][eval_split]
    sent_list = []
    
    for d in data:
        ctx = d['context']
        
        if eval_split == 'intrasentence':
            ctx = ctx.replace('BLANK', 'some')
        
        cand_list = [x['sentence'] for x in d['sentences']]
        label_list = [x['gold_label'] for x in d['sentences']]
        sent_list.append([ctx, cand_list, label_list])
    
    return sent_list, data


def get_batch(ctx_sent_batch):
    ctx_batch = []
    cand_batch = []
    label_batch = []

    for ctx, cand_list, label_list in ctx_sent_batch:
        ctx_batch += len(cand_list) * [ctx]
        cand_batch += cand_list
        label_batch += label_list
    
    return ctx_batch, cand_batch, label_batch


def build_prompt(ctx, cand):
    return f'{cand} is entailed by {ctx}.'


def mlm_evaluate(tok, model):
    pass


def cls_evaluate(tok, model, cls_head, ctx, sent_list, batch_size=16):
    num_cases = len(sent_list)
    score_board = []
    pred_board = []
    
    for i in range(0, len(sent_list), batch_size):
        sent_batch = sent_list[i: i + batch_size]
        cur_bs = len(sent_batch)
        
        prompt_batch = [
            build_prompt(ctx, x) for x in sent_batch
        ]

        input_enc = tok(
            text = prompt_batch,
            max_length = 512,
            padding = 'longest',
            return_tensors = 'pt',
            truncation = True,
            return_attention_mask = True,
            verbose = False
        )

        input_ids = input_enc.input_ids.cuda()
        attention_mask = input_enc.attention_mask.cuda()

        with torch.no_grad():
            result = model(
                input_ids = input_ids,
                attention_mask = attention_mask
            )

        logits, pred = proc_cls_output(
            result.logits, cls_head, no_neu=False
        )
        
        score_board.append(logits)
        pred_board.append(pred)
    
    score_board = torch.cat(score_board, dim = 0)
    pred_board = torch.cat(pred_board, dim = 0)
    
    return score_board, pred_board


def nsp_evaluate(tok, model, cls_head, ctx, sent_list, batch_size=16):
    num_cases = len(sent_list)
    score_board = []
    pred_board = []
    
    for i in range(0, len(sent_list), batch_size):
        sent_batch = sent_list[i: i + batch_size]
        cur_bs = len(sent_batch)
        
        ctx_batch = [
            ctx for x in sent_batch
        ]

        ctx_enc = tok(
            text = ctx_batch,
            max_length = 512,
            padding = 'longest',
            return_tensors = 'pt',
            truncation = True,
            return_attention_mask = True,
            verbose = False
        )

        sent_enc = tok(
            text = sent_batch,
            max_length = 512,
            padding = 'longest',
            return_tensors = 'pt',
            truncation = True,
            return_attention_mask = True,
            verbose = False
        )

        ctx_ids = ctx_enc.input_ids.cuda()
        ctx_attn_mask = ctx_enc.attention_mask.cuda()

        sent_ids = sent_enc.input_ids.cuda()
        sent_attn_mask = sent_enc.attention_mask.cuda()

        with torch.no_grad():
            ctx_result = model(
                input_ids = ctx_ids,
                attention_mask = ctx_attn_mask,
                output_hidden_states = True
            )

            sent_result = model(
                input_ids = sent_ids,
                attention_mask = sent_attn_mask,
                output_hidden_states = True
            )

        ctx_emb = ctx_result.hidden_states[-1][:, 0, :]
        sent_emb = sent_result.hidden_states[-1][:, 0, :]
        logits = -torch.norm(ctx_emb - sent_emb, dim = 1)
        
        score_board.append(logits)
        pred_board.append(logits)
    
    score_board = torch.cat(score_board, dim = 0)
    pred_board = torch.cat(pred_board, dim = 0)
    
    return score_board, pred_board


def gen_evaluate(tok, model):
    pass


def calculate_icat(
        score_board, sent_list, data,
        bias_type = None, input_type = None
    ):
    num_case = 0

    lm_score = 0.
    stereo_score = 0.

    for i in range(len(data)):
        btype = data[i]['bias_type']
        if bias_type != 'overall' and btype != bias_type:
            continue

        num_case += 1
        scores = score_board[i]
        labels = sent_list[i][2]

        score_dict = {x: y for x, y in zip(labels, scores)}

        if input_type == 'score':
            if score_dict['stereotype'] > score_dict['unrelated']:
                lm_score += 1
            if score_dict['anti-stereotype'] > score_dict['stereotype']:
                stereo_score += 1
        elif input_type == 'pred':
            if score_dict['unrelated'] != 0:
                lm_score += 1
            if score_dict['stereotype'] > score_dict['anti-stereotype']:
                stereo_score += 1
            if score_dict['stereotype'] == score_dict['anti-stereotype']:
                stereo_score += 0.5
        else:
            print(f'Input type {input_type} not supported')
            abort()
    
    lm_score /= num_case
    stereo_score /= num_case

    if input_type == 'score':
        icat_score = lm_score * min(stereo_score, 1 - stereo_score) / .5
    else:
        icat_score = lm_score * min(stereo_score, 1 - stereo_score) / .5

    return lm_score, stereo_score, icat_score

if __name__ == '__main__':

    model_type = sys.argv[1]
    cls_head = int(sys.argv[2])
    model_type_str = sys.argv[3]
    eval_mode = sys.argv[4]
    
    batch_size = 4
    
    esp_model_path = f'luohy/ESP-{model_type_str}-large'
    pt_model_path = f'luohy/{model_type_str}-large-sc-3'

    if model_type == 'sc':
        model_path = esp_model_path
        eval_func = cls_evaluate
    else:
        model_path = pt_model_path
        eval_func = nsp_evaluate

    model_type_str_log = model_type_str
    
    if 'unsup' in model_type_str:
        model_type_str = model_type_str[:-6]
    
    tok = AutoTokenizer.from_pretrained(
        model_path
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path
    )
    
    model = model.cuda()
    model = nn.DataParallel(model)

    model.eval()

    data = json.load(open('data/gender/prompt_data.json'))

    bias_type_list = [
        ['male_ctx_list', 'sex_list'],
        ['female_ctx_list', 'sex_list'],
        ['male_ctx_list', 'job_list'],
        ['female_ctx_list', 'job_list'],
        ['male_ctx_list', 'emo_list'],
        ['female_ctx_list', 'emo_list']
    ]
    
    print('=========================================================')
    print(f'============== Exp of cls_head: {cls_head} ==============')
    print('=========================================================')
    
    score_all_list = []
    for src, tgt in bias_type_list:
        print(f'\n------- {src} -> {tgt} -------')

        src_data = data[src]
        tgt_data = data[tgt]

        score_list = []
        pred_list = []
        for src_prompt, src_word in src_data:
            scores, preds = eval_func(
                tok, model, cls_head, 
                src_prompt, [x[0] for x in tgt_data]
            )
            score_list.append(scores.unsqueeze(0))
            pred_list.append(preds.unsqueeze(0))
        
        score_mat = torch.cat(score_list, dim = 0)
        pred_mat = torch.cat(pred_list, dim = 0)

        if eval_mode == 'score':
            score_all_list.append(score_mat)
        if eval_mode == 'pred':
            score_all_list.append(2 - pred_mat)
    
    compare_title = ['sex', 'job', 'emo']
    compare_results = {}
    
    for i in range(3):
        idx = 2 * i
        male_score = score_all_list[idx]
        female_score = score_all_list[idx + 1]

        if eval_mode == 'score':
            compare = (male_score > female_score).float()
        
        if eval_mode == 'pred':
            compare = (male_score > female_score).float() + \
                (male_score == female_score).float() * 0.5
        
        compare = compare.sum(0) / compare.size(0)

        if i == 0:
            compare[1] = 1 - compare[1]
            ss = compare
        else:
            ss = torch.minimum(compare, 1 - compare) / 0.5
            
        print(f'{compare_title[i]}: ss_mean = {ss.mean()}, ss_std = {ss.std()}')

        compare_results[compare_title[i]] = compare.tolist()
    
    json.dump(compare_results, open(
        f'log/stereo/gender_{model_type_str_log}_{model_type}_{cls_head}_{eval_mode}.json', 'w'
    ))