from string import Template
import numpy as np
import torch
from collections import Counter
import torch.nn.functional as F
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, classification_report
from transformers import TrainerCallback
import json
import os


class SaveMetricsPerEpoch(TrainerCallback):

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        epoch_num = int(state.epoch)
        json_path = os.path.join(args.output_dir, f"metrics_epoch_{epoch_num}.json")

        with open(json_path, "w") as f:
            json.dump(metrics, f, indent=4)

        print(f"Epoch {epoch_num}, Saved metrics to {json_path}")


def score_choice(choice, model, tokenizer):
    model.eval()
    device = model.device

    with torch.no_grad():
        q, a = choice.split(' <sep> ')
        q, a = q.strip(), a.strip()

        enc = tokenizer(q, a, padding='max_length', truncation=True, max_length=128, return_tensors='pt')

        input_ids, attn_mask = enc.input_ids.to(device), enc.attention_mask.to(device)

        is_answer, answer_pos = False, []
        for idx, input_id in enumerate(input_ids[0]):
            # first [SEP]
            if not is_answer and input_id.item() == tokenizer.sep_token_id:
                is_answer = True
                continue
            # final [SEP]
            elif is_answer and input_id.item() == tokenizer.sep_token_id:
                break
            # answer is in-between [SEP] tokens
            if is_answer:
                answer_pos.append(idx)

        batch_input_ids, batch_attn_mask, target_token_ids = [], [], []
        for idx in answer_pos:
            token_id_original = input_ids[0, idx].item()

            masked = input_ids.clone()
            masked[0, idx] = tokenizer.mask_token_id

            batch_input_ids.append(masked[0])
            batch_attn_mask.append(attn_mask[0])
            target_token_ids.append(token_id_original)

        batch_input_ids = torch.stack(batch_input_ids).to(device)
        batch_attn_mask = torch.stack(batch_attn_mask).to(device)
        target_token_ids = torch.tensor(target_token_ids).to(device)

        outputs = model(input_ids=batch_input_ids, attention_mask=batch_attn_mask)

        logits = outputs.logits
        log_probs = F.log_softmax(logits, dim=-1)

        token_logprobs = []

        for i, idx in enumerate(answer_pos):
            token_logprob = log_probs[i, idx, target_token_ids[i]].item()
            token_logprobs.append(token_logprob)

        logprob = sum(token_logprobs) / len(token_logprobs)

        return logprob, np.exp(logprob)


class EvalQA(object):
    def __init__(self, test_path, tokenizer):
        self.trainer = None
        self.tokenizer = tokenizer

        with open(test_path, 'r') as f:
            self.test_data = json.load(f)

    def on_epoch_end(self, model):

        prompt_template = Template('$question <sep> $answer')
        Y, Y_hat = [], []
        ## this can also be random, but then is harder to assess whether its effective
        test_data_sample = self.test_data[: 500]
        print(f"\n\n*** Evaluate QA ***\nNum Questions : {len(test_data_sample)}\n\n")
        for test_example in test_data_sample:
            choices = [
                prompt_template.substitute(question=test_example['question'], answer=test_example[f'op{clet}']) for
                clet
                in ['a', 'b', 'c', 'd']]

            cop = test_example['cop']
            max_prob, ans = float('-inf'), None
            for cop_idx in range(len(choices)):
                logs, _ = score_choice(choices[cop_idx], model, self.tokenizer)
                if max_prob < logs:
                    max_prob = logs
                    ans = cop_idx + 1
            Y.append(cop)
            Y_hat.append(ans)

        labels = [1, 2, 3, 4]

        p_micro = precision_score(Y, Y_hat, average='micro', zero_division=0)
        r_micro = recall_score(Y, Y_hat, average='micro', zero_division=0)
        f1_micro = f1_score(Y, Y_hat, average='micro', zero_division=0)

        p_macro = precision_score(Y, Y_hat, average='macro', zero_division=0, labels=labels)
        r_macro = recall_score(Y, Y_hat, average='macro', zero_division=0, labels=labels)
        f1_macro = f1_score(Y, Y_hat, average='macro', zero_division=0, labels=labels)

        acc = accuracy_score(Y, Y_hat)

        report = classification_report(Y, Y_hat, zero_division=0, labels=labels,
                                       target_names=["opa", "opb", "opc", "opd"])

        print(f'Gold Ans: {Counter(Y)}')
        print(f'Pred Ans: {Counter(Y_hat)}')

        print(report)

        d = {'P-micro': p_micro, 'R-micro': r_micro, 'F1-micro': f1_micro, 'P-macro': p_macro, 'R-macro': r_macro,
             'F1-macro': f1_macro, 'Acc': acc}

        return d
