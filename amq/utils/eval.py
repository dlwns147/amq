from tqdm import tqdm

import torch
import torch.nn as nn

from utils.loss import JSD

@torch.inference_mode()
def get_logits(model, loader):
    logits = []
    for inputs in tqdm(loader, desc='Get Logits'):
        outputs = model(inputs)
        lm_logits = outputs.logits
        logits.append(lm_logits)

    dense_logits_list = torch.cat(logits, dim=0).detach()

    return dense_logits_list


@torch.inference_mode()
def eval_loss(model, accelerator, loader, dense_logits_list, seqlen=2048):
    losses = []
    
    for i, inputs in enumerate(loader):
        # Forward pass through the model
        outputs = model(inputs)
        lm_logits = outputs.logits

        # Shift logits and labels for next token prediction
        shift_logits = lm_logits[:, :-1, :].reshape(-1, lm_logits.size(-1)).contiguous()
        
        # Compute loss
        dense_logits = dense_logits_list[i]
        dense_logits = dense_logits[:-1, :].reshape(-1, lm_logits.size(-1)).contiguous()
        loss_fct = JSD()
        loss = loss_fct(shift_logits, dense_logits)

        # Calculate negative log likelihood
        loss = loss.float() * seqlen * lm_logits.shape[0]
        losses.append(loss)

    losses = torch.stack(accelerator.gather_for_metrics(losses)).flatten()
    loss_sum = losses.sum() / (len(losses) * seqlen)

    return loss_sum.item()


@torch.inference_mode()
def eval_ppl(model, accelerator, loader, seqlen=2048):
    ppls = []

    for i, inputs in enumerate(loader):
        outputs = model(inputs)
        lm_logits = outputs.logits

        # Shift logits and labels for next token prediction
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_logits = shift_logits.reshape(-1, shift_logits.size(-1))
        shift_labels = inputs[:, 1:].reshape(-1)

        # Compute loss
        loss_fct = torch.nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits, shift_labels)

        # Calculate negative log likelihood
        neg_log_likelihood = loss.float() * seqlen * lm_logits.shape[0]

        # Append to list of negative log likelihoods
        ppls.append(neg_log_likelihood)

    ppls = torch.stack(accelerator.gather_for_metrics(ppls)).flatten()
    ppl = torch.exp(ppls.sum() / (len(ppls) * seqlen))
    
    return ppl.item()