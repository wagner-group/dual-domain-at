import os
import time
import numpy as np


def evaluate(config, wrapper_model, data_loader, device, lg, mode):
    # Initialize metrics
    num_examples = 0
    cl_loss, cl_correct = 0, 0
    adv_loss, adv_correct = 0, 0

    # Accumulate loss/accuracy over batches
    for _, (x, y) in enumerate(data_loader):
        x, y = x.to(device), y.to(device)

        # Get model output
        wrapper_model.eval()
        output = wrapper_model(x, y)

        cl_logits, adv_logits = output[0], output[1]
        cl_loss_batch, adv_loss_batch = output[2], output[3]

        # Aggregate clean loss/accuracy
        num_examples += x.shape[0]
        cl_loss += cl_loss_batch.item()
        _, cl_logits_dig = cl_logits.max(1)
        cl_correct += cl_logits_dig.eq(y).float().sum().item()

        # Aggregate adversarial loss/accuracy
        adv_loss += adv_loss_batch.item()
        _, adv_logits_dig = adv_logits.max(1)
        adv_correct += adv_logits_dig.eq(y).float().sum().item()

        if num_examples >= config['test']['num_sample']:
            break

    # Compute average loss/accuracy
    cl_loss, cl_acc = cl_loss/num_examples, cl_correct/num_examples
    adv_loss, adv_acc = adv_loss/num_examples, adv_correct/num_examples

    # Print and log loss/accuracy metrics
    lg.print(f'Clean {mode} loss: {cl_loss:.4f}, accuracy: {cl_acc * 100:.2f}')
    lg.print(f'Advers {mode} loss: {adv_loss:.4f}, accuracy: {adv_acc * 100:.2f}')
    return adv_acc
