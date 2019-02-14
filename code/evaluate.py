import codecs
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
import logging


def f1(scores, target):
    scores = scores.cpu()
    target = target.cpu()

    predicted_labels = (scores > 0).long()
    predicted_correct_labels_sum = (predicted_labels * target).sum().data.numpy()
    #print target
    correct_labels_sum = target.sum().data.numpy()
    predicted_labels_sum = predicted_labels.sum().data.numpy()
    f1 = 2 * float(predicted_correct_labels_sum) / float(correct_labels_sum + predicted_labels_sum)
    #print predicted_correct_labels_sum, correct_labels_sum, predicted_labels_sum
    return f1


def map(scores, target):
    scores = scores.cpu().data.numpy()
    target = target.cpu().data.numpy()
    sort_idx_scores = np.argsort(-scores)
    idx_target = np.where(target >= 1 - 0.000001)[0]
    rank = 1
    index = 0
    ap = 0
    for item in list(sort_idx_scores):
        if item in list(idx_target):
            index += 1
            ap += float(index) / rank
        rank += 1
    ap /= len(idx_target)
    return ap


def acc(scores, target):
    scores = scores.cpu().data.numpy()
    target = target.cpu().data.numpy()
    idx_pred = np.argmax(scores)
    if idx_pred == target:
        return 1.0
    else:
        return 0.0


def test(cnn, test_iter, text_field, label_field, method='acc', cuda=False, verbose=True):
    logger = logging.getLogger()
    cnn.eval()
    if cuda:
        cnn.cuda()
    logger.info('Evaluation method: ' + method)
    ins_test = globals()[method]
    all_acc = 0
    ss = 0
    for batch in test_iter:
        input, target = batch.text, batch.label
        criterion = nn.CrossEntropyLoss()
        # Setup data
        input.data.t_()
        if cuda:
            input = input.cuda()
            target = target.cuda()
            criterion.cuda()
        scores = cnn(input, test=True)
        loss = criterion(scores, target)

        for i in range(input.size(0)):
            ss += 1
            if verbose:
                logger.info('Test sample: ' + str(ss))
                logger.info('Input:')
                logger.info(' '.join([text_field.vocab.itos[id.item()] if text_field.vocab.itos[id.item()]!='<pad>' else '' for id in input[i]]))
                logger.info('Target:')
                logger.info(label_field.vocab.itos[target[i].item() + 1])
                logger.info('Output:')
                logger.info(label_field.vocab.itos[np.argmax(scores[i].cpu().data.numpy()) + 1])
            acc = ins_test(scores[i], target[i])
            if verbose:
                logger.info('Acc = ' + str(acc))
                logger.info('')
            all_acc += acc
    logger.info('Average Acc = ' + str(all_acc / ss))
    return all_acc / ss


def predict(cnn, test_iter, text_field, label_field, output_file, cuda=False, verbose=True):
    logger = logging.getLogger()
    cnn.eval()
    if cuda:
        cnn.cuda()
    ss = 0
    for batch in test_iter:
        input, target = batch.text, batch.label
        # Setup data
        input.data.t_()
        if cuda:
            input = input.cuda()
        scores = cnn(input, test=True)

        with codecs.open(output_file, 'w') as f:
            for i in range(input.size(0)):
                ss += 1
                if verbose:
                    logger.info('Test sample: ' + str(ss))
                    logger.info('Input:')
                    logger.info(' '.join([text_field.vocab.itos[id.item()] if text_field.vocab.itos[id.item()]!='<pad>' else '' for id in input[i]]))
                    logger.info('Output:')
                    predict_label = label_field.vocab.itos[np.argmax(scores[i].cpu().data.numpy()) + 1]
                    f.write(predict_label + '\n')
                    logger.info(predict_label)
                    logger.info('')
