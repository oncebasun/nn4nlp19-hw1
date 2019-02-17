# -*- coding: utf-8 -*-
import os
import time
import codecs
import pickle
import argparse
import torchtext.data as data

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred', '-p', type=str, required=True, help='load prediction val file')
    parser.add_argument('--datadir', '-d', type=str, default="../data", help='path to the data directory [default: ../data]')
    parser.add_argument('--text_vocab', type=str, required=True, help='path to training text vocabulary')
    parser.add_argument('--label_vocab', type=str, required=True, help='path to training label vocabulary')
    args = parser.parse_args()

    val_labels_out = []
    val_labels_trg = []
    val_sentence = []
    with codecs.open(os.path.join(args.datadir, 'topicclass_val.txt')) as f:
        for line in f:
            line = line.strip()
            label = line.split('|||')[0].strip()
            sentence = line.split('|||')[1].strip().split()
            sentence = [w.lower() for w in sentence]
            if len(line) > 0:
                val_labels_trg.append(label)
                val_sentence.append(sentence)

    with codecs.open(os.path.join(args.pred, 'predict_val.txt')) as f:
        for line in f:
            l = line.strip()
            if len(l) > 0:
                val_labels_out.append(l)

    text_field = data.Field(lower=True)
    label_field = data.Field(sequential=False)
    text_field.vocab = pickle.load(open(args.text_vocab, 'rb'))
    label_field.vocab = pickle.load(open(args.label_vocab, 'rb'))

    maxnum = 0

    for word in text_field.vocab.freqs:
        num = text_field.vocab.freqs[word]
        if maxnum < num:
            maxnum = num

    print(maxnum)
    
    all_count = [0] * (maxnum + 1)
    cor_set = set()
    cor_count = [0] * (maxnum + 1)
    err_set = set()
    err_count = [0] * (maxnum + 1)

    for word in text_field.vocab.freqs:
        num = text_field.vocab.freqs[word]
        all_count[num] += 1

    for i in range(len(val_labels_trg)):
        if val_labels_trg[i] == val_labels_out[i]:
            for w in val_sentence[i]:
                print(w, text_field.vocab.freqs[w.lower()])
                if w.lower() not in cor_set:
                    cor_set.add(w)
                    cor_count[text_field.vocab.freqs[w.lower()]] += 1
        else:
            for w in val_sentence[i]:
                print(w, text_field.vocab.freqs[w.lower()])
                if w.lower() not in err_set:
                    err_set.add(w)
                    err_count[text_field.vocab.freqs[w.lower()]] += 1

    segs = [0, 1, 2, 5, 20, 100, 1000, 10000, 100000, 1000000]

    print()
    print("all:")
    segcount = [0] * (len(segs) - 1)
    for i in range(1, len(segs)):
        for j in range(segs[i-1], min(segs[i], len(all_count))):
            segcount[i-1] += all_count[j]
    for i in range(1, len(segs)):
        print(segs[i-1], segs[i], "%.2f%%" %(100.0 * segcount[i-1] / sum(segcount)))

    print()
    print("correct:")
    segcount = [0] * (len(segs) - 1)
    for i in range(1, len(segs)):
        for j in range(segs[i-1], min(segs[i], len(all_count))):
            segcount[i-1] += cor_count[j]
    for i in range(1, len(segs)):
        print(segs[i-1], segs[i], "%.2f%%" %(100.0 * segcount[i-1] / sum(segcount)))

    print()
    print("error:")
    segcount = [0] * (len(segs) - 1)
    for i in range(1, len(segs)):
        for j in range(segs[i-1], min(segs[i], len(all_count))):
            segcount[i-1] += err_count[j]
    for i in range(1, len(segs)):
        print(segs[i-1], segs[i], "%.2f%%" %(100.0 * segcount[i-1] / sum(segcount)))

    print()
    labels = {}
    labelsum = 0

    for label in label_field.vocab.freqs:
        if label not in labels:
            labels[label] = [label_field.vocab.freqs[label], 0, 0, {}]
        labelsum += label_field.vocab.freqs[label]

    for i in range(len(val_labels_trg)):
        label_trg = val_labels_trg[i]
        label_out = val_labels_out[i]
        labels[label_trg][2] += 1
        if label_trg == label_out:
            labels[label_trg][1] += 1
        else:
            if label_out not in labels[label_trg][3]:
                labels[label_trg][3][label_out] = 0
            labels[label_trg][3][label_out] += 1
            
    for label in labels:
        mmax = 0
        mlab = ""
        for out in labels[label][3]:
            if labels[label][3][out] > mmax:
                mmax = labels[label][3][out] 
                mlab = out
        print("%s: %.2f%%, %d, %.2f%%, %s" %(label, (100.0*labels[label][0]/labelsum), labels[label][2], (100.0*labels[label][1]/(labels[label][2]+0.0000001)), mlab))

    for label in labels:
        print(label, labels[label][3])

    #print(label_field.vocab.freqs.keys())

    