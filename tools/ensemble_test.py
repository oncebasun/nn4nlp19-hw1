# -*- coding: UTF-8 -*-
import os
import time
import codecs
import pickle
import argparse
import torchtext.data as data


def test(label_out, label_trg):
    assert len(label_out) == len(label_trg)
    acc = 0
    for i in range(len(label_out)):
        if label_out[i] == label_trg[i]:
            acc += 1
    acc /= float(len(label_out))
    return acc

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred', '-p', type=str, required=True, help='load prediction dirs')
    parser.add_argument('--datadir', '-d', type=str, default="../data", help='path to the data directory [default: ../data]')
    parser.add_argument('--out', '-o', type=str, help='output file')
    args = parser.parse_args()

    timestamp = time.strftime('%Y_%m_%d_%H_%M_%S')
    data_ciph = args.datadir.replace('.', '_').replace('/', '-')
    output_path = os.path.join('../ensemble_results/', timestamp + data_ciph)
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    test_labels_outs = []
    val_labels_outs = []
    val_labels_trg = []
    with codecs.open(os.path.join(args.datadir, 'topicclass_val.txt')) as f:
        for line in f:
            line = line.strip()
            if len(line) > 0:
                val_labels_trg.append(line.split('|||')[0].strip())
    
    args.pred = args.pred.split(',')
    for p in args.pred:
        val_labels_outs.append([])
        test_labels_outs.append([])
        with codecs.open(os.path.join(p, 'predict_val.txt')) as f:
            for line in f:
                l = line.strip()
                if len(l) > 0:
                    val_labels_outs[-1].append(l)
        with codecs.open(os.path.join(p, 'predict_test.txt')) as f:
            for line in f:
                l = line.strip()
                if len(l) > 0:
                    test_labels_outs[-1].append(l)

    model_acc = []
    val_num = len(val_labels_outs[-1])
    for i in range(len(val_labels_outs)):
        out = val_labels_outs[i]
        acc = test(out, val_labels_trg)
        model_acc.append(acc)
        print("model %d: %f" %(i,acc))

    val_ensemble_out = []
    for j in range(val_num):
        cnt = {}
        for i in range(len(val_labels_outs)):
            if val_labels_outs[i][j] not in cnt:
                cnt[val_labels_outs[i][j]] = 0
            cnt[val_labels_outs[i][j]] += model_acc[i]
        max_key = max(cnt, key=lambda k: cnt[k])
        val_ensemble_out.append(max_key)

    test_ensemble_out = []
    test_num = len(test_labels_outs[-1])
    for j in range(test_num):
        cnt = {}
        for i in range(len(test_labels_outs)):
            if test_labels_outs[i][j] not in cnt:
                cnt[test_labels_outs[i][j]] = 0
            cnt[test_labels_outs[i][j]] += model_acc[i]
        max_key = max(cnt, key=lambda k: cnt[k])
        test_ensemble_out.append(max_key)

    print("Ensemble val: %f" %test(val_ensemble_out, val_labels_trg))

    with codecs.open(os.path.join(output_path, 'predict_val.txt'), 'w') as f:
        for label in val_ensemble_out:
            f.write(label + '\n')

    with codecs.open(os.path.join(output_path, 'predict_test.txt'), 'w') as f:
        for label in test_ensemble_out:
            f.write(label + '\n')

    
        

