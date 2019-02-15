# -*- coding: UTF-8 -*-
import os
import codecs
import pickle
import argparse
import torchtext.data as data

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--vocab', '-v', type=str, required=True, help='vocabulary pkl file')
    parser.add_argument('--emb', '-e', type=str, required=True, help='embedding file')
    parser.add_argument('--out', '-o', type=str, help='output file')
    args = parser.parse_args()

    text_field = data.Field(lower=True)
    text_field.vocab = pickle.load(open(args.vocab, 'rb'))

    bj = {}
    for key in text_field.vocab.freqs.keys():
        bj[key] = ""

    ss = 0
    with codecs.open(args.emb) as f:
        next(f)
        for line in f:
            l = line.split()
            if l[0] in bj:
                bj[l[0]] = l[0]
            elif l[0].lower() in bj:
                if bj[l[0].lower()] == "":
                    bj[l[0].lower()] = l[0]
            ss += 1
            if ss % 200000 == 0:
                print(ss)

    count = 0
    for key in bj:
        if bj[key] != "":
            count += 1
    
    print("%d / %d" %(count, len(text_field.vocab)))

    with codecs.open(args.out, 'w') as fout:
        fout.write("%d %d\n" %(count, 300))
        with codecs.open(args.emb) as fin:
            next(fin)
            for line in fin:
                l = line.split()
                if l[0].lower() in bj and bj[l[0].lower()] == l[0]:
                    fout.write(line)
