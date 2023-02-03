#!/usr/bin/env python3
import argparse
import json
import pickle
import re

import sklearn
import pandas as pd
import numpy as np

# import torch
from sklearn.feature_extraction.text import TfidfVectorizer


def parse_args():
    parser = argparse.ArgumentParser(
        description='This is a baseline for task 1 that predicts that each clickbait post warrants a passage spoiler.')

    parser.add_argument('--input', type=str, help='The input data (expected in jsonl format).', required=True)
    parser.add_argument('--output', type=str, help='The classified output in jsonl format.', required=False)

    return parser.parse_args()


def numberSymbol(text):
    properNames = '(?:[A-Z][a-z]+\s)+'
    punctuation = '[?!]+'
    nr = '[0-9]+'

    c = re.findall(
        r'' + properNames + '|' + punctuation,
        text)
    return len(c)


def cosine_sim(text1, text2):
    vectorizer = TfidfVectorizer(
        # tokenizer=normalize, stop_words='english'
    )
    tfidf = vectorizer.fit_transform([text1, text2])
    return ((tfidf * tfidf.T).A)[0, 1]


def load_input(df):
    print(df)
    if type(df) != pd.DataFrame:
        df = pd.read_json(df, lines=True)
    df["postText"] = df["postText"].apply(lambda x: " ".join(x))
    df["allText"] = df[["targetTitle", "postText"]].apply(" ".join, axis=1)
    df["nr_numerals"] = df["targetTitle"].apply(lambda x: numberSymbol(x))
    df["cos"] = df.apply(lambda x: cosine_sim(x.targetTitle, x.postText), axis=1)
    ret = []
    for _, i in df.iterrows():
        ret += [{'targetParagraphs': ' '.join(i['targetParagraphs']),
                 'postText': i['postText'],
                 'targetTitle': i['targetTitle'],
                 'uuid': i['uuid'],
                 'allText': i['allText'],
                 'nr_numerals': i['nr_numerals'],
                 'cos': i['cos'],
                 }]

    return pd.DataFrame(ret)


# def use_cuda():
#   return torch.cuda.is_available() and torch.cuda.device_count() > 0


def predict(df):
    df = load_input(df)
    labels = ['phrase', 'passage', 'multi']
    filename = '/models/phrase_plus_passage_div_multi.sav'
    file_model2 = '/models/finalized_model.sav'
    model1 = pickle.load(open(filename, 'rb'))
    model2 = pickle.load(open(file_model2, 'rb'))
    uuids = list(df['uuid'])
    predictions = model1.predict(df[['targetParagraphs', 'targetTitle', 'postText', 'allText', "nr_numerals", "cos"]])
    for i in range(len(df)):
        if predictions[i] == 2:
            yield {'uuid': uuids[i], 'spoilerType': 'multi'}
        else:
            yield {'uuid': uuids[i], 'spoilerType': labels[model2.predict(df[['targetParagraphs', 'targetTitle', 'postText', 'allText', "nr_numerals", "cos"]].iloc[[i]])[0]]}


def run_baseline(input_file, output_file):
    with open(output_file, 'w') as out:
        for prediction in predict(input_file):
            print(prediction)
            out.write(json.dumps(prediction) + '\n')


if __name__ == '__main__':
    args = parse_args()
    run_baseline(args.input, args.output)
