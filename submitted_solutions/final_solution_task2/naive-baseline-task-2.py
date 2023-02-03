#!/usr/bin/env python3
import argparse
import json

from transformers import pipeline, AutoTokenizer
import pickle
import pandas as pd



def parse_args():
    parser = argparse.ArgumentParser(description='This is a baseline for task 2 that spoils each clickbait post with the title of the linked page.')

    parser.add_argument('--input', type=str, help='The input data (expected in jsonl format).', required=True)
    parser.add_argument('--output', type=str, help='The spoiled posts in jsonl format.', required=False)

    return parser.parse_args()


def load_input(df):
    print(df)
    if type(df) != pd.DataFrame:
        df = pd.read_json(df, lines=True)
    df["tags"] = list(map(lambda x: x[0], df["tags"].tolist()))
    df["tags"] = df["tags"].apply(lambda x: 0 if x == 'phrase' else 1 if x == 'passage' else 2)
    df["lung_par"] = df["targetParagraphs"].apply(lambda x: [len(i) for i in x])
    df["targetParagraphs"] = df["targetParagraphs"].apply(lambda x: " ".join(x))
    df["postText"] = df["postText"].apply(lambda x: " ".join(x))
    df["allText"] = df[["targetTitle", "targetParagraphs"]].apply(" ".join, axis=1)

    ret = []
    for _, i in df.iterrows():
        ret += [{'targetParagraphs': ' '.join(i['targetParagraphs']),
                 'postText': i['postText'],
                 'targetTitle': i['targetTitle'],
                 'uuid': i['uuid'],
                 'allText': i['allText'],
                 'tags': i['tags'],
                 'lung_par': i['lung_par']
                 }]

    return pd.DataFrame(ret)


def predict(df):
    df = load_input(df)





    labels = ['phrase', 'passage', 'multi']
    filename = '/models/task2_phrase.sav'
    filename2 ='/models/task2_passage.sav'
    filename3 ='/models/task2_multi.sav'
    tokenizer = AutoTokenizer.from_pretrained('/models/deepset/roberta-base-squad2')
    model1 = pickle.load(open(filename, 'rb'))
    model2 = pickle.load(open(filename2, 'rb'))
    model3 = pickle.load(open(filename3, 'rb'))
    uuids = list(df['uuid'])

    phrase = pipeline("question-answering", model=model1, tokenizer=tokenizer, device=0)
    passage = pipeline("question-answering", model=model2, tokenizer=tokenizer, device=0)
    multi = pipeline("question-answering", model=model3, tokenizer=tokenizer, device=0)

    for i in range(len(df)):
        if df['tags'][i] == 2:
            yield {'uuid': uuids[i], 'spoiler': multi(question=df["postText"].tolist()[i], context=df["allText"].tolist()[i])['answer']}
        elif df['tags'][i] == 1:
            yield {'uuid': uuids[i], 'spoiler': passage(question=df["postText"].tolist()[i], context=df["allText"].tolist()[i])['answer']}
        else:
            yield {'uuid': uuids[i], 'spoiler': phrase(question=df["postText"].tolist()[i], context=df["allText"].tolist()[i])['answer']}



def run_baseline(input_file, output_file):
    with open(output_file, 'w') as out:
        for prediction in predict(input_file):
            print(prediction)
            out.write(json.dumps(prediction) + '\n')

if __name__ == '__main__':
    args = parse_args()
    run_baseline(args.input, args.output)

