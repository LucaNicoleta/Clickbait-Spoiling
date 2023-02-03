
import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer



# functions for calculating additional features
def count_symbols_and_proper_nouns(text):
    proper_names = '(?:[A-Z][a-z]+\s)+'
    punctuation = '[?!]+'

    c = re.findall(
        r'' + proper_names + '|' + punctuation,
        text)
    return len(c)


def count_numbers(text):
    nr = '[0-9]+'

    c = re.findall(
        r'' + nr,
        text)
    return len(c) > 0


def cosine_sim(text1, text2):
    tfidf = TfidfVectorizer().fit_transform([text1, text2])
    return (tfidf * tfidf.T).A[0, 1]


# function for reading and preparing data
def prepare_data(file_name: str) -> pd.DataFrame:
    # citim json si il stocam intr-un dataframe
    df = pd.read_json(file_name, lines=True)
    # coloana tags contine momentan un array cu un singur element asa ca desfacem array-ul
    df["tags"] = list(map(lambda x: x[0], df["tags"].tolist()))
    # etichetam valorile din tags cu valori numerice
    df["tags"] = df["tags"].apply(lambda x: 0 if x == 'phrase' else 1 if x == 'passage' else 2)
    df["nrPar"] = df["targetParagraphs"].apply(lambda x: len(x))
    # desfacem array-ul de stringuri din coloana ce contine paragrafele si coloana ce contine textul postarii
    df["targetParagraphs"] = df["targetParagraphs"].apply(lambda x: " ".join(x))
    df["postText"] = df["postText"].apply(lambda x: " ".join(x))
    # ne cream o coloana care sa combine titlul cu textul postarii(pentru extragerea topicilor)
    df["allText"] = df[["targetTitle", "postText"]].apply(" ".join, axis=1)
    # ne cream o coloana in care sa stocam numarul de substantive proprii si a semnelor de punctuatie
    df["nrProperNounsAndSymbols"] = df["targetTitle"].apply(lambda x: count_symbols_and_proper_nouns(x))
    # ne cream o coloana in care sa stocam procentajul de similiraitate dintre titlu si postare
    df["cos"] = df.apply(lambda x: cosine_sim(x.targetTitle, x.postText), axis=1)
    return df

