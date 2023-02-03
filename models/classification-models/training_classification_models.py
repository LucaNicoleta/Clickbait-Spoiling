import pickle
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import NMF
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from nltk.tokenize import NLTKWordTokenizer, word_tokenize
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from preprocessing_functions import prepare_data

df_train = prepare_data("../data/train.jsonl")
df_test = prepare_data("../data/validation.jsonl")

text_columns = ['targetParagraphs', 'targetTitle', 'postText']
numeric_columns = ["nrProperNounsAndSymbols", "cos"]
custom = NLTKWordTokenizer()
# cream pipeline-uri pt preprocesarea datelor care sunt comune ambelor modele

# pipeline pentru extragerea topicelor
nmf_pipe = Pipeline([('tdf', TfidfVectorizer(
    ngram_range=(1, 3), max_features=2000, min_df=2, max_df=0.7, stop_words="english", tokenizer=word_tokenize
)), ('lda',
     NMF(
         n_components=2,
         random_state=1,
         init="nndsvda",
         beta_loss="kullback-leibler",
         solver="mu",
         max_iter=1000,
         alpha_W=0.00005,
         alpha_H=0.00005,
         l1_ratio=0.5,
     ))])
# pipeline pentru coloane numerice
num_pipeline = Pipeline(steps=[
    ('impute', SimpleImputer(strategy='mean')),
    ('scale', MinMaxScaler())
])


def svm():
    title_pipeline = Pipeline([
        ('vect', CountVectorizer(stop_words="english")),
        ('tdf', TfidfTransformer(sublinear_tf=True))
    ])
    description_pipeline = Pipeline([
        ('vect', CountVectorizer(stop_words="english")),
        ('tdf', TfidfTransformer(sublinear_tf=True))
    ])
    par_pipeline = Pipeline([
        ('vect', CountVectorizer(stop_words="english", ngram_range=(1, 3), min_df=2
                                 , tokenizer=custom.tokenize
                                 )),
        ('tdf', TfidfTransformer(sublinear_tf=True))
    ])
    preprocessor = ColumnTransformer([
        ('targetTitle', title_pipeline, text_columns[1]),
        ('postText', description_pipeline, text_columns[2]),
        ('targetParagraphs', par_pipeline, text_columns[0]),
        ('topic', nmf_pipe, 'allText'),
        ('numericFeatures', num_pipeline, ["nrProperNounsAndSymbols"])
    ])
    model_svc = Pipeline([
        ('preprocessor', preprocessor),
        ('clf',
         SVC(
             C=10, gamma=0.1, probability=True
         )),
    ])
    model_svc.fit(df_train[['targetTitle', 'targetParagraphs', 'postText', 'allText', "nrProperNounsAndSymbols", "cos"]],
                  df_train['tags'])

    y_pred = model_svc.predict(
        df_test[['targetTitle', 'targetParagraphs', 'postText', 'allText', "nrProperNounsAndSymbols", "cos"]])
    # calculam scorul
    print('Accuracy SVC:', accuracy_score(df_test["tags"], y_pred))


def logistic_regression():
    # convert the text data into vectors
    title_pipeline = Pipeline([
        ('vect', CountVectorizer(stop_words="english")),
        ('tdf', TfidfTransformer())
    ])
    description_pipeline = Pipeline([
        ('vect', CountVectorizer(stop_words="english")),
        ('tdf', TfidfTransformer())
    ])
    par_pipeline = Pipeline([
        ('vect', CountVectorizer(stop_words="english", ngram_range=(1, 3), min_df=2, tokenizer=word_tokenize)),
        ('tdf', TfidfTransformer())
    ])
    # definim modelul
    preprocessor = ColumnTransformer([
        ('targetTitle', title_pipeline, text_columns[1]),
        ('postText', description_pipeline, text_columns[2]),
        ('targetParagraphs', par_pipeline, text_columns[0]),
        ('topicExtraction', nmf_pipe, 'allText'),
        ('numericFeatures', num_pipeline, numeric_columns)
    ])
    model_lr = Pipeline([
        ('preprocessor', preprocessor),
        ('clf',
         LogisticRegression(
             max_iter=300
             # C=1.623776739188721, penalty='l2', solver='liblinear'
         )),
    ])
    # antrenam modelul
    model_lr.fit(df_train[['targetTitle', 'targetParagraphs', 'postText', 'allText', "nrProperNounsAndSymbols", "cos"]],
                 df_train['tags'])

    y_pred = model_lr.predict(
        df_test[['targetTitle', 'targetParagraphs', 'postText', 'allText', "nrProperNounsAndSymbols", "cos"]])
    # calculam scorul
    print('Accuracy Logistic Regression:', sklearn.metrics.classification_report(df_test["tags"], y_pred))


    pickle.dump(model_lr, open('finalized_model.sav', 'wb'))

logistic_regression()