
from sklearn.feature_extraction.text import TfidfVectorizer
from feature_engineering import *
from gensim.models import word2vec
from scipy import spatial

import os
import gensim
import re
import numpy as np

def clean(data_str):
    word_re = r'(' \
              r'(?:[A-Za-z0-9]+[\.]+[A-Za-z0-9]+[\-]+[A-Za-z0-9]+)|' \
              r'(?:(?:(?:[A-Za-z0-9]+[-]+)+(?:[A-Za-z0-9]+)*))|' \
              r'(?:[$£]+[A-Za-z0-9]+)|' \
              r'(?:[A-Za-z0-9]+)|' \
              r'(?:[\!\?]+)' \
              r')'
    return re.findall(word_re, data_str)

# def clean(s):
#     # Cleans a string: Lowercasing, trimming, removing non-alphanumeric
#
#     return " ".join(re.findall(r'\w+', s, flags=re.UNICODE)).lower()


def tfidf_feature(headline_set, body_set):
    X = []
    train_set = []

    train_set.extend(body_set)
    train_set.extend(headline_set)


    tfidf_vectorizer = TfidfVectorizer()
    tfidf_train = tfidf_vectorizer.fit_transform(train_set).todense()

    row_similarities = [1 - spatial.distance.cosine(tfidf_train[x],tfidf_train[x+len(body_set)]) for x in range(len(body_set)) ]
    X = [[x] for x in row_similarities]
    return X


def word2vecModel():
    # more_sents = get_sents(body_set, headline_set)
    srcPath = r'GoogleNews-vectors-negative300-small.bin'
    path = os.path.abspath(srcPath)
    # load the model
    model = gensim.models.KeyedVectors.load_word2vec_format(path, binary=True)
    # model = gensim.models.Word2Vec.load(path, binary=True)
    # model.train(more_sents)
    return model

model = word2vecModel()


def get_word2vector_f(headline_str, body_str):
    feature =[]
    # word_re = r'(' \
    #           r'(?:[A-Za-z0-9]+[\.]+[A-Za-z0-9]+[\-]+[A-Za-z0-9]+)|' \
    #           r'(?:(?:(?:[A-Za-z0-9]+[-]+)+(?:[A-Za-z0-9]+)*))|' \
    #           r'(?:[$£]+[A-Za-z0-9]+)|' \
    #           r'(?:[A-Za-z0-9]+)|' \
    #           r'(?:[\!\?]+)' \
    #           r')'

    clean_headline = clean(headline_str)
    clean_body = clean(body_str)

    dim = 300

    headline_word_count = 0
    word_vec_head = np.zeros(dim)
    for i, word_head in enumerate(clean_headline):
        try:
            word_vec_head += model[word_head]
            headline_word_count += 1
        except KeyError:
            # ignore if word not in vocabulary
            continue

    body_word_count = 0
    word_vec_body = np.zeros(dim)
    for j, word_body in enumerate(clean_body):
        try:
            word_vec_body += model[word_body]
            body_word_count += 1
        except KeyError:
            # ignore if word not in vocabulary
            continue


    word_vec_head = word_vec_head / headline_word_count
    word_vec_body = word_vec_body / body_word_count


    feature.extend(word_vec_head)
    feature.extend(word_vec_body)

    # print('feature size: {}'.format(len(feature)))
    return feature

def word_overlap_features(headline_str, body_str):


    clean_headline = clean(headline_str)
    clean_body = clean(body_str)
    feature = [
        len(set(clean_headline).intersection(clean_body)) / float(len(set(clean_headline).union(clean_body)))]

    return feature

def refuting_features(headline_str, body_str):
    _refuting_words = [
        'fake',
        'fraud',
        'hoax',
        'false',
        'deny', 'denies',
        # 'refute',
        'not',
        'despite',
        'nope',
        'doubt', 'doubts',
        'bogus',
        'debunk',
        'pranks',
        'retract'
    ]
    clean_headline = clean(headline_str)
    feature = [1 if word in clean_headline else 0 for word in _refuting_words]

    return feature

def polarity_features(headline_str, body_str):
    _refuting_words = [
        'fake',
        'fraud',
        'hoax',
        'false',
        'deny', 'denies',
        'not',
        'despite',
        'nope',
        'doubt', 'doubts',
        'bogus',
        'debunk',
        'pranks',
        'retract'
    ]

    def calculate_polarity(text):

        return sum([t in _refuting_words for t in text]) % 2


    clean_headline = clean(headline_str)
    clean_body = clean(body_str)
    feature = []
    feature.append(calculate_polarity(clean_headline))
    feature.append(calculate_polarity(clean_body))

    return feature


# hand_features
def ngrams(input, n):
    input = input.split(' ')
    output = []
    for i in range(len(input) - n + 1):
        output.append(input[i:i + n])
    return output

def chargrams(input, n):
    output = []
    for i in range(len(input) - n + 1):
        output.append(input[i:i + n])
    return output

def append_chargrams(features, text_headline, text_body, size):
    grams = [' '.join(x) for x in chargrams(" ".join(remove_stopwords(text_headline.split())), size)]
    grams_hits = 0
    grams_early_hits = 0
    grams_first_hits = 0
    for gram in grams:
        if gram in text_body:
            grams_hits += 1
        if gram in text_body[:255]:
            grams_early_hits += 1
        if gram in text_body[:100]:
            grams_first_hits += 1
    features.append(grams_hits)
    features.append(grams_early_hits)
    features.append(grams_first_hits)
    return features

def append_ngrams(features, text_headline, text_body, size):
    grams = [' '.join(x) for x in ngrams(text_headline, size)]
    grams_hits = 0
    grams_early_hits = 0
    for gram in grams:
        if gram in text_body:
            grams_hits += 1
        if gram in text_body[:255]:
            grams_early_hits += 1
    features.append(grams_hits)
    features.append(grams_early_hits)
    return features

def hand_features(headline_str, body_str):
    feature = []
    def binary_co_occurence(headline, body):
        # Count how many times a token in the title
        # appears in the body text.
        bin_count = 0
        bin_count_early = 0
        for headline_token in clean(headline).split(" "):
            if headline_token in clean(body):
                bin_count += 1
            if headline_token in clean(body)[:255]:
                bin_count_early += 1
        return [bin_count, bin_count_early]

    def binary_co_occurence_stops(headline, body):
        # Count how many times a token in the title
        # appears in the body text. Stopwords in the title
        # are ignored.
        bin_count = 0
        bin_count_early = 0
        for headline_token in remove_stopwords(clean(headline).split(" ")):
            if headline_token in clean(body):
                bin_count += 1
                bin_count_early += 1
        return [bin_count, bin_count_early]

    def count_grams(headline, body):
        # Count how many times an n-gram of the title
        # appears in the entire body, and intro paragraph

        clean_body = clean(body)
        clean_headline = clean(headline)
        features = []
        features = append_chargrams(features, clean_headline, clean_body, 2)
        features = append_chargrams(features, clean_headline, clean_body, 8)
        features = append_chargrams(features, clean_headline, clean_body, 4)
        features = append_chargrams(features, clean_headline, clean_body, 16)
        features = append_ngrams(features, clean_headline, clean_body, 2)
        features = append_ngrams(features, clean_headline, clean_body, 3)
        features = append_ngrams(features, clean_headline, clean_body, 4)
        features = append_ngrams(features, clean_headline, clean_body, 5)
        features = append_ngrams(features, clean_headline, clean_body, 6)
        return features


    feature.append(binary_co_occurence(headline_str, body_str)
             + binary_co_occurence_stops(headline_str, body_str)
             + count_grams(headline_str, body_str))

    return feature


# def word2vecFeature(headline_str, body_str):
#     X = []
#     model = word2vecModel()
#     for (headline, body) in zip(headline_set, body_set):
#         feature = []
#         # clean_headline = clean(headline)
#         # clean_body = clean(body)
#         clean_headline = get_tokenized_lemmas(clean_headline)
#         clean_body = get_tokenized_lemmas(clean_body)
#
#         dim = len(model['cat'])
#
#         headline_word_count = 0
#         word_vec_head = np.zeros(dim)
#         for i, word_head in enumerate(clean_headline):
#             try:
#                 word_vec_head += model[word_head]
#                 headline_word_count += 1
#             except KeyError:
#                 # ignore if word not in vocabulary
#                 continue
#
#         body_word_count = 0
#         word_vec_body = np.zeros(dim)
#         for j, word_body in enumerate(clean_body):
#             try:
#                 word_vec_body += model[word_body]
#                 body_word_count += 1
#             except KeyError:
#                 # ignore if word not in vocabulary
#                 continue
#
#
#         word_vec_head = word_vec_head / headline_word_count
#         word_vec_body = word_vec_body / body_word_count
#
#
#         feature.extend(word_vec_head)
#         feature.extend(word_vec_body)
#
#         X.append(feature)
#
#     return X