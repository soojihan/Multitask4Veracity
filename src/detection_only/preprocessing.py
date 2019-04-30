import numpy as np
import os
from collections import Counter
from pprint import pprint as pp
import re
from typing import List
from gensim.utils import deaccent
import string
from nltk.tokenize import TweetTokenizer
import os
import pandas as pd
import csv
import unicodedata
import nltk
import gensim
from glob import glob
from data_loader import *
from datetime import datetime
from operator import itemgetter
import pickle
"""
Preprocess data
"""
np.set_printoptions(threshold=np.nan)
pd.set_option('display.expand_frame_repr', False)

### Check the original data released by the authors
# # data_path = './saved_data/saved_data_MTL2_detection/putinmissing/'
# # data_path = './saved_data/source-tweets/boston-9000-0.3/'
# data_path = os.path.join('..', 'saved_data/saved_data_MTL2_detection/sydneysiege/')
#
# ### Check the original data released by the authors
# # x = np.load(os.path.join(data_path, 'rnr_labels.npy'))
# x = np.load(os.path.join(data_path, 'ids.npy'))
# print(len(x))
# print(x.shape)
# print(type(x[0]))
# # x = np.load(os.path.join(data_path, 'train_array.npy'))
# x = np.load(os.path.join(data_path, 'train_array.npy'))
# print(len(x))
# print(x.shape)
# print(type(x[0]))

def cleantweet(tweettext):
    tweettext = re.sub(r"pic.twitter.com\S+", "picpicpic", tweettext)
    tweettext = re.sub(r"http\S+", "urlurlurl", tweettext)
    return tweettext

def preprocessing_tweet_text(tweet_text) -> List[str]:
    """
    Neural Language Model like ELMo does not need much normalisation. Pre-trained ELMo model only need pre-tokenised text.

    :param tweet_text:
    :return:
    """
    if not isinstance(tweet_text, str):
        raise ValueError("Text parameter must be a Unicode object (str)!")

    norm_tweet = tweet_text.lower()
    # remove retweets
    norm_tweet = re.sub(r'^(rt)( @\w*)?[: ]', '', norm_tweet)
    # remove URL
    norm_tweet = re.sub(r"http\S+", "", norm_tweet)
    # remove pic URL
    norm_tweet = re.sub(r"pic.twitter.com\S+", "", norm_tweet)
    # remove user mentions
    norm_tweet = re.sub(r"(?:\@|https?\://)\S+", "", norm_tweet)
    # remove punctuations:
    # norm_tweet = re.sub(pattern=r'[\!"#$%&\*+,-./:;<=>?@^_`()|~=]', repl='', string=norm_tweet).strip()
    # deaccent
    norm_tweet = deaccent(norm_tweet)
    return norm_tweet
    #
    # tknzr = TweetTokenizer()
    # tokenised_norm_tweet = tknzr.tokenize(norm_tweet)
    #
    # # Set the minimum number of tokens to be considered
    # if len(tokenised_norm_tweet) < 4:
    #     return []
    #
    # num_unique_terms = len(set(tokenised_norm_tweet))
    #
    # # Set the minimum unique number of tokens to be considered (optional)
    # if num_unique_terms < 2:
    #     return []
    #
    # return tokenised_norm_tweet

def loadW2vModel():
    # LOAD PRETRAINED MODEL
    global model
    print ("Loading the model")
    model = gensim.models.KeyedVectors.load_word2vec_format(
            os.path.join('..', '..', 'data/downloaded_data', 'GoogleNews-vectors-negative300.bin'), binary=True)
    print ("Done!")

def str_to_wordlist(tweettext, remove_stopwords=False):

    #  Remove non-letters
    # NOTE: Is it helpful or not to remove non-letters?
    # str_text = re.sub("[^a-zA-Z]"," ", str_text)
    tweettext = preprocessing_tweet_text(tweettext)
    str_text = re.sub("[^a-zA-Z]", " ", tweettext)
    # Convert words to lower case and split them
    # words = str_text.lower().split()
    words = nltk.word_tokenize(str_text.lower())
    # Optionally remove stop words (false by default)
    # NOTE: generic list of stop words, should i remove them or not?
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        words = [w for w in words if w not in stops]
    if len(words) < 4:
        return []
    # 5. Return a list of words
    return(words)

def sumw2v(tweet, avg=True):
    global model
    num_features = 300
    temp_rep = np.zeros(num_features)
    wordlist = str_to_wordlist(tweet,  remove_stopwords=False)
    for w in range(len(wordlist)):
        if wordlist[w] in model:
            temp_rep += model[wordlist[w]]
    if avg:
        if len(wordlist) == 0:
            sumw2v = temp_rep
        else:
            sumw2v = temp_rep/len(wordlist)
    else:
        # sum
        sumw2v = temp_rep
    return sumw2v

def get_max_branch_len():
    """
    Compute the maximum depth of conversational threads per event
    :return:
    """
    source_tweets = load_abs_path(os.path.join("..", 'downloaded_data/pheme_annotations'))
    source_files = glob(os.path.join(source_tweets, '*.csv'))
    max_branch ={}
    for f in source_files:
        df = pd.read_csv(f)
        pattern = r"[a-zA-Z*]+"
        re.compile(pattern)
        event = re.findall(pattern, os.path.basename(f))[0]
        max_br = 0
        for i, row in df.iterrows():
            source_id = str(row['tweet_id'])
            contexts = list(load_source_tweet_context(source_id))  # load context tweet json
            if max_br < len(contexts):
                max_br = len(contexts)
        max_branch[event] = max_br
    print(max_branch)
    with open(os.path.join('..', 'saved_data/pheme5_max_branch.pickle'), 'wb') as f:
        pickle.dump(max_branch, f)

def preprocessing_context(event, data_path):
    loadW2vModel()
    # with open(os.path.join('..', 'saved_data/pheme5_max_branch.pickle'), 'rb') as f:
    #     max_branch = pickle.load(f)
    # print(max_branch.keys())

    num_features =300
    source_tweet_metadata = os.path.join(data_path, '{}'.format(event), 'aug_metadata.pickle')
    # source_tweet_metadata = os.path.join(data_path, '{}'.format(event), 'saved_tweets/aug_metadata.pickle')
    with open(source_tweet_metadata, 'rb') as infile:
        source_df = pickle.load(infile)

    # pattern = r"[a-zA-Z*]+"
    # re.compile(pattern)
    # event = re.findall(pattern, os.path.basename(f))[0]

    ## Set the maximum length of branches
    # max_br = max_branch[event]
    # max_br = min(max_br, 24)
    max_br = 24

    print("Event {}".format(event))
    # raise SystemExit
    source_df['created_at'] = pd.to_datetime(source_df['created_at'], format='%a %b %d %H:%M:%S +0000 %Y')
    # source_df['created_at'] = pd.to_datetime(source_df['created_at'], format='%Y-%m-%d %H:%M:%S')
    source_df = source_df.sort_values(by='created_at')

    ids = []
    labels = []
    temp_list =[]
    pp(list(source_df))
    for i, row in source_df.iterrows():
        temp = np.zeros((max_br+1, num_features), dtype=np.float32)
        print(temp.shape)
        # source_id = str(row['tweet_id']).encode('UTF-8')
        source_id = str(row['tweet_id'])
        ids.append(source_id)
        text = row['text']
        avgw2v = sumw2v(text, avg=True)
        features = np.asarray(avgw2v, dtype=np.float32).reshape(1, -1)
        temp[0] = features
        # label = True if row['label'] == 1 else False
        label = row['label']
        labels.append(label)
        contexts = list(load_source_tweet_context(source_id)) # TODO: change 'social_context_dir' in data_loader.py; load context tweet json
        # print(contexts['created_at'])
        contexts = sorted(contexts, key=lambda x: datetime.strptime(x['created_at'], '%a %b %d %H:%M:%S +0000 %Y'))
        print("Number of replies ", len(contexts))
        for j, c in enumerate(contexts):
            if 'text' in c:
                c_text = c['text']
            else:
                c_text = c['full_text'] # replies collected using twarc
            avgw2v_c = sumw2v(c_text, avg=True)
            features_c = np.asarray(avgw2v, dtype=np.float32).reshape(1, -1)
            temp[j+1] = features_c
            if j==23:
                break
        # print("Feature shape ", temp.shape)
        temp_list.append(temp)
    print(event, max_br)
    labels = np.asarray(labels, dtype=bool)
    ids = np.asarray(ids).reshape(-1, 1)
    final_features = np.asarray(temp_list)
    print("id shape ", ids.shape)
    print("label shape ", labels.shape)
    print("feature shape ", final_features.shape)
    outpath = os.path.join('..', '..', 'data/saved_data_hydrator/augpheme-top25-complete/{}'.format(event))
    # outpath = os.path.join('..', '..', 'data/saved_data_twitter1516/{}'.format(event, event))
    # outpath = os.path.join('..', '..', 'data/saved_data_aprilfools/{}'.format(event, event))
    os.makedirs(outpath, exist_ok=True)
    np.save(os.path.join(outpath, 'ids'), ids)
    np.save(os.path.join(outpath, 'train_array'), final_features)
    np.save(os.path.join(outpath, 'rnr_labels'), labels)
    print("")


def preprocessing():
    """
    Generate input data for the rumour detection model
    without context
    :return:
    """
    # loadW2vModel()

    # files = glob('./input_data/boston-9000.csv')
    files = glob('./input_data/*.csv')

    for f in files:
        pattern = r"[a-zA-Z*]+"
        re.compile(pattern)
        event = re.findall(pattern, os.path.basename(f))[0]
        print(event)
        df = pd.read_csv(f)
        print(df.head())
        df['created_at'] = pd.to_datetime(df['created_at'], format='%Y-%m-%d %H:%M:%S')
        df = df.sort_values(by='created_at')
        print(df.head())
        ids =[]
        feature_set =[]
        labels = []
        for i, row in df.iterrows():
            thread_features =[]
            # tweet_id = row['tweet_id']
            tweet_id = row['id']
            text = row['text']
            tweet_id = str(tweet_id).encode('UTF-8')
            ids.append(tweet_id)
            avgw2v = sumw2v(text, avg=True)
            features = np.asarray(avgw2v, dtype=np.float32).reshape(1,-1) # (1, 300)
            feature_set.append(features)
            label = True if row['label']==1 else False
            labels.append(label)

        labels = np.asarray(labels, dtype=bool)
        ids = np.asarray(ids).reshape(-1,1)
        feature_set = np.asarray(feature_set)
        print(ids.shape)
        print(feature_set.shape)
        outpath = os.path.join(os.path.dirname(__file__), 'saved_data/source-tweets/{}'.format(event))
        os.makedirs(outpath, exist_ok=True)
        np.save(os.path.join(outpath, 'ids'), ids)
        np.save(os.path.join(outpath, 'train_array'), feature_set)
        np.save(os.path.join(outpath, 'rnr_labels'), labels)

def main():
    event = 'ferguson'
    # data_path = load_abs_path(os.path.join("..", 'downloaded_data/augmented_data_pheme'))
    # data_path = load_abs_path(os.path.join("..", '..', 'data/twitter1516'))
    data_path = load_abs_path(os.path.join("..", '..', 'data/downloadeD_data/augmented_data_pheme-asonam'))
    # data_path = load_abs_path(os.path.join("..", '..', 'data/training_data/aprilfools'))
    preprocessing_context(event, data_path)
    # get_max_branch_len()

main()