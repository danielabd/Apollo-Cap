import os

from tweepy import OAuthHandler
from tweepy import API
from tweepy import Cursor
from datetime import datetime, date, time, timedelta
from collections import Counter
import sys
import pandas as pd
import csv
import tweepy
from bs4 import BeautifulSoup
import emoji
import re
import itertools
#import wandb
#wandb.init(project="my-test-project", entity="bdaniela")

#clean text - example - https://towardsdatascience.com/twitter-sentiment-analysis-using-fasttext-9ccd04465597
def clean_text(tweet):
    #source_text= tweet
    #print('source text:\n'+source_text)

    # skip rt tweets
    if tweet.split()[0].lower()=="rt":
        return ""
    #Remove HTML tags
    tweet = BeautifulSoup(tweet).get_text()
    #Remove hashtags
    tweet = ' '.join(re.sub("(@[A-Za-z0-9]+)|(#[A-Za-z0-9]+)", " ", tweet).split())
    #Remove URLs
    tweet = ' '.join(re.sub("(\w+:\/\/\S+)", " ", tweet).split())
    # Lower case
    tweet = tweet.lower()

    #Remove punctuations
    #tweet = ' '.join(re.sub("[\.\,\!\?\:\;\-\=]", " ", tweet).split())
    #Replace contractions #todo: create manually a list of all
    #CONTRACTIONS = {"mayn't":"may not", "may've":"may have"}
    #tweet = tweet.replace("’","'")
    #words = tweet.split()
    #reformed = [CONTRACTIONS[word] if word in CONTRACTIONS else word for word in words]
    #tweet = " ".join(reformed)
    #Fix misspelled words - checking that each character should occur not more than 2 times in every word. It’s a very basic misspelling check.
    #tweet = ''.join(''.join(s)[:2] for _, s in itertools.groupby(tweet))

    ##Replace emojis or emoticons -  todo: maybe there is more emoticons
    ##emoticons
    #SMILEY = {":‑(":"sad", ":‑)":"smiley"}
    #words = tweet.split()
    #reformed = [SMILEY[word] if word in SMILEY else word for word in words]
    #tweet = " ".join(reformed)
    ##emojis
    #tweet = emoji.demojize(tweet)
    #tweet = tweet.replace(":"," ")
    #tweet = ' '.join(tweet.split())

    # remove emojis or emoticons -  todo: maybe there is more emoticons
    # emoticons
    SMILEY_LIST = [":‑(", ":‑)", ":)", ":("]
    words = tweet.split()
    reformed = [word if word not in SMILEY_LIST else '' for word in words]
    tweet = " ".join(reformed)

    # emojis
    '''
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               "]+", flags=re.UNICODE)
    tweet = emoji_pattern.sub(r'', tweet)
    

    allchars = [str for str in tweet.decode('utf-8')]
    emoji_list = [c for c in allchars if c in emoji.UNICODE_EMOJI]
    tweet = ' '.join([str for str in tweet.decode('utf-8').split() if not any(i in str for i in emoji_list)])
    '''
    tweet = emoji.get_emoji_regexp().sub(r'', tweet)
    #Why not use NLTK stop words? - big todo: think if to remove that or not
    #from nltk.corpus import stopwords
    #stop_words = stopwords.words('english')
    #print(stop_words)


    #print('cleaned text:\n'+tweet)
    return tweet


def create_df_for_user(api, user, num_of_tweets, min_tweet_len, max_tweet_len):
    num_of_tweets_to_search = num_of_tweets * 10
    tweets = api.user_timeline(screen_name=user, count=num_of_tweets_to_search, tweet_mode='extended')
    total_tweets = list(tweets)
    early_tweets_id = tweets[-1].id_str
    iterations_to_do = round(num_of_tweets_to_search/len(tweets))
    for i in range(iterations_to_do):
        additional_tweets = api.user_timeline(screen_name=user, count=num_of_tweets_to_search, tweet_mode='extended',max_id = early_tweets_id)
        if len(additional_tweets)==0:
            break
        total_tweets.extend(additional_tweets[1:]) #the first one is duplicated from the last iteration
        early_tweets_id = additional_tweets[-1].id_str
    data_list = []
    sum_of_tweets = 0
    for tweet in total_tweets:
        cleaned_text = clean_text(tweet.full_text)
        if cleaned_text == '':
            continue
        # chack validity of tweet len
        if len(cleaned_text.split()) <= max_tweet_len and len(cleaned_text.split()) >= min_tweet_len:
            data_list.append(cleaned_text)
            sum_of_tweets += 1
            if sum_of_tweets >= num_of_tweets:
                break
    print('User ', tweet.user.screen_name, 'has ', len(data_list), 'tweets.')
    return data_list

def create_twitter_db(max_users,auth,source_db,num_of_tweets,target_dir, min_tweet_len, max_tweet_len, desired_users):
    print('Starting to create twitter DB...')
    columns = ['User', 'Tweet']
    api = tweepy.API(auth)
    data = {columns[0]:[],columns[1]:[]}
    if len(desired_users)>0:
        for user in desired_users:
            tweets = create_df_for_user(api, user, num_of_tweets, min_tweet_len, max_tweet_len)
            for t in tweets:
                data[columns[0]].append(user)
                data[columns[1]].append(t)
    else:
        with open(source_db) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            for i,row in enumerate(csv_reader[1:]):
                if i > max_users:
                    break
                else:
                    # print(f'twitter_username: {row[0]}, twitter_userid: {row[1]}, domain: {row[2]}, name: {row[3]}, followers_count: {row[4]}, tweet_count: {row[5]}.')
                    user = row[0]
                    tweets =  create_df_for_user(api, user, num_of_tweets, min_tweet_len,
                                                                    max_tweet_len)
                    for t in tweets:
                        data[columns[0]].append(user)
                        data[columns[1]].append(t)
            print(f'Processed {len(data)} users\'s tweets.')

    df = pd.DataFrame(data, columns=columns)
    df.head()

    target_file = os.path.join(target_dir,'preprocessed_data.csv')
    df.to_csv(target_file, index=False)
    print('created new db in: ', target_file)



def main():
    # set my keys
    # import tweepy
    # your Twitter API key and API secret

    data_file = '/home/bdaniela/personal_keys.txt'
    with open(data_file) as f:
        for line in f:
            splitted_line = line.split()
            if splitted_line[0]== 'api_key':
                api_key = splitted_line[-1]
            elif splitted_line[0]== 'api_secret':
                api_secret = splitted_line[-1]
            elif splitted_line[0]== 'access_token':
                access_token = splitted_line[-1]
            elif splitted_line[0]== 'access_token_secret':
                access_token_secret = splitted_line[-1]
            elif splitted_line[0]== 'bearer':
                bearer = splitted_line[-1]

    # authenticate
    auth = tweepy.OAuthHandler(api_key, api_secret)
    #api = tweepy.API(auth, wait_on_rate_limit=True)

    # Db of famous names
    datapath = 'DB.csv'
    base_path = '/home/bdaniela/zero-shot-style/zero_shot_style/model/data'
    df = pd.read_csv(os.path.join(base_path,datapath))
    df.head()

    # create database of twitter
    source_db = os.path.join(base_path,'DB.csv')
    max_users = 100  # maximum users to classify
    num_of_tweets = 1000  # take number of last tweets
    target_dir = base_path
    min_tweet_len = 3 #10
    max_tweet_len = 20
    desired_users = ['justinbieber', 'BillGates','rihanna','KendallJenner','elonmusk','JLo']
    create_twitter_db(max_users, auth, source_db, num_of_tweets, target_dir, min_tweet_len, max_tweet_len,desired_users)
    print('finish!')
#justinbieber
#BillGates
#MichelleObama
if __name__=='__main__':
    main()