# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 17:07:29 2020

@author: Rui Lobo
"""
import tweepy
from tweepy import API
from tweepy import Cursor
from tweepy import OAuthHandler

from textblob import TextBlob
from wordcloud import WordCloud
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.style.use('fivethirtyeight')

import re

from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix


# # --- AUTHENTICATION --- # # 
CONSUMER_KEY = "HtpQB7yavzzcAZuNTogKQLiBK"
CONSUMER_SECRET = "NnMLsXpvH1KOGCNvApnBEZm6dV9cueqErlk5XN8P8ZFimTFXnH"
ACCESS_TOKEN = "1082924245217411072-9dve9pDQOgFgTWeS9BDM4a779CV6H6"
ACCESS_TOKEN_SECRET = "jXIAkDiYIpchTVJW5iTULu1eza6XDeAnA0GUw1tkLPqnx"


# # --- TWITTER AUTHENTICATOR - Authentication Twitter credencials --- # #
class TwitterAuthenticator():
    
    # Get the autenthication keys and authenticate in Twitter API.
    def authenticate_twitter_app():
        auth = OAuthHandler(CONSUMER_KEY, CONSUMER_SECRET)
        auth.set_access_token(ACCESS_TOKEN, ACCESS_TOKEN_SECRET)
        api = tweepy.API(auth, wait_on_rate_limit=True)
        return api


# # --- TWEET ANALYZER - Funcionality for analyzing and categorizing content from tweets --- # #
class TweetAnalyzer():
    
    # Clean the tweet texs, removing unnecessary characters.
    def clean_tweet(self, tweet):
        return ' '.join(re.sub("RT (@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", tweet).split())
    
    # Create the DataFrame with the searched tweets .
    def tweets_to_data_frame(self):
        tweet_df=pd.DataFrame()
        api = TwitterAuthenticator.authenticate_twitter_app()
        for hashtag in hash_tag_list:
            tweets = tweepy.Cursor(api.search,
                           q = hashtag,
                           lang = "en",
                           since = date_since).items(100)
           
            data = pd.DataFrame(data=[tweet.text for tweet in tweets], columns = ["text"])
            data["hashtag"] = np.array([hashtag for tweet in data["text"]])
            data["text"] = data["text"].apply(lambda x: self.clean_tweet(x))
            
            tweet_df=pd.concat([tweet_df, data])
        return tweet_df


# # --- TRAIN SET CREATOR --- # # 
class TrainSet():
    
    # Split the DataFrame to create a train set.
    def createTrainSet(df):
        train_set = pd.DataFrame()
        count = 0
        row = []
        columns = 0
        while count < len(df):
            row.append(columns)
            columns += 2
            count += 2
        origin = df
        data = origin.iloc[row]
        train_set=pd.concat([train_set, data])
        
        return train_set


# # --- TEST SET CREATOR --- # # 
class TestSet():
    
    # Split the DataFrame to create a test set.
    def createTestSet(df):
        test_set = pd.DataFrame()
        count = 1
        row = []
        columns = 1
        while count < len(df):
            row.append(columns)
            columns += 2
            count += 2
        origin = df
        data = origin.iloc[row]
        test_set=pd.concat([test_set, data])
        
        return test_set


# # --- TWEET CLASSIFICATOR - functionality to classify and analyze accuracy --- # #
class TweetClassificator():
    
    # This function execute the SGDClassifier.
    def sdg(tweet_df):
        X = tweet_df.text
        y = tweet_df.hashtag
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state = 42)
    
        sgd = Pipeline([('vect', CountVectorizer()),
                        ('tfidf', TfidfTransformer()),
                        ('clf', SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, random_state=42, max_iter=5, tol=None)),
                       ])
        sgd.fit(X_train, y_train)       
        
        y_pred = sgd.predict(X_test)
        
        return {"y_pred": y_pred, "y_test": y_test}


# # --- SENTIMENT ANALYSIS --- # #
class SentimentAnalysis():
    
    # Function to get the subjectivity
    def getSubjectivity(text):
        return TextBlob(text).sentiment.subjectivity
    
    # Function to get the polarity
    def getPolarity(text):
        return TextBlob(text).sentiment.polarity
    
    # Get the polarity and return classification.
    def getAnalysis(score):
        if score > 0:
            return "Positive"
        elif score == 0:
            return "Neutral"
        else:
            return "Negative"


# # --- SENTIMENT ANALYSIS GRAPH--- # #
class GraphAnalysis():

    # Get the sentiment analysis and create bar graphs for each hashtag . 
    def getGraph(dataframe, hash_tag_list):
        for hashtag in hash_tag_list:
            df = pd.DataFrame()
            df=pd.concat([dataframe, df])
            dfc = df[df['hashtag'].str.contains(hashtag)]
            
            plt.title("Sentiment Analysis - %s" % hashtag)
            plt.xlabel('Sentiment')
            plt.ylabel('Counts')
            dfc['Analysis'].value_counts().plot(kind = 'bar')
            plt.show()
            

# # --- WORDCLOUD --- # #
class CloudOfWord():
    from wordcloud import WordCloud
    
    # Get tweet text and create a word cloud with most used words ofr each hashtag.
    def getWordCloud(dataframe, hash_tag_list):
        for hashtag in hash_tag_list:
            df = pd.DataFrame()
            df=pd.concat([dataframe, df])
            dfc = df[df['hashtag'].str.contains(hashtag)]
            
            allWords = ' '.join([twts for twts in dfc['text']])
            wordCloud = WordCloud(width=500, height=300, random_state=21, max_font_size=110).generate(allWords)
            plt.imshow(wordCloud, interpolation="bilinear")
            plt.axis('off')
            plt.show()


# # --- MAIN CLASS --- # #
if __name__=="__main__":
    
    # # ---CREATE DATAFRAME --- # #
    tweet_analyzer = TweetAnalyzer()
    hash_tag_list = ["Covid-19", "Ireland", "Data Mining", "Holiday"]
    date_since = "2020-04-01"
    tweet_df = tweet_analyzer.tweets_to_data_frame()
    
    # # --- PRINT DATAFRAME--- # # 
    pd.set_option("max_colwidth", 40)
    print("\n-------------DataFrame-------------")
    print(tweet_df.head(400))

    # # --- CREATE AND PRINT TRAIN SET --- # # 
    train_set = TrainSet.createTrainSet(tweet_df)
    pd.set_option("max_colwidth", 40)
    print("\n-------------Train Set-------------")
    print(train_set.head(200))
    
    # # --- CREATE AND PRINT TEST SET --- # #
    test_set = TestSet.createTestSet(tweet_df)
    pd.set_option("max_colwidth", 40)
    print("\n-------------Test Set-------------")
    print(test_set.head(200))
    
    # # --- ANALYSIS --- # #
    tweetClassificator = TweetClassificator.sdg(tweet_df)
    y_pred = tweetClassificator["y_pred"]
    y_test = tweetClassificator["y_test"]
    
    print("\n-------------Confusion Matrix-------------\n")
    results = confusion_matrix(y_test, y_pred)
    print(results)
    
    print("\n-------------Accurancy-------------\n")
    print('accuracy %s' % accuracy_score(y_pred, y_test))
    
    print("\n-------------Classification Report-------------\n")
    print(classification_report(y_test, y_pred, target_names=hash_tag_list))
    
    # # --- SENTIMENT ANALYSIS --- # #
    tweet_df['Subjectivity'] = tweet_df['text'].apply(SentimentAnalysis.getSubjectivity)
    tweet_df['Polarity'] = tweet_df['text'].apply(SentimentAnalysis.getPolarity)
    tweet_df['Analysis'] = tweet_df['Polarity'].apply(SentimentAnalysis.getAnalysis)
    pd.set_option("max_colwidth", 40)
    print("\n-------------Sentiment Analysis Set-------------")
    print(tweet_df.head(400))
    
    # # --- SENTIMENT ANALYSIS GRAPH --- # #
    graph = GraphAnalysis.getGraph(tweet_df, hash_tag_list)
    print("\n-------------Sentiment Analysis Graph-------------\n")
    print("\nGraphs successfully created.\n")
    
    # # --- WORD CLOUD --- # #
    cloud = CloudOfWord.getWordCloud(tweet_df, hash_tag_list)
    print("\n-------------Word Cloud-------------\n")
    print("\nWord Clouds successfully created.\n")
    