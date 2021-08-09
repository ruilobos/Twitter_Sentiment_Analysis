# Twitter - Sentiment Analysis

## Objective
The purpose of this project is to apply several Data Analysis techniques to extract the necessary information from tweets posted on Twitter. The data is then used to perform Sentiment Analysis in relation to certain keywords.

The Software analyses all tweets posted in English since April 2020 that contain the specified keywords. Each tweet is classified as being positive, neutral or negative. The program also makes a word cloud with the words most related to the keywords.

## Description
The software uses the Twitter API to access tweets. They are filtered by date, language and keyword. Half of the tweets are used to create the training set and the other half is used in the test set, in this step the panda library is used. In the next step, the tweets are classified (using Sklearn) and the sentiment analysis is done using TextBlob.

![image](https://user-images.githubusercontent.com/34349410/128758330-80619f55-536f-4523-bb1d-583f136ca151.png)

The next step is to generate a graph (Matplotlib) with the amount of positive, neutral and negative tweets for each keyword. Finally, the word cloud is created, using WordCloud, which displays the words that most appear along with the keyword. Words with higher occurrences are displayed with a larger font than words with fewer occurrences.

## Features
-	Extraction of tweets related to keywords;
-	Analysis of data;
-	Word Cloud - Words with more relation to keywords;
-	Graphics with sentiment analysis.

Technical Features:

-	Twitter API.
-	Anaconda;
-	Spyder;
-	Python libraries: Tweepy, Textblod, Wordcloud, Numpy, Pandas, Matplotlib and Sklearn.

## Credits
This project was part of the Data Mining class of my Master of Applied Software Development program. The premises and the keywords were defined by the professor. All code developed for this project is my authorship.

## Project Status
This project is finished because I understand that it fulfilled all the objectives, both in the use of Twitter API and in Data Analysis (sentiment analysis).


