#!/usr/bin/env python
# coding: utf-8

# In[20]:


import pandas as pd
import numpy as np
import re
import string
import nltk
import matplotlib.pyplot as plt
from wordcloud import WordCloud,STOPWORDS
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from pandas import DataFrame

from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from nltk.stem import WordNetLemmatizer, PorterStemmer

import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D


# In[21]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[22]:


get_ipython().system(' pip install nltk')


# In[23]:


nltk.download('vader_lexicon')


# In[24]:


from nltk.sentiment.vader import SentimentIntensityAnalyzer


# In[25]:


get_ipython().system(' pip install vaderSentiment')
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


# In[26]:


df = pd. read_csv('covidvaccine1130.csv')


# In[8]:


df


# In[9]:


print(df.text)


# In[10]:


df.dtypes


# In[27]:


from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyser = SentimentIntensityAnalyzer()


# In[28]:


pip install tweepy


# In[29]:


#cleaning the tweets
def remove_pattern(input_txt, pattern):
    r = re.findall(pattern, input_txt)
    for i in r:
        input_txt = re.sub(i, '', input_txt)        
    return input_txt
def clean_tweets(tweets):
    #remove twitter Return handles (RT @xxx:)
    tweets = np.vectorize(remove_pattern)(tweets, "RT @[\w]*:") 
    
    #remove twitter handles (@xxx)
    tweets = np.vectorize(remove_pattern)(tweets, "@[\w]*")
    
    #remove URL links (httpxxx)
    tweets = np.vectorize(remove_pattern)(tweets, "https?://[A-Za-z0-9./]*")
    
    #remove special characters, numbers, punctuations (except for #)
    tweets = np.core.defchararray.replace(tweets, "[^a-zA-Z]", " ")
    
    return tweets


# In[30]:


df['text'] = clean_tweets(df['text'])
df['text'].head()


# In[31]:


# initialize the Analyzer

sid = SentimentIntensityAnalyzer()

# For each sentence get the score


for text in df.text:

 print(text)

 ss = sid.polarity_scores(text)

  # Print the scores

 for k in sorted(ss):
         print('{0}: {1}, '.format(k, ss[k]))
 print()


# In[32]:


scores = []
# Declare variables for scores
compound_list = []
positive_list = []
negative_list = []
neutral_list = []
for i in range(df['text'].shape[0]):
#print(analyser.polarity_scores(sentiments_pd['text'][i]))
    compound = analyzer.polarity_scores(df['text'][i])["compound"]
    pos = analyzer.polarity_scores(df['text'][i])["pos"]
    neu = analyzer.polarity_scores(df['text'][i])["neu"]
    neg = analyzer.polarity_scores(df['text'][i])["neg"]
    
    scores.append({"Compound": compound,
                       "Positive": pos,
                       "Negative": neg,
                       "Neutral": neu
                  })


# In[ ]:


sentiments_score = pd.DataFrame.from_dict(scores)
df = df.join(sentiments_score)
df.head()


# In[35]:


tweet = []
vs_compound = []
vs_pos = []
vs_neu = []
vs_neg = []
vs_sentiment = []


             
for i in range(0, len(df.text)):

    tweet.append(df.text[i])

    vs_compound.append(sid.polarity_scores(df.text[i])['compound'])

    vs_pos.append(sid.polarity_scores(df.text[i])['pos'])

    vs_neu.append(sid.polarity_scores(df.text[i])['neu'])

    vs_neg.append(sid.polarity_scores(df.text[i])['neg'])


# In[36]:


twitter_df = pd.DataFrame({'Tweet': tweet,
                        'Compound': vs_compound,
                        'Positive': vs_pos,
                        'Neutral': vs_neu,
                        'Negative': vs_neg})

twitter_df = twitter_df[['Tweet', 'Compound',
                         'Positive', 'Neutral', 'Negative']]


# In[37]:


twitter_df


# In[38]:


sentiment =  []

for row in twitter_df['Compound']:
    if row > 0.5:
        sentiment.append('positive')
    elif row >-0.5:
        sentiment.append('neutral')
    else:
        sentiment.append('negative')


# In[39]:


twitter_df['sentiment']=sentiment


# In[40]:


twitter_df


# In[41]:


sns.countplot(twitter_df.sentiment)


# In[42]:


twitter_df.sentiment.value_counts(normalize=True)


# In[43]:


##Emotional Analysis


# In[44]:


lemmatizer = WordNetLemmatizer()


# In[45]:


tweet_text = []


# In[47]:


for i in range(0, len(df.text)):
    tweet_text.append(df.text[i])


# In[48]:


tweet_text[0]


# In[49]:


re.sub('http[s]?://\S+\n','', tweet_text[0])


# In[50]:


len(tweet_text)


# In[51]:


for i in range(0, len(tweet_text)):
    print(re.sub('http[s]?://\S+\n','', tweet_text[i]))


# In[52]:


new_tweet_text = []

for i in range(0, len(tweet_text)):
    new_tweet_text.append(re.sub('http[s]?://\S+\n','', tweet_text[i]))


# In[53]:


pd_tweet = pd.DataFrame(new_tweet_text, columns =['text'])


# In[54]:


pd_tweet


# In[55]:


stop = stopwords.words('english')
description_list=[]
for description in pd_tweet['text']:
    description=re.sub("[^a-zA-Z]", " ", description)
    description=description.lower()
    description=nltk.word_tokenize(description)
    description=[word for word in description if not word in set(stopwords.words("english"))]
    lemma=nltk.WordNetLemmatizer()
    description=[lemma.lemmatize(word) for word in description]
    description=" ".join(description)
    description_list.append(description)
pd_tweet["normalized_text_new"]=description_list
pd_tweet.head(5)


# In[56]:


from nrclex import NRCLex 
text_object = NRCLex(' '.join(pd_tweet['normalized_text_new']))


# In[57]:


text_object.affect_frequencies


# In[58]:


text_object.top_emotions


# In[59]:


sentiment_scores = pd.DataFrame(list(text_object.raw_emotion_scores.items())) 


# In[60]:


sentiment_scores = sentiment_scores.rename(columns={0: "Sentiment", 1: "Count"})
sentiment_scores


# In[61]:


import plotly
import plotly.express as px

fig = px.pie(sentiment_scores, values='Count', names='Sentiment',
             title='Sentiment Scores',
             hover_data=['Sentiment'])
fig.update_traces(textposition='inside', textinfo='percent+label')
fig.show()


# In[62]:


#Model


# In[63]:


from sklearn.feature_extraction.text  import TfidfVectorizer, CountVectorizer

bow_transformer = CountVectorizer().fit(twitter_df['Tweet'])

print (len(bow_transformer.vocabulary_))


# In[64]:


twitter_bow = bow_transformer.transform(twitter_df['Tweet'])


# In[65]:


from sklearn.feature_extraction.text  import TfidfTransformer

tfidf_transformer = TfidfTransformer().fit(twitter_bow)

# To transform the entire bag-of-words corpus into TF-IDF corpus at once:
   
twitter_tfidf = tfidf_transformer.transform(twitter_bow)

print (twitter_tfidf.shape)


# In[66]:


Y = sentiment


# In[67]:


from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(twitter_tfidf, twitter_df['sentiment'], test_size=0.4, random_state=33)


# In[68]:


from sklearn.naive_bayes import MultinomialNB
nb = MultinomialNB()


# In[69]:


nb.fit(X_train, Y_train)


# In[70]:


print("Accuracy on training set: {:.3f}".format(nb.score(X_train, Y_train)))
print("Accuracy on testing set: {:.3f}".format(nb.score(X_test, Y_test)))


# In[72]:


from sklearn.metrics import confusion_matrix
nb_y_pred = nb.predict(X_test)


# In[73]:


confusion_matrix(Y_test,nb_y_pred)


# In[74]:


from sklearn.metrics import classification_report
print(classification_report(Y_test,nb_y_pred))


# In[40]:


from wordcloud import WordCloud,STOPWORDS

def word_cloud(wd_list):
    stopwords = set(STOPWORDS)
    all_words = ' '.join([text for text in wd_list])
    wordcloud = WordCloud(
        background_color='white',
        stopwords=stopwords,
        width=1600,
        height=800,
        random_state=1,
        colormap='jet',
        max_words=80,
        max_font_size=200).generate(all_words)
    plt.figure(figsize=(12, 10))
    plt.axis('off')
    plt.imshow(wordcloud, interpolation="bilinear");
word_cloud(df['text'])


# In[ ]:





# In[ ]:





# In[ ]:




