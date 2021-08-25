# Import Library
import nltk as nltk
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import string
import spacy as sp
import matplotlib.patches as mpatches
import plotly
import plotly.offline as pyo
pyo.init_notebook_mode()
import plotly.graph_objs as go
import plotly.figure_factory as ff
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import NMF  
from gensim.parsing.preprocessing import remove_stopwords
from wordcloud import WordCloud
from wordcloud import WordCloud,STOPWORDS
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.sentiment import vader
from nltk.tokenize import RegexpTokenizer
from nltk.util import ngrams
from nltk.tag import pos_tag
from nltk.chunk import ne_chunk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk.corpus import stopwords
from nltk import word_tokenize
from scipy import stats
from heapq import nlargest
sns.set_style('darkgrid')

#Import Dataset
# Trump Rally Text as Corpus
corpus = []          
for dirname, _, filenames in os.walk("Trump_Rally_Speech_Text_Corpora/"):
    for filename in filenames:
        with open((os.path.join(dirname, filename)), encoding="UTF-8") as file_input: corpus.append(file_input.read(),)
# Creating Main Dataframe
place_name = [i.replace(".txt"," ") for i in filenames]
months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
place, month_list, date = [],[],[]
for name in place_name:
    index= -1
    for month in months:
        index =name.find(month)
        if index != -1:
            month_list.append(month)
            break
    place.append(name[:index])
    date.append(name[index+3:])

trump_df = pd.DataFrame({"Month":month_list,"Year":date,"City":place, "Speech":corpus})
trump_df[['Day','Year']] = trump_df.Year.str.split("_",expand=True,)

trump_df["City"] = trump_df["City"].str.replace( r"([A-Z])", r" \1").str.strip()
trump_df["City"] = trump_df["City"].replace("Winston- Salem", "Winston-Salem")
trump_df.head(5)

# Importing Other Dataset (State List)
metrics1_df = pd.read_csv("list.csv",keep_default_na=False, na_values=[""])
metrics1_df.head(5) 

# Importing Other Dataset (State Metrics)
metrics2_df = pd.read_csv("other data.csv",keep_default_na=False, na_values=[""])
metrics2_df.head(5) 

# Importing Other Dataset (Race Metrics)
metrics3_df = pd.read_csv("Race Data.csv",keep_default_na=False, na_values=[""])
metrics3_df = metrics3_df.drop(["IndianTotalPerc", "AsianTotalPerc", "HawaiianTotalPerc"], axis = 1)
metrics3_df.head(5)

# Importing Other Dataset (City Latitude)
metrics4_df = pd.read_csv("uscities.csv",keep_default_na=False, na_values=[""])
metrics4_df = metrics4_df.drop(["county_fips", "county_name", "density","source","military","incorporated","timezone","ranking","zips","id"], axis = 1)
metrics4_df.head(5)

# Importing Other Dataset (State Latitude)
metrics5_df = pd.read_csv("us_states_geo_loc.csv",keep_default_na=False, na_values=[""])
metrics5_df[['State','CCode']] = metrics5_df.State.str.split(", ",expand=True,)
metrics5_df = metrics5_df.drop("CCode", axis = 1)
metrics5_df.head(5)

# Merging Dataset
merged_inner = pd.merge(left=trump_df, right=metrics1_df, left_on="City", right_on="City", how="left")
merged_inner.head(5)

merged_inner = pd.merge(left=merged_inner, right=metrics2_df, left_on="State", right_on="Row Labels")
merged_inner = merged_inner.drop("Row Labels", axis = 1)
merged_inner.head(5)

merged_inner = pd.merge(left=merged_inner, right=metrics3_df, on= ["State"], how="inner")
merged_inner.head(5)

merged_inner = pd.merge(left=merged_inner, right=metrics4_df, left_on=["City","State"], right_on=["city_ascii","state_name"])
merged_inner = merged_inner.drop(["city_ascii", "state_name", "state_id"], axis = 1)
merged_inner.head(5)

merged_inner = pd.merge(left=merged_inner, right=metrics5_df, on= ["State"], how="inner")
merged_inner.head(5)

trump_df = merged_inner.drop(merged_inner.index[[7,9,27,28,30,31,33,34]])

# Preprocessing Data
## Tokenization
trump_df["Speech"] = trump_df["Speech"].apply(lambda x: re.sub(r'\[.*?\]','',x))
trump_df["Speech"] = trump_df["Speech"].apply(lambda x: x.lower())
tokenizer = RegexpTokenizer("[a-z][a-z]+[a-z]")

def nltk_tag_to_wordnet_tag(nltk_tag):
    if nltk_tag.startswith("J"):
        return wordnet.ADJ
    elif nltk_tag.startswith("V"):
        return wordnet.VERB
    elif nltk_tag.startswith("N"):
        return wordnet.NOUN
    elif nltk_tag.startswith("R"):
        return wordnet.ADV
    else:          
        return None
    
## Lemmatization
def lemmatize_sentence(sentence):
     tokenizer = RegexpTokenizer('[a-z][a-z]+[a-z]')
     nltk_tagged = nltk.pos_tag(tokenizer.tokenize(sentence))  
     wordnet_tagged = map(lambda x: (x[0], nltk_tag_to_wordnet_tag(x[1])), nltk_tagged)
     lemmatized_sentence = []
     for word, tag in wordnet_tagged:
         if tag is None:
             lemmatized_sentence.append(word)
         else:        
             lemmatized_sentence.append(WordNetLemmatizer().lemmatize(word, tag))
     return " ".join(lemmatized_sentence)
 
trump_df["Speech"] = trump_df["Speech"].apply(lambda x: lemmatize_sentence(x))

## Stopword Removal
trump_df["Speech_No_Stopwords"] = trump_df["Speech"].apply(lambda x: remove_stopwords(x))

# Natural Language Processing
## Linguistic Feature Extractions
### Words Extraction
trump_df["Numbers Of Words"] = trump_df["Speech"].apply(lambda x: len(x.split(" ")))
trump_df["Numbers Of StopWords"] = trump_df["Speech"].apply(lambda x: len([word for word in x.split(" ") if word in list(STOPWORDS)]))
trump_df["Average Word Length"] = trump_df["Speech"].apply(lambda x: np.mean(np.array([len(va) for va in x.split(" ") if va not in list(STOPWORDS)])))
trump_df["Speech"] = trump_df["Speech"].apply(lambda x: re.sub(r"[,.;@#?!&$]+", " ", x))

### Lexical Diversity
def lexical_diversity(text):
    return (len(text) / len(set(text))/10000)
trump_df["Lexical Diversity"] = trump_df["Speech"].apply(lambda x: lexical_diversity(x))

### Named Entities
nlpspacy = sp.load("en_core_web_sm")
trump_df["Numbers of Geo Location Mentioned"] = trump_df["Speech"].apply(lambda x: len([tok for tok in nlpspacy(x).ents if tok.label_ == "GPE" ]))
trump_df["Numbers of Monetary Keywords Mentioned"] = trump_df["Speech"].apply(lambda x: len([tok for tok in nlpspacy(x).ents if tok.label_ == "MONEY" ]))
trump_df["Numbers of Peoples' Name Mentioned"] = trump_df["Speech"].apply(lambda x: len([tok for tok in nlpspacy(x).ents if tok.label_ == "PERSON" ]))

## Sentiment Analysis
### Sentiment Intensity Analyzer
sid = SentimentIntensityAnalyzer()
trump_df["Sentiments"] = trump_df["Speech_No_Stopwords"].apply(lambda x: sid.polarity_scores(x))
trump_df["Positive Sentiment"] = trump_df["Sentiments"].apply(lambda x: x["pos"]) 
trump_df["Neutral Sentiment"] = trump_df["Sentiments"].apply(lambda x: x["neu"])
trump_df["Negative Sentiment"] = trump_df["Sentiments"].apply(lambda x: x["neg"])
trump_df.drop(columns=["Sentiments"],inplace=True)
trump_df.head(5)

### Sentiment Distribution
pink = mpatches.Patch(color='tab:pink', label='Positive Sentiment')
green = mpatches.Patch(color='tab:green', label='Neutral Sentiment')
cyan = mpatches.Patch(color='tab:cyan', label='Negative Sentiment')

plt.figure(figsize=(20,11))
ax = sns.kdeplot(trump_df['Positive Sentiment'],lw=3, label = "Positive Sentiment", color= "tab:pink")
ax = sns.kdeplot(trump_df['Neutral Sentiment'],lw=3, label = "Neutral Sentiment", color= "tab:green")
ax = sns.kdeplot(trump_df['Negative Sentiment'],lw=3, label = "Negative Sentiment", color= "tab:cyan")
ax.set_xlabel('Value',fontsize=20)
ax.set_ylabel('Density',fontsize=20)
ax.set_title('Trumps Speech Sentiments Distribution',fontsize=20,fontweight='bold')
plt.legend(handles=[pink, green, cyan])
plt.show()

## Topic Modelling (Non-Negative Matrix Factorization)
### Terms Vectorizing
cvec_ = CountVectorizer(stop_words ="english")
doc_term = cvec_.fit_transform(list(trump_df['Speech']))
countvec = cvec_.fit(list(trump_df['Speech']))
trump_ft = pd.DataFrame(doc_term.toarray().round(3), index=[i for i in trump_df['City']], columns=cvec_.get_feature_names()).head(10)
trump_ft 

### Fitting NMF Model with Terms Vector
nmf_model = NMF(n_components = 3, random_state=123)
model = nmf_model.fit(doc_term)
trump_topic = model.transform(doc_term)

topic_word = pd.DataFrame(nmf_model.components_.round(3),
             index = ["topic_1","topic_2","topic_3"],
             columns = cvec_.get_feature_names())
topic_word

### Generating Main Topics
def display_topics(model, feature_names, no_top_words, topic_names=None):
    for ix, topic in enumerate(model.components_):
        if not topic_names or not topic_names[ix]:
            print("\nTopic ", ix)
        else:
            print("\nTopic: '",topic_names[ix],"'")
        print(", ".join([feature_names[i]
                        for i in topic.argsort()[:-no_top_words - 1:-1]]))
        
display_topics(nmf_model, cvec_.get_feature_names(), 30)

topic_ = []
feature_ = cvec_.get_feature_names()
for ix, topic in enumerate(nmf_model.components_):
    topic_.append(" ".join([feature_[i] for i in topic.argsort()]))

### Creating Topic Database
topic_df = pd.DataFrame(trump_topic.round(3),
             index = [i for i in trump_df['City']],
             columns = ["topic_1","topic_2","topic_3"])
topic_df.head(5)

topic_df = topic_df.reset_index()

### Merging Topic With Main Dataframe
trump_df = pd.merge(left=trump_df, right=topic_df, left_on="City", right_on="index")
trump_df = trump_df.drop("index", axis = 1)
trump_df.head(4)

### Cleaning Dataframe
trump_df = trump_df.drop(trump_df.index[[7,9,27,28,30,31,33,34]])
trump_df = trump_df.rename(columns={'topic_1':'Achievements','topic_2':'Plans, Vision and Appealing Support','topic_3':'Political Adversaries'})
trump_df.to_csv(r"/rally.csv", index=False)

# Data Visualization
## Word Cloud
### Most Frequent Words Spoken (All Rallies Combined)
topwords_plus = set(STOPWORDS)
stopwords_plus.update(["say", "go", "know", "don"])
most_words_global = " ".join(tmpwords for tmpwords in trump_df.Speech)

wordcloud = WordCloud(max_words=40, stopwords=stopwords_plus, background_color="white").generate(most_words_global)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.tight_layout(pad=0)
plt.title('Most Spoken Words',fontsize=23,fontweight='bold')
plt.show()

### Most Frequent Words Spoken (Per Rallies)
fig,axes = plt.subplots(8,4)
fig.set_figwidth(25)
fig.set_figheight(25)
r,c=0,0
for cit in trump_df.City:
    text = trump_df.query('City == "{}"'.format(cit))
    text = ' '.join(text.Speech.values)
    axes[r][c].imshow(WordCloud(max_words=25, stopwords=stopwords_plus, background_color="white").generate(text))
    axes[r][c].axis('off')
    axes[r][c].set_title('Most Frequent Words Spoken In {}'.format(cit),fontsize=13,fontweight='bold')
    c+=1
    if c == 4:
        c = 0
        r+=1
    if r == 8:
        break
plt.tight_layout()

### Sentiments Wordcloud
#### Positive Sentiment Dictionary
pos_word_list = []
text1 = ' '.join(trump_df.Speech.values)
tokenized_sentence = text1.split(' ')
tokenized_sentence = [w for w in tokenized_sentence if w not in list(stopwords_plus)]
for word1 in tokenized_sentence:
    if (sid.polarity_scores(word1)['compound']) >= 0.1:
        pos_word_list.append(word1)
pos_word_list
pos_word_dict = {word1:len(re.findall(word1,text1)) for word1 in pos_word_list}
top_10_pos = nlargest(10, pos_word_dict, key=pos_word_dict.get)  

#### Negative Sentiment Dictionary
neg_word_list = []
text2 = ' '.join(trump_df.Speech.values)
tokenized_sentence = text.split(' ')
tokenized_sentence = [w for w in tokenized_sentence if w not in list(stopwords_plus)]
for word2 in tokenized_sentence:
    if (sid.polarity_scores(word2)['compound']) <= -0.1:
        neg_word_list.append(word2)
neg_word_dict = {word2:len(re.findall(word2,text2)) for word2 in neg_word_list}       

top_10_neg = nlargest(10, neg_word_dict, key=neg_word_dict.get)

#### Positive Words (Plotting)
plt.figure(figsize=(20,15), facecolor = None)
plt.imshow(WordCloud(width = 800, height = 800,background_color='white').generate(' '.join(nlargest(30, pos_word_dict, key=pos_word_dict.get))))
plt.axis('off')
plt.tight_layout(pad=0)
plt.title('Top 30 Positive Words',fontsize=23,fontweight='bold')
plt.show()

#### Negative Words (Plotting)
plt.figure(figsize=(20,15), facecolor = None)
plt.imshow(WordCloud(width = 800, height = 800,background_color='white').generate(' '.join(nlargest(30, neg_word_dict, key=neg_word_dict.get))))
plt.axis('off')
plt.tight_layout(pad=0)
plt.title('Top 30 Negative Words',fontsize=23,fontweight='bold')
plt.show()



