#!/usr/bin/env python
# coding: utf-8

# In[65]:


from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn import metrics
from sklearn.metrics import roc_auc_score, accuracy_score
import requests
from bs4 import BeautifulSoup
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import warnings
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from nltk.tokenize import word_tokenize
import re
import nltk
import emoji
import string
from textblob import TextBlob
import langid
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from gensim import models, corpora
from sklearn.model_selection import train_test_split
warnings.filterwarnings('ignore')


# In[66]:


from bs4 import BeautifulSoup
import jsonpickle
import requests
from datetime import datetime, timedelta
from textblob import TextBlob
from productClass import Product


def main():
    baseUrl = "https://www.amazon.in"
    mainCategory = "electronics"
    productCategory = "Samsung SSD"
    pagesToFetch = 51
    productObjectDataset = []

    print("Processing...")
    ## interate over amazon pages where upper limit is a big number as we donts know how many pages there can be
    for i in range(1, pagesToFetch + 1):
        urlToFetch = baseUrl + "/s?k=" + productCategory + "&i=" + mainCategory
        if (i > 1):
            urlToFetch += "&page=" + str(i)
        #endif

        res = requests.get(urlToFetch)

        soup = BeautifulSoup(res.text, 'html.parser')
        content = soup.find_all('a',
                                   class_='a-link-normal a-text-normal',
                                   href=True)

        print("Fetching: " + urlToFetch)

        # breaking the loop if page not found
        if (len(content) == 0):
            print("Nothing found in: " + str(i))
            break
        #endif

        for title in content:
            productUrl = baseUrl + title.get('href')
            productTitle = title.text
            productObject = Product(productTitle, productUrl)

            productObjectDataset.append(productObject)
        #endfor
    #endfor

    for productObject in productObjectDataset:
        reviews = []
        needToReplace = "/product-reviews/"
        for i in range(1, 1000000):
            urlToFetch = extract_url(productObject).replace(
                "/dp/", needToReplace) + "?pageNumber=" + str(i)
            res = requests.get(urlToFetch)
            soup = BeautifulSoup(res.text, 'html.parser')
            content = soup.find_all(
                'span', class_='a-size-base review-text review-text-content')
            if (len(content) == 0):
                break
            #endif

            for title in content:
                reviews.append(title.text.strip())
            #endfor
        #endfor
        productObject.add_reviews(reviews)
        print(
            extract_url(productObject) +
            ": status completed!, review found :" + str(len(reviews)))
    #endfor

    print(len(productObjectDataset))
    jsonProductObjectDataset = jsonpickle.encode(productObjectDataset)
    outputFile = open('filepath.json', 'w')
    outputFile.write(jsonProductObjectDataset)
    outputFile.close()
#enddef


def extract_title(productObject):
    return productObject.title
#enddef


def extract_url(productObject):
    return productObject.url
#enddef


def extract_review_list(productObject):
    return productObject.review_list
#enddef

if __name__ == "__main__":
    main()
#############################################################################
import requests
from bs4 import BeautifulSoup

# links and Headers
HEADERS = ({'User-Agent': 
           'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/44.0.2403.157 Safari/537.36', 
                           'Accept-Language': 'en-US, en;q=0.5'})

# Link to the amazon product reviews
url = 'https://www.amazon.in/Samsung-Internal-Solid-State-MZ-V7S500BW/product-reviews/B07MFBLN7K/ref=cm_cr_arp_d_paging_btm_next_2?ie=UTF8&reviewerType=all_reviews&pageNumber='

review_list = []

def retrieve_reviews(soup):
    # Get only those divs from the website which have a property data-hook and its value is review
    reviews = soup.find_all("div", {'data-hook': "review"})

    # Retrieving through the raw text inside the reviews
    for item in reviews:
        
        review = {
        # Get the title of the review
        'title': item.find("a", {'data-hook': "review-title"}).text.strip(),

        # Get the rating. It will be like 4.5 out of 5 stars. So we have to remove out of 5 stars from it and only keep float value 4.5, 3.4, etc.
        'rating': item.find("i", {'data-hook': "review-star-rating"}).text.replace("out of 5 stars", "").strip(),

        # Get the actual review text 
        'review_text': item.find("span", {'data-hook': "review-body"}).text.strip()
        }

        review_list.append(review)

# Get the page content from amazon
# as we know we have 43 pages to visit and get content from
for pageNumber in range(1, 51):
    raw_text = requests.get(url=url+(str(pageNumber)), headers = HEADERS)
    soup = BeautifulSoup(raw_text.text, 'lxml')
    retrieve_reviews(soup)

for index in range(len(review_list)):
    # Print out all the reviews inside of a reviews_list
    print(f"{index+1})  {review_list[index]}")
    print("")

import csv
import pandas as pd

# Create dataframe out of all the reviews from amazon
reviews_df = pd.DataFrame(review_list)

# Put that dataframe into an excel file
reviews_df.to_excel('samsung.xlsx', index = False)


print("Done.")


# In[67]:


def remove_emojis(text):
    reg = emoji.get_emoji_regexp()
    emoji_free_text = reg.sub(r'', text)
    return emoji_free_text


# Cleaining function
def preprocess(input_text):
    lower_text = review.lower()

    punctuations = '''`!()-[]{};:'"\,<>./?@#$%^&*_~=+°'''

    lower_text = re.sub(r"@[A-Za-z0-9]+", "", lower_text)   # Removes the @mentions from the tweets
    lower_text = re.sub(r"[0-9]+", "", lower_text)          # Removes the Numbers from the tweets

    # tokenization
    tokens = word_tokenize(lower_text)

    stopwords = stopwords.words("english")
    # Removing stopwords
    filtered_text = [word for word in tokens if word not in stopwords]

    # look for empty words or words just made of two letters and remove that
    for token in filtered_text:
        if token == "":
            filtered_text.remove(token)

    filtered_text = ' '.join([word for word in filtered_text])

    clean_text = remove_emojis(filtered_text)

    # Removing punctuations in string 
    # Using loop + punctuation string 
    for ele in clean_text:  
        if ele in punctuations:  
            clean_text = clean_text.replace(ele, "")

    # Removing small words with length less than 3
    clean_text = ' '.join([t for t in clean_text.split() if len(t)>=3])

    return word_tokenize(clean_text)


# In[70]:


reviews = pd.read_excel("samsung.xlsx")
reviews.head()


# In[71]:


reviews.shape


# In[72]:


plt.figure(figsize = (7, 7))
sns.countplot(reviews["rating"])


# In[73]:


rating_count = pd.DataFrame(reviews["rating"].value_counts().reset_index())
rating_count


# In[74]:


explode = [0.05, 0.04, 0, 0.02, 0]
names = ["Rating 5.0", "Rating 4.0", "Rating 1.0", "Rating 3.0", "Rating 2.0"]
plt.figure(figsize = (10, 10))
plt.pie(rating_count["rating"],
        labels = names,
        labeldistance=1.05,
        wedgeprops = { 'linewidth' : 1.5, 'edgecolor' : 'white' },
        explode = explode, 
        autopct = '%.2f%%',
        shadow = True,
        pctdistance = .85,
        textprops = {"fontsize": 14, "color":'w'}, 
        rotatelabels = True,
        radius = 1.3
       )

plt.show()


#  The most given rating to the product is 5.0 and 4.0. We can say here that the product is working fine.

# In[75]:


review_text = list(reviews["review_text"])
review_text[:5]


# In[76]:


reviews_df.shape


# In[77]:


product_review = list(reviews_df["review_text"])


# In[78]:


product_review[0]


# In[79]:


import emoji

def remove_emojis(text):
    reg = emoji.get_emoji_regexp()
    emoji_free_text = reg.sub(r'', text)
    return emoji_free_text


# In[80]:


# Cleaining function
def preprocess(reviews, stopwords):
    cleaned_reviews = []
    for review in reviews:
        lower_text = review.lower()

        punctuations = '''`!()-[]{};:'"\,<>./?@#$%^&*_~=+°'''

        lower_text = re.sub(r"@[A-Za-z0-9]+", "", lower_text)   # Removes the @mentions from the tweets
        lower_text = re.sub(r"[0-9]+", "", lower_text)          # Removes the Numbers from the tweets

        # tokenization
        tokens = word_tokenize(lower_text)

        # Removing stopwords
        filtered_text = [word for word in tokens if word not in stopwords]

        # look for empty words or words just made of two letters and remove that
        for token in filtered_text:
            if token == "":
                filtered_text.remove(token)

        filtered_text = ' '.join([word for word in filtered_text])

        clean_text = remove_emojis(filtered_text)

        # Removing punctuations in string 
        # Using loop + punctuation string 
        for ele in clean_text:  
            if ele in punctuations:  
                clean_text = clean_text.replace(ele, "")

        # Removing small words with length less than 3
        clean_text = ' '.join([t for t in clean_text.split() if len(t)>=3])
        
        cleaned_reviews.append(clean_text)
        
    return cleaned_reviews


# In[81]:


from nltk.corpus import stopwords
stopwords = stopwords.words("english")
len(stopwords)


# #### Call the preprocess function and pass the text string to clean data

# In[82]:


clean_reviews = preprocess(product_review, stopwords)
clean_reviews


# #### Stemming and Lemmatization

# In[83]:


wn_lem = nltk.wordnet.WordNetLemmatizer()
stemmer = nltk.stem.PorterStemmer()
def lemmatization(reviews):
    lemmatized_reviews = []
    
    for review in reviews:
        # Tokenization
        tokens = word_tokenize(review)

        for index in range(len(tokens)):
            tokens[index] = wn_lem.lemmatize(tokens[index])
            tokens[index] = stemmer.stem(tokens[index])

        lemmatized = ' '.join([token for token in tokens])
        lemmatized_reviews.append(lemmatized)

    return lemmatized_reviews


# In[84]:


clean_reviews = lemmatization(clean_reviews)
# 5 reviews from the list
for index in range(5):
    print(f"{index+1})  {clean_reviews[index]}\n")


# ### Frequencies

# In[85]:


from collections import Counter
frequencies = Counter(' '.join([review for review in clean_reviews]).split())
frequencies.most_common(10)


# In[86]:


# Words with least frequency that is 1
singletons = [k for k, v in frequencies.items() if v == 1]
singletons[0:10]


# In[87]:


print(f"Total words used once are {len(singletons)} out of {len(frequencies)}")  # 993 words that have been used only once


# In[88]:


# This function will remove words with less frequencies
def remove_useless_words(reviews, useless_words):
    filtered_reviews = []
    for single_review in reviews:
        tokens = word_tokenize(single_review)

        usefull_text = [word for word in tokens if word not in useless_words]
        usefull_text = ' '.join([word for word in usefull_text])
        filtered_reviews.append(usefull_text)
        
    return filtered_reviews


# In[89]:


# Store a copy so we not need to go back for any mistake
clean_reviews_copy = clean_reviews


# In[90]:


clean_reviews = remove_useless_words(clean_reviews, singletons)
# 5 reviews from the list
for index in range(5):
    print(f"{index+1})  {clean_reviews[index]}\n")


# In[91]:


# count vectoriser tells the frequency of a word.
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(min_df = 1, max_df = 0.9)
X = vectorizer.fit_transform(clean_reviews)
word_freq_df = pd.DataFrame({'term': vectorizer.get_feature_names(), 'occurrences':np.asarray(X.sum(axis=0)).ravel().tolist()})
word_freq_df['frequency'] = word_freq_df['occurrences']/np.sum(word_freq_df['occurrences'])


# In[92]:


word_freq_df = word_freq_df.sort_values(by="occurrences", ascending = False)
word_freq_df.head()


# #### TfidfVectorizer

# In[93]:


from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(stop_words='english', max_df = 0.5, smooth_idf=True)
doc_vec = vectorizer.fit_transform(clean_reviews)
names_features = vectorizer.get_feature_names()
dense = doc_vec.todense()
denselist = dense.tolist()
df = pd.DataFrame(denselist, columns = names_features)

df.head()


# # N-gram

# In[94]:


#Bi-gram
def get_top_n2_words(corpus, n=None):
    vec1 = CountVectorizer(ngram_range=(2,2),  #for tri-gram, put ngram_range=(3,3)
            max_features=2000).fit(corpus)
    bag_of_words = vec1.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in     
                  vec1.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], 
                reverse=True)
    return words_freq[:n]


# In[95]:


top2_words = get_top_n2_words(clean_reviews, n=200) #top 200
top2_df = pd.DataFrame(top2_words)
top2_df.columns=["Bi-gram", "Freq"]
top2_df.head()


# In[96]:


#Bi-gram plot
import matplotlib.pyplot as plt
import seaborn as sns
top20_bigram = top2_df.iloc[0:20,:]
fig = plt.figure(figsize = (10, 5))
plot=sns.barplot(x=top20_bigram["Bi-gram"],y=top20_bigram["Freq"])
plot.set_xticklabels(rotation=45,labels = top20_bigram["Bi-gram"])


# In[97]:


#Tri-gram
def get_top_n3_words(corpus, n=None):
    vec1 = CountVectorizer(ngram_range=(3,3), 
           max_features=2000).fit(corpus)
    bag_of_words = vec1.transform(corpus)
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in     
                  vec1.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], 
                reverse=True)
    return words_freq[:n]


# In[98]:


top3_words = get_top_n3_words(clean_reviews, n=200)
top3_df = pd.DataFrame(top3_words)
top3_df.columns=["Tri-gram", "Freq"]


# In[99]:


top3_df


# In[100]:


#Tri-gram plot
import seaborn as sns
top20_trigram = top3_df.iloc[0:20,:]
fig = plt.figure(figsize = (10, 5))
plot=sns.barplot(x=top20_trigram["Tri-gram"],y=top20_trigram["Freq"])
plot.set_xticklabels(rotation=45,labels = top20_trigram["Tri-gram"])


# # WordCloud

# In[101]:


string_Total = " ".join(clean_reviews)


# In[102]:


#wordcloud for entire corpus
plt.figure(figsize=(20, 20))
from wordcloud import WordCloud
wordcloud_stw = WordCloud(
                background_color= 'black',
                width = 1800,
                height = 1500
                ).generate(string_Total)
plt.imshow(wordcloud_stw)
plt.axis("off")
plt.show()


# #### Singularity and Polarity using the textblob

# In[103]:


from textblob import TextBlob


# In[104]:


# Get Subjectivity of each tweet
def getSubjectivity(tweet):
    return TextBlob(tweet).sentiment.subjectivity

# Get polarity of each tweet
def getPolarity(tweet):
    return TextBlob(tweet).sentiment.polarity


# In[105]:


sentiment_df = pd.DataFrame(clean_reviews, columns=["reviews"])


# In[106]:


sentiment_df["Subjectivity"] = sentiment_df["reviews"].apply(getSubjectivity)
sentiment_df["Polarity"] = sentiment_df["reviews"].apply(getPolarity)


# In[107]:


sentiment_df.head()


# In[108]:


# Funciton to compute Sentiment Analysis 
def getAnalysis(score):
    if score < 0:
        return "Negative"
    
    elif score == 0:
        return "Neutral"
    
    else:
        return "Positive"


# In[109]:


sentiment_df["Analysis"] = sentiment_df["Polarity"].apply(getAnalysis)
sentiment_df.head()


# In[110]:


plt.figure(figsize=(3, 6))
sns.countplot(sentiment_df["Analysis"])


# In[111]:


# All Positive Reviews
pos_rvs = sentiment_df[sentiment_df["Analysis"] == "Positive"].sort_values(by = ["Polarity"])

print("All Positive Reviews are: \n")
for index in range(pos_rvs.shape[0]):
    print(f"{index + 1} )  {pos_rvs.iloc[index, 0]} \n")


# In[112]:


# All Negative Reviews
neg_rvs = sentiment_df[sentiment_df["Analysis"] == "Negative"].sort_values(by = ["Polarity"])

print("All Negative Reviews are: \n")
for index in range(neg_rvs.shape[0]):
    print(f"{index + 1} )  {neg_rvs.iloc[index, 0]} \n")


# In[113]:


token_reviews = []
for review in clean_reviews:
    token_reviews.append(word_tokenize(review))

dictionary = corpora.Dictionary(token_reviews)
dictionary.items()


# In[114]:


dictionary = corpora.Dictionary(token_reviews)
 for key in dictionary:
        print(key, dictionary[key])


# In[115]:


corpus = [dictionary.doc2bow(review) for review in token_reviews]
corpus


# In[116]:


clean_reviews[200]


# In[117]:


corpus[200]


# ### Building a Tfidf model

# In[118]:


tfidf_model = models.TfidfModel(corpus)
corpus_tfidf = tfidf_model[corpus]
corpus_tfidf


# ### LSI Model (Latent Semantic Indexing)

# In[119]:


from gensim.models.lsimodel import LsiModel
from gensim import similarities


# In[120]:


lsi_model = LsiModel(corpus = corpus_tfidf, id2word = dictionary, num_topics = 400)
index = similarities.MatrixSimilarity(lsi_model[corpus])


# ### The function will return 10 similar reviews to a given review

# In[121]:


def text_lsi(new_text, num = 10):
    text_tokens = word_tokenize(new_text)
    new_vec = dictionary.doc2bow(text_tokens)
    vec_lsi = lsi_model[new_vec]
    similars = index[vec_lsi]
    similars = sorted(enumerate(similars), key = lambda item: -item[1])
    
    return [(s, clean_reviews[s[0]]) for s in similars[:num]]


# In[122]:


clean_reviews[100]


# In[123]:


text_lsi(clean_reviews[100])


# # ML Algorithm 

# In[124]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(reviews['review_text'], reviews['rating'],                                                     test_size=0.1, random_state=0)

print('Load %d training examples and %d validation examples. \n' %(X_train.shape[0],X_test.shape[0]))
print('Show a review in the training set : \n', X_train.iloc[10])
X_train,y_train


# In[125]:


def cleanText(raw_text, remove_stopwords=False, stemming=False, split_text=False,              ):
    '''
    Convert a raw review to a cleaned review
    '''
    text = BeautifulSoup(raw_text, 'html.parser').get_text()
    letters_only = re.sub("[^a-zA-Z]", " ", text)
    words = letters_only.lower().split() 
    
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        words = [w for w in words if not w in stops]
        
    if stemming==True:

        stemmer = SnowballStemmer('english') 
        words = [stemmer.stem(w) for w in words]
        
    if split_text==True:
        return (words)
    
    return( " ".join(words))


# In[126]:


X_train_cleaned = []
X_test_cleaned = []

for d in X_train:
    X_train_cleaned.append(cleanText(d))
print('Show a cleaned review in the training set : \n',  X_train_cleaned[10])
    
for d in X_test:
    X_test_cleaned.append(cleanText(d))


# In[127]:


countVect = CountVectorizer() 
X_train_countVect = countVect.fit_transform(X_train_cleaned)
mnb = MultinomialNB()
mnb.fit(X_train_countVect, y_train)


# In[128]:


def modelEvaluation(predictions):
    print ("\nAccuracy {:.4f}".format(accuracy_score(y_test, predictions)))
    print("\nClassification report : \n", metrics.classification_report(y_test, predictions))


# In[129]:



predictions = mnb.predict(countVect.transform(X_test_cleaned))
modelEvaluation(predictions)


# In[130]:


tfidf = TfidfVectorizer(min_df=5)
X_train_tfidf = tfidf.fit_transform(X_train)

# Logistic Regression
lr = LogisticRegression()
lr.fit(X_train_tfidf, y_train)


# In[131]:


feature_names = np.array(tfidf.get_feature_names())
sorted_coef_index = lr.coef_[0].argsort()
print('\nTop 10 features with smallest coefficients :\n{}\n'.format(feature_names[sorted_coef_index[:10]]))
print('Top 10 features with largest coefficients : \n{}'.format(feature_names[sorted_coef_index[:-11:-1]]))


# In[132]:


predictions = lr.predict(tfidf.transform(X_test_cleaned))
modelEvaluation(predictions)


# In[ ]:




