#!/usr/bin/env python
# coding: utf-8

# In[ ]:


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


# In[ ]:


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

