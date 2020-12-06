import pandas as pd
import numpy as np
import datetime as dt
from requests import get
from bs4 import BeautifulSoup
import os
import time
import unicodedata
import re
import json
import nltk
from nltk.tokenize.toktok import ToktokTokenizer
from nltk.corpus import stopwords
import ngrams
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from gensim.parsing.preprocessing import remove_stopwords



##################################################### Acquire #####################################################

def make_soup(url):
    '''
    This helper function takes in a url and requests and parses HTML
    returning a soup object.
    '''
    # set headers and response variables
    headers = {'User-Agent': 'Codeup Data Science'} 
    response = get(url, headers=headers)
    # use BeartifulSoup to make object
    soup = BeautifulSoup(response.text, 'html.parser')
    return soup

def github_geology_urls():
    '''
    This function scrapes all of the urls from
    the github search page and returns a list of the most recently updated Geology urls.
    '''
    # get the first 500 pages to allow for those that don't have readme or language
    pages = range(1, 500)
    urls = []
    
    for p in pages:
        
        # format string of the base url for the main github search page we are using to update with page number
        url = f'https://github.com/search?%7Bp%7Do=desc&p={p}&q=geology&s=updated&type=Repositories'   

        # Make request and soup object using helper
        soup = make_soup(url)

        # Create a list of the anchor elements that hold the urls on this search page
        page_urls_list = soup.find_all('a', class_='v-align-middle')
        # for each url in the find all list get just the 'href' link
        page_urls = {link.get('href') for link in page_urls_list}
        # make a list of these urls
        page_urls = list(page_urls)
        # append the list from the page to the full list to return
        urls.append(page_urls)
        time.sleep(5)
    # flatten the urls list
    urls = [y for x in urls for y in x]
    return urls

def get_geo_results(cached=False):
    '''
    This function with default cached == False does a fresh scrape of github pages returned from
    search of 'environmental' and writes the returned df to a json file.
    cached == True returns a df read in from a json file.
    '''
    # option to read in a json file instead of scrape for df
    if cached == True:
        df = pd.read_json('readgeo.json')
        
    # cached == False completes a fresh scrape for df    
    else:
        # get url list
        url_list = github_geology_urls()

        # Set base_url that will be used in get request
        base_url = 'https://github.com'
        
        # List of full url needed to get readme info
        readme_url_list = []
        for url in url_list:
            full_url = base_url + url
            readme_url_list.append(full_url)
        
        # Create an empty list, readmes, to hold our dictionaries
        readmes = []

        for readme_url in readme_url_list:
            # Make request and soup object using helper
            soup = make_soup(readme_url)

            if soup.find('article', class_="markdown-body entry-content container-lg") != None:            
                # Save the text in each readme to variable text
                content = soup.find('article', class_="markdown-body entry-content container-lg").text
            
            if soup.find('span', class_="text-gray-dark text-bold mr-1") != None:
            # Save the first language in each readme to variable text
                # NOTE: this is the majority language, not all of the languages used
                language = soup.find('span', class_="text-gray-dark text-bold mr-1").text

                # anything else useful on the page?

                # Create a dictionary holding the title and content for each blog
                readme = {'language': language, 'content': content}

                # Add each dictionary to the articles list of dictionaries
                readmes.append(readme)
            
        # convert our list of dictionaries to a df
        df = pd.DataFrame(readmes)

        # Write df to a json file for faster access
        df.to_json('readgeo.json')

    return df

##################################################### Prepare #####################################################

def basic_clean(text):
    '''
    Initial basic cleaning/normalization of text string
    '''
    # change to all lowercase
    low_case = text.lower()
    # remove special characters, encode to ascii and recode to utf-8
    recode = unicodedata.normalize('NFKD', low_case).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    # Remove numbers from text
    cleaned = re.sub(r'\d+', '', recode)
    # Replace anything that is not a letter, number, whitespace or a single quote
    cleaned = re.sub(r"[^a-z0-9'\s]", '', cleaned)
    return cleaned

def tokenize(text):
    '''
    Use NLTK TlktokTokenizer to seperate/tokenize text
    '''
    # create the NLTK tokenizer object
    tokenizer = nltk.tokenize.ToktokTokenizer()
    return tokenizer.tokenize(text, return_str=True)

def stem(text):
    '''
    Apply NLTK stemming to text to remove prefix and suffixes
    '''
    # Create the nltk stemmer object, then use it
    ps = nltk.porter.PorterStemmer()
    stems = [ps.stem(word) for word in text.split()]
    article_stemmed = ' '.join(stems)
    return article_stemmed

def lemmatize(text):
    '''
    Apply NLTK lemmatizing to text to remove prefix and suffixes
    '''
    # Create the nltk lemmatize object, then use it
    wnl = nltk.stem.WordNetLemmatizer()
    lemmas = [wnl.lemmatize(word) for word in text.split()]
    article_lemmatized = ' '.join(lemmas)
    return article_lemmatized

def remove_extra_words(text):
    '''
    Removes stopwords from text, allows for additional words to exclude, or words to not exclude
    '''
    # define additional words
    extra_words=[]
    words = text.split()
    # filter the words
    filtered_words = [w for w in words if w not in extra_words]
    # produce string without stopwords
    article_without_stopwords = ' '.join(filtered_words)
    return article_without_stopwords

def prep_data(df, column):
    '''
    This function take in a df and the string name for a text column with 
    option to pass lists for extra_words and exclude_words and
    returns a df with the text article title, original text, stemmed text,
    lemmatized text, cleaned, tokenized, & lemmatized text with stopwords removed.
    '''
 
    df['content_clean'] = df[column].apply(basic_clean)\
                                    .apply(tokenize)\
                                    .apply(lemmatize)\
                                    .apply(remove_stopwords)\
                                    .apply(remove_extra_words)

    # add a column with a list of words
    words = [re.sub(r'([^a-z0-9\s]|\s.\s)', '', doc).split() for doc in df.content_clean]

    # column name will be words, and the column will contain lists of the words in each doc
    df = pd.concat([df, pd.DataFrame({'words': words})], axis=1)

    # add column with number of words in readme content
    df['doc_length'] = [len(wordlist) for wordlist in df.words]
    
    # Adds column with bigrams and trigrams
    df['bigrams'] =  df['text_filtered'].apply(lambda row: list(nltk.bigrams(row.split(' '))))
    df['trigrams'] =  df['text_filtered'].apply(lambda row: list(nltk.trigrams(row.split(' '))))
    
    # removing non-english languages 
    # language_list = ['JavaScript', 'R', 'Jupyter Notebook','Python','TypeScript']
    # df = df[df.language.isin(language_list)]
    
    # Specify dataframe content
    df = df[['language','content','content_clean','doc_length','words','bigrams','trigrams']]
    return df

    ####################################### Train, Validate, Test ######################################

def train_validate_test(df):
    
    train_validate, test = train_test_split(df[['language', 'content_clean', 'words', 'doc_length']], 
                                            random_state = 123,
                                            stratify=df.language, 
                                            test_size=.2)

    train, validate = train_test_split(train_validate, 
                                       random_state = 123,
                                       stratify=train_validate.language, 
                                       test_size=.25)
    return train, validate, test