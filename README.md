# Identifyng Current Trends in Geologic Resarch via Topic Modeling
Corey Solitaire
12.7.2020

## About the Project
This project scrapped data from 1800 of the most recently updated GitHub README pages associated with the dicpiline of Geology to examine current trends in the field.  Using topic modeling 5 trends were identifed in the repositories (list trends).  Trend association became clearer when common words were removed.  Using identified trend clusters dictornays of trending words were crated and scored providing a ranked list of trends for the field.  TFIDF Corpus

## Goals
To identify common trends in GitHub repository users who focus on Geological research

## Background

***Topic modeling*** 

Topic modeling is an unsupervised learning technique that intends to analyze large volumes of text data by clustering the documents into groups. In the case of topic modeling, the text data do not have any labels attached to it. Rather, topic modeling tries to group the documents into clusters based on similar characteristics.

A typical example of topic modeling is clustering a large body of text by clustering the documents into groups. In other words, cluster documents that have the same topic. It is extremely difficult to evaluate the performance of topic modeling since there are no right answers. It depends upon the users domain knowledge to find similar characteristics between the documents of one cluster and assign it an appropriate label or topic.

Two approaches are mainly used for topic modeling: Latent Dirichlet Allocation and Non-Negative Matrix factorization. This investigation will focus in the first.

***Latent Dirichlet Allocation (LDA)***

LDA imagines a fixed set of topics. Each topic represents a set of words. And the goal of LDA is to map all the documents to the topics in a way, such that the words in each document are mostly captured by those imaginary topics.

The LDA is based upon two general assumptions:

- Documents that have similar words usually have the same topic
- Documents that have groups of words frequently occurring together usually have the same topic.

These assumptions make sense because the documents that have the same topic, for instance, Business topics will have words like the "economy", "profit", "the stock market", "loss", etc. The second assumption states that if these words frequently occur together in multiple documents, those documents may belong to the same category.

Mathematically, the above two assumptions can be represented as:

- Documents are probability distributions over latent topics
- Topics are probability distributions over words

***Remove Words with Ginsim Library***

The Gensim library is an extremely useful NLP library for Python. Gensim was primarily developed for topic modeling. However, it now supports a variety of other NLP tasks such as converting words to vectors (word2vec), document to vectors (doc2vec), finding text similarity, and text summarization. I will be using the Gensim library to remove stop words in my corpus, due to its optimization for topic modeling.      

## Deliverables
- Github Repo w/ Final Notebook and README
- Class Presentation + Slide Deck

## Acknowledgments
For more information to Topic Modeling please see [Link to Topic Modeling](https://stackabuse.com/python-for-nlp-topic-modeling/)    
For more information on the Ginsim Library please see [Link to Ginsim Library](https://stackabuse.com/python-for-nlp-working-with-the-gensim-library-part-1/)  

## Data Dictionary
  ---                    ---
| **Terms**             | **Definition**                                                                                                                     |
| ---                   | ---                                                                                                                                |
| document              | A single observation, like the body of an email                                                                                    |
| corpus                | Set of documents, dataset, sample, etc                                                                                             |
| tokenize              | Breaking text up into linguistic units such as words or n-grams                                                                    |
| lemmatize             | Return the base or dictionary form of a word, which is the lemma                                                                   |
| stopwords             | Commonly used word (such as “the”, “a”, “an”, “in”) that are ignored                                                               |
| Beautiful Soup        | A Python library for pulling data out of HTML and XML files                                                                        |
| web scraper           | A data science technique used for extracting data from websites                                                                    |
| programing language   | A set of commands that a computer understands                                                                                      |
| TF                    | Term Frequency; how often a word appears in a document                                                                             |
| IDF                   | Inverse Document Frequency; a measure based on in how many documents will a word appear                                            |
| TF-IDF                | A holistic combination of TF and IDF                                                                                               |
| Topic Modeling        | An unsupervided machine learning technique to analyze large volumes of text data by clustering into groups                         |
| LDA                   | A modeling technique where documents are described by a distribution of topics, and topic are described by a distribution of words |
| NLP                   | Natural language processing is defined as the automatic manipulation of natural language, like speech and text, by software        |
  ---                  ---  
  
## Initial Thoughts & Hypotheses
## Thoughts
## Hypotheses
## Project Steps
## Acquire
## Prepare
## Explore
## Model
## Conclusions
## How to Reproduce
## Steps
## Tools & Requirements
## License
## Creators
