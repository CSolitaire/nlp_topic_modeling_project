# Project Title
## About the Project
### Goals
### Background

**Topic modeling** 

Topic modeling is an unsupervised learning technique that intends to analyze large volumes of text data by clustering the documents into groups. In the case of topic modeling, the text data do not have any labels attached to it. Rather, topic modeling tries to group the documents into clusters based on similar characteristics.

A typical example of topic modeling is clustering a large body of text by clustering the documents into groups. In other words, cluster documents that have the same topic. It is extremely difficult to evaluate the performance of topic modeling since there are no right answers. It depends upon the users domain knowledge to find similar characteristics between the documents of one cluster and assign it an appropriate label or topic.

Two approaches are mainly used for topic modeling: Latent Dirichlet Allocation and Non-Negative Matrix factorization. This investigation will focus in the first.

**Latent Dirichlet Allocation (LDA)**

LDA imagines a fixed set of topics. Each topic represents a set of words. And the goal of LDA is to map all the documents to the topics in a way, such that the words in each document are mostly captured by those imaginary topics.

The LDA is based upon two general assumptions:

- Documents that have similar words usually have the same topic
- Documents that have groups of words frequently occurring together usually have the same topic.

These assumptions make sense because the documents that have the same topic, for instance, Business topics will have words like the "economy", "profit", "the stock market", "loss", etc. The second assumption states that if these words frequently occur together in multiple documents, those documents may belong to the same category.

Mathematically, the above two assumptions can be represented as:

- Documents are probability distributions over latent topics
- Topics are probability distributions over words

**Remove Words with Ginsim Library**

The Gensim library is an extremely useful NLP library for Python. Gensim was primarily developed for topic modeling. However, it now supports a variety of other NLP tasks such as converting words to vectors (word2vec), document to vectors (doc2vec), finding text similarity, and text summarization. I will be using the Gensim library to remove stop words in my corpus, due to its optimization for topic modeling.  


### Deliverables
### Acknowledgments
## Data Dictionary
|   |   |   |   |   |
|---|---|---|---|---|
|   |   |   |   |   |
|   |   |   |   |   |
|   |   |   |   |   |
## Initial Thoughts & Hypotheses
### Thoughts
### Hypotheses
## Project Steps
### Acquire
### Prepare
### Explore
### Model
### Conclusions
## How to Reproduce
### Steps
### Tools & Requirements
## License
## Creators
