# Identifying Current Trends in Geologic Resarch Using Topic Modeling
Corey Solitaire   
12.7.2020   

![](https://gloucester-ma.gov/ImageRepository/Document?documentID=6277)

## About the Project
This project utilized web scraping and topic modeling to identify current trends in geologic research. A function was developed that takes in a list of GitHub URL addresses and collects README text data and the repository's primary programming language. Topic modeling was performed using the LatentDirichletAllocation (LDA) class from the sklearn.decomposition library. Five trends were identified in the corpus over the course of two rounds of modeling, with trend association becoming clearer when common words were removed. 

## Goals
To identify common trends in GitHub repository users who focus on Geological research

## Background

***Web scraping:***

Web scraping is the process of collecting structured web data in an automated fashion. In this project, we leveraged the speed of automated data collection to collect data from 500 GitHub repository README texts. To extract the necessary information, I developed a function based on a popular natural language processing (NLP) library known as Beautiful Soup. This tool allowed me to identify and extract information on websites and store them in a large file that made up the body (corpus) of the project.

***Topic modeling:***

Topic modeling is an unsupervised learning technique that intends to analyze large volumes of text data by clustering the documents into groups. In the case of topic modeling, the text data does not have any labels attached to it. Instead, topic modeling tries to group the documents into clusters based on similar characteristics. A typical example of topic modeling is clustering a large body of text by clustering the documents into groups--in other words, cluster documents that have the same topic. It is challenging to evaluate the performance of topic modeling since there are no objectivly right answers. "Correctness" depends on the user's domain knowledge to find similar characteristics between one cluster's documents and assign it an appropriate label or topic.

***Latent Dirichlet Allocation (LDA)***

LDA is one of the two most common approaches used in topic modeling. Using a specified set of “topics” (each topic represents a set of words), LDA maps all the documents to the topics so that those imaginary topics mostly capture the words in each document.

The LDA is based upon two general assumptions:

    Documents which have similar words often share similar topics.
    Documents which have groups of words frequently occurring together usually have the same topic.

These assumptions make sense because the documents that have the same topic. For instance, business topics will commonly have words such as "the economy," "profit," "the stock market," "loss," etc. The second assumption states that if these words frequently occur together in multiple documents, those documents may belong to the same category.

Mathematically, the above two assumptions can be represented as:

    Documents are probability distributions over latent topics.
    Topics are probability distributions over words.

***Remove Words with Ginsim Library***

The Gensim library is a powerful NLP library for Python. Gensim was primarily developed for topic modeling. However, it now supports a variety of other NLP tasks such as converting words to vectors (word2vec), document to vectors (doc2vec), finding text similarity, and text summarization. I will be using the Gensim library to remove stop words in my corpus due to its optimization for topic modeling.    

## Deliverables
- Github repo w/ final notebook and README
- Three min presentation + slide deck

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
## Hypotheses
- Can words be used to identify topics in a body of text?

## Project Steps
1. Create list of words for corpus using feature extraction.
2. Apply LDA.
3. Use LDA to predict (n) topics.
4. Explore topics for correlations.
5. Repeat and tune model.

## Acquire
Using the Beautiful Soup library, the README's of 507 GitHub repositories were scraped. I focused my search exclusively on repositories identified by the keyword 'Geology'. The results of this scrape were saved in a JSON file and were used as the research corpus.

## Prepare
Functions were used to modify the text.  The text's case was reduced to lowercase, special characters were removed, and anything that is not a letter, number, whitespace, or a single quote was replaced with an empty string. The text was then broken up into a list of individual words where the words had been simplified to their basic dictionary roots. Finally, stop words were removed using the Gensim library.   

## Model and Explore

***Topic modeling Round 1: Corpus with only stop words removed***

    In the first round of modeling, topics were hard to observe due to the large volume of common words shared between the documents. One Spanish language library (TypeScript) was removed from the corpus along with ~800 common words. These words were identified using domain knowledge and included common expressions and words associated with technological terms (git, website, module, etc.)

***Topic modeling Round 2: Corpus with extra stop words removed***

    The second round of the model was more successful than the previous and was terminated when the list of the 20 most common words was content-related, and individual word frequencies were topic-specific. This second round of modeling produced five topics that could be generally described as follows:

Topic 0: Seismology
Topic 1: Structural Geology
Topic 2: Economic Geology
Topic 3: Geomorphology
Topic 4: Geochemistry 

## Reflections:

***How did the project go? Did you meet your goal? Why or why not?***

This project ended differently than I thought it would. My goal was to see if it was possible to use NLP to discover topics inside a text body. However, I did not know that this was a sub-field of NLP known as topic modeling. After speaking to Ryan and doing a lot of research, I used a topic modeling library (LDA) to assist me with my goal. If I had known that this project would use a specific machine learning library to map the text into topics, I would have picked a cleaner data source (blog posts/research journals). I was bogged down in cleaning my corpus (removed 800 common words by hand), but I learned a lot about topic modeling in the process.

***What are your next steps?***

The next steps are to scrape more README's and incorporate the repository title as a second corpus. I want to perform LDA on the main body of text and the titles to see to test how well I did at cleaning the data and implementing the machine learning library.

***For each stage of the pipeline, where could you make improvements?***

*Planning*

I knew I wanted to keep working on an NPL project that I had started. It was easy to transition that exploration into this project.

*Acquire*

I misdiagnosed an error in my acquire function that caused me many problems throughout this project. Initially, the function was designed to scrape repositories on the first 500 search pages, but instead it only scrapped the first search page 500 times. This setback cost me a lot of time. Wen I could make the fix a large enough corpus was returned to perform this project.

*Prepare*

I struggled to get my function to remove stop words correctly. Finally, I installed the Gensim library to handle all of my stop words (you can apply Gensim to a body of text and not think about it) and created a function that looped through my list 0f 800 custom stop words that were used in the second round of modeling.

*Explore*

In explore, I needed a function that would provide me with common words for my corpus so they could be removed. Most of my time in explore was spent identifying common words that needed to be removed before topic modeling would be effective.

*Model*

Topic modeling and exploration were combined on this project. The first round of topic modeling produced muddy results and needed lots of time to explore identifying and removing common words before the second round of modeling providing generally identifiable topics.

*Delivery*

Delivery is the part of the pipeline that I am most comfortable with. I want to make sure that I have lots of time to prepare for delivery and lots of time to read up on Topic Modeling. Since it is an NLP technique that we did not learn in class, I want to make sure that I can explain how it works and why I chose to work with it before I present my findings.
