from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import random
import re
import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt

def set_plotting_defaults():
    '''
    sets default sizes for plots
    '''
    # plotting defaults
    plt.rc('figure', figsize=(18, 10))
    plt.style.use('seaborn-whitegrid')
    plt.rc('font', size=16)

def nlp_topic_modeling(df, max_df, min_df, n_components):
    '''
    This function takes a df and several model parameters to return a column with n identfied topcs
    '''
    count_vect = CountVectorizer(max_df=max_df, min_df=min_df, stop_words='english')
    doc_term_matrix = count_vect.fit_transform(df['content_clean'].values.astype('U'))
    LDA = LatentDirichletAllocation(n_components=n_components, random_state=123)
    LDA.fit(doc_term_matrix)
    for i,topic in enumerate(LDA.components_):
        print(f'Top {n_components} words for topic #{i}:')
        print([count_vect.get_feature_names()[i] for i in topic.argsort()[-10:]])
        print('\n')
    topic_values = LDA.transform(doc_term_matrix)
    df['topic'] = topic_values.argmax(axis=1)
    return df

def create_lang_word_list(df):
    '''
    creates a list of words in the readme text by language and removes single letter words
    '''
    # create a list of words for each language category
    t0_words = ' '.join(df[df.topic==0].content_clean)
    t1_words = ' '.join(df[df.topic==1].content_clean)
    t2_words = ' '.join(df[df.topic==2].content_clean)
    t3_words = ' '.join(df[df.topic==3].content_clean)
    t4_words = ' '.join(df[df.topic==4].content_clean)

    # remove single letter words to reduce noise
    t0_words = re.sub(r'\s.\s', '', t0_words)
    t1_words = re.sub(r'\s.\s', '', t1_words)
    t2_words = re.sub(r'\s.\s', '', t2_words)
    t3_words = re.sub(r'\s.\s', '', t3_words)
    t4_words = re.sub(r'\s.\s', '', t4_words)
    return t0_words,t1_words,t2_words,t3_words,t4_words


def create_wordcloud(t0_words, t1_words, t2_words, t3_words, t4_words):
    '''
    creates wordcloud for each language
    '''
    # create bigrams by category
    t0_word_cloud = WordCloud(background_color='white').generate(t0_words)
    t1_word_cloud = WordCloud(background_color='white').generate(t1_words)
    t2_word_cloud = WordCloud(background_color='white').generate(t2_words)
    t3_word_cloud = WordCloud(background_color='white').generate(t3_words)
    t4_word_cloud = WordCloud(background_color='white').generate(t4_words)
    return t0_word_cloud, t1_word_cloud, t2_word_cloud, t3_word_cloud, t4_word_cloud 

def plot_wordcloud(df):
    '''
    creates subplots of bigrams for each category
    '''
    set_plotting_defaults()
    t0_words, t1_words, t2_words, t3_words, t4_words = create_lang_word_list(df)
    t0_word_cloud, t1_word_cloud, t2_word_cloud, t3_word_cloud, t4_word_cloud  = create_wordcloud(t0_words, t1_words, t2_words, t3_words, t4_words)
    
    # plot bigrams
    plt.subplot(3,2,1)
    plt.title("Topic 0 Word Cloud")
    plt.imshow(t0_word_cloud)
    plt.axis('off')

    plt.subplot(3,2,2)
    plt.title("Topic 1 Word Cloud")
    plt.imshow(t1_word_cloud)
    plt.axis('off')

    plt.subplot(3,2,3)
    plt.title("Topic 2 Word Cloud")
    plt.imshow(t2_word_cloud)
    plt.axis('off')

    plt.subplot(3,2,4)
    plt.title("Topic 3 Word Cloud")
    plt.imshow(t3_word_cloud)
    plt.axis('off')

    plt.subplot(3,2,5)
    plt.title("Topic 4 Word Cloud")
    plt.imshow(t4_word_cloud)
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()