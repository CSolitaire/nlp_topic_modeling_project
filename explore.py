from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import random
import re
import pandas as pd
from wordcloud import WordCloud
import matplotlib.pyplot as plt


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

def get_count_word_freq(t0_words,t1_words,t2_words,t3_words,t4_words):
    '''
    split the list of words and get frequency count
    '''
    # get the count of words by category
    t0_freq = pd.Series(t0_words.split()).value_counts()
    t1_freq = pd.Series(t1_words.split()).value_counts()
    t2_freq = pd.Series(t2_words.split()).value_counts()
    t3_freq = pd.Series(t3_words.split()).value_counts()
    t4_freq = pd.Series(t4_words.split()).value_counts()
    return t0_freq, t1_freq, t2_freq, t3_freq, t4_freq

def create_df_word_counts(t0_freq, t1_freq, t2_freq, t3_freq, t4_freq):
    '''
    combines the frequencies to create new dataframe, word_counts
    '''
    # combine list of word counts into df for further exploration
    word_counts = (pd.concat([t0_freq, t1_freq, t2_freq, t3_freq, t4_freq], axis=1, sort=True)
                .set_axis(['t0', 't1', 't2', 't3', 't4'], axis=1, inplace=False)
                .fillna(0)
                .apply(lambda s: s.astype(int))
                )
    # create a column of all words as well
    word_counts['all_words'] = word_counts['t0'] + word_counts['t1'] + word_counts['t2'] + word_counts['t3'] + word_counts['t4']
    return word_counts


def word_counts_proportion(word_counts):
    '''
    compute proportion of each string that for each language
    '''
    # add columns for each langauge proportion
    word_counts['prop_t0'] = word_counts['t0']/word_counts['all_words']
    word_counts['prop_t1'] = word_counts['t1']/word_counts['all_words']
    word_counts['prop_t2'] = word_counts['t2']/word_counts['all_words']
    word_counts['prop_t3'] = word_counts['t3']/word_counts['all_words']
    word_counts['prop_t4'] = word_counts['t4']/word_counts['all_words']
    return word_counts


def proportion_visualization(word_counts):
    '''
    creates a plot that shows the proportion of the top 20 words by language
    '''
    ## visualize the % of the term in each language
    plt.figure(figsize=(12,8))
    (word_counts
    .assign(p_t0=word_counts.t0 / word_counts['all_words'],
            p_t1=word_counts.t1 / word_counts['all_words'],
            p_t2=word_counts.t2 / word_counts['all_words'],
            p_t3=word_counts.t3 / word_counts['all_words'],
            p_t4=word_counts.t4 / word_counts['all_words']
            )
    .sort_values(by='all_words')
    [['p_t0', 'p_t1', 'p_t2', 'p_t3', 'p_t4']]
    .tail(20)
    .sort_values('p_t0')
    .plot.barh(stacked=True))

    plt.legend(bbox_to_anchor=(1.05, 1))
    plt.title('Proportion of Topics for the 20 most common words')
    plt.show()

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
    def set_plotting_defaults():
        '''
        sets default sizes for plots
        '''
        # plotting defaults
        plt.rc('figure', figsize=(18, 10))
        plt.style.use('seaborn-whitegrid')
        plt.rc('font', size=16)

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