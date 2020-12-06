from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import random
import pandas as pd

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
    df['Topic'] = topic_values.argmax(axis=1)
    return df