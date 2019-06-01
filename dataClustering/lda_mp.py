import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
from wordcloud import WordCloud
import pickle
import re
import pandas as pd
import numpy as np
from collections import Counter,defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from time import time

from pprint import pprint

# Gensim
import gensim
import gensim.corpora as corpora
from gensim.utils import simple_preprocess
from gensim.models import CoherenceModel


# Plotting tools
import pyLDAvis
import pyLDAvis.gensim  # don't skip this
import matplotlib.pyplot as plt

# Enable logging for gensim - optional
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.ERROR)

import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)
from gensim.test.utils import common_texts
from gensim.corpora.dictionary import Dictionary

texts = pickle.load(open("texts.pickle","rb"))
corpus = pickle.load(open("corpus.pickle","rb"))
id2word = pickle.load(open("id2word.pickle","rb"))

pickle_loc = lambda t:"lda_ml_pickles/lda_mp_{}_topics_{}_songs.pickle".format(t,len(texts))

min_topics = 3
max_topics = 100

topics_to_coherence = {}

for topics in range(min_topics,max_topics+1):
    lda_model_dist = gensim.models.LdaMulticore(corpus=corpus,
                                           id2word=id2word,
                                           num_topics=topics, 
                                           random_state=100,
                                           chunksize=100,
                                           passes=10,
                                           alpha='symmetric',
                                           per_word_topics=True)
    coherence_model_lda_dist = CoherenceModel(model=lda_model_dist, texts=texts, dictionary=id2word, coherence='c_v')
    coherence_lda_dist = coherence_model_lda_dist.get_coherence()
    topics_to_coherence[topics]=coherence_lda_dist
    pickle.dump(lda_model_dist,open(pickle_loc(topics),'wb'))
    pickle.dump(topics_to_coherence,open("lda_ml_pickles/topics_to_coherence.pickle","wb"))
    print("Done with {} topics using {} song records!".format(topics,len(texts)))
print("\n\nwhew, all done! :)")




