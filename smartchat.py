from newspaper import Article
import random
import string
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import warnings
import ssl
warnings.filterwarnings('ignore')

# Resolves certificate verification error
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

# Download punkt package
nltk.download('punkt', quiet=True)

# Get article
article = Article(
    'https://www.cdc.gov/stroke/signs_symptoms.htm')
article.download()
article.parse()
article.nlp()
corpus = article.text

# Print text from article
print(corpus)