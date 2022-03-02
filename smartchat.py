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
# print(corpus)

# Tokenization
text = corpus
sentence_list = nltk.sent_tokenize(text) # sent_tokenize splits doc into sentences

print(sentence_list)

def greeting(text):
    text = text.lower()
    
    bot_greetings = ['hello', 'hi', 'hey', 'hey there', 'greetings']
    
    user_greetings = ['hi', 'hello', 'hey']
    
    for word in text.split():
        if word in user_greetings:
            return random.choice(bot_greetings)
     
def index_sort(lst):
    length = len(lst)
    list_index = list(range(length))
    
    x = lst
    for i in range(length):
        for j in range(length):
            if x[list_index[i]] > x[list_index[j]]:
                list_index[i], list_index[j] = list_index[j], list_index[i]
                
    return list_index
   
def bot_response(user_input):
    user_input = user_input.lower()
    sentence_list.append(user_input)
    bot_response = ''
    cm = CountVectorizer().fit_transform(sentence_list) # transform list into count matrix
    similarity_scores = cosine_similarity(cm[-1], cm) # compare last sentence of user input to entire cm
    similarity_scores_list = similarity_scores.flatten()
    index = index_sort(similarity_scores_list) # calculate where highest values are in similarity scores list
    index = index[1:] # include values that are not itself
    response_flag = 0
    
    j = 0
    for i in range(len(index)):
        if similarity_scores_list[index[i]] > 0.0: # found a similarity
            bot_response += ' ' + sentence_list[index[i]]
            response_flag = 1
            j += 1
        if j > 2:
            break
        
    if response_flag == 0:
        bot_response += ' ' + "I apologize, I don't understand."
        
    sentence_list.remove(user_input)
    
    return bot_response
    