import nltk

from porterStemmer import PorterStemmer
from nltk.corpus import stopwords

documents = {}
porter = PorterStemmer() 

def processingQuery(query):

    processedQuery = porter.stem(query) # we use porter stemmer for the query

    print(processedQuery)

    return processedQuery

def processingDocument(document):

    tokens = document.split() # we tokenize the document
    
    for i in range(len(tokens)): # for now we remove all commas, exclamation marks, etc. from tokens
        tokens[i] = tokens[i].replace("!", "")
        tokens[i] = tokens[i].replace(",", "")
        tokens[i] = tokens[i].replace("?", "")
        tokens[i] = tokens[i].replace(".", "")

    stop_words = set(stopwords.words('english')) # we use NLTK library to get common stop words 

    tokens = [token for token in tokens if not token.lower() in stop_words] # we remove all stop words from document

    for i in range(len(tokens)): # stem the tokens of the new document
        tokens[i] = porter.stem(tokens[i])

    try:   # we add the new document to our document dictionary by giving it an id as key 
        lastKey = list(documents)[-1]
    except IndexError:
        lastKey = 0

    documents["doc" + str(lastKey)] = tokens # adding new document to our document dict

    return tokens


def tf(): # function that calculates term frequency (tf)
    return ""

def df(): # function that calculates document frequency (df)
    return ""

def tfidf(): # function that calculates tfidf-weights
    return ""

def vectorSimilarity(): # function that calculates vector distance
    return ""


'''
We can try w2v, standard approach, BoW, etc. to compare the effectivity 
'''
    




#processingDocument("Hello, my name is Josh!")
#processingDocument("Tokenization is often used to protect credit card data, bank account information and other sensitive data handled by a payment processor. Payment processing use cases that tokenize sensitive credit card information include")


