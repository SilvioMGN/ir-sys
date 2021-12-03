import nltk
import math
import json

from porterStemmer import PorterStemmer
from nltk.corpus import stopwords
import nltk
from nltk.tokenize.treebank import TreebankWordDetokenizer

nltk.download('inaugural')

porter = PorterStemmer()

documents = {}
termFrequency = {}
documentFrequency = {}
tfidfdictionary = {}
tokens = [] # unique tokens


def processingQuery(query):

    # we use porter stemmer for the query
    processedQuery = porter.stem(query)

    print(processedQuery)

    return processedQuery


def processingDocument(document):

    global tokens

    tokens_local = document.split()  # we tokenize the document

    # for now we remove all commas, exclamation marks, etc. from tokens
    for i in range(len(tokens_local)):
        tokens_local[i] = tokens_local[i].replace("!", "")
        tokens_local[i] = tokens_local[i].replace(",", "")
        tokens_local[i] = tokens_local[i].replace("?", "")
        tokens_local[i] = tokens_local[i].replace(".", "")

    tokens_local.remove('')
    tokens_local.remove("")

    # we use NLTK library to get common stop words
    stop_words = set(stopwords.words('english'))

    # we remove all stop words from document
    tokens_local = [token for token in tokens_local if not token.lower() in stop_words]

    for i in range(len(tokens_local)):  # stem the tokens of the new document
        tokens_local[i] = porter.stem(tokens_local[i])

    try:   # we add the new document to our document dictionary by giving it an id as key
        # needs optimization
        a = list(documents)[-1]
        b = [int(s) for s in a.split() if s.isdigit()]
        lastKey = b[0] + 1
    except IndexError:
        lastKey = 0

    # adding new document to our document dict
    documents["doc " + str(lastKey)] = tokens_local

    for elem in tokens_local:
        if elem not in tokens:
            tokens.append(elem)

    return tokens


def tf():  # function that calculates term frequency (tf) for all terms for every document -> key = (term, document) -> value = number of occurrences of term in document
    for doc in documents.keys():
        for term in documents[doc]:
            if (term, doc) not in termFrequency.keys():
                termFrequency[term, doc] = 1
            else:
                termFrequency[term, doc] += 1

    return termFrequency


def df():  # function that calculates document frequency (df) for all terms
    # needs update

    for doc in documents.keys():
        for term in documents[doc]:
            if term not in documentFrequency.keys():
                documentFrequency[term] = []
            if doc not in documentFrequency[term]:
                documentFrequency[term].append(doc)

    return documentFrequency


def getDocFreq(term):  # returns document frequency for a specific term
    return len(set(documentFrequency[term]))


def getTermFreq(term, document):  # returns term frequency for a specific term and document
   
    return termFrequency[term, document]


def idf(term):  # function that calculates inverse document frequency (idf) of term t

    numOfDocs = len(documents.keys())
    docFreq = len(set(documentFrequency[term]))

    idf = math.log(numOfDocs/docFreq)

    return idf


def tfidf():  # function that calculates tfidf-weights

    tfidf = 0

    global tfidfdictionary

    for term in tokens:
        for doc in documents:
            if doc in documentFrequency[term]:
                try:
                    tfidf = getTermFreq(term, doc) * idf(term)
                except KeyError:
                    tfidf = 0

                if (term, doc) not in tfidfdictionary.keys():
                        tfidfdictionary[term, doc] = []
                        tfidfdictionary[term, doc].append(tfidf)
                        
    return tfidf 



    def vectorSimilarity(): # function that calculates vector distance
        return ""


# EXAMPLE PROGRAM

for i in nltk.corpus.inaugural.fileids():
    words = nltk.corpus.inaugural.words(i)
    text = TreebankWordDetokenizer().detokenize(words)

    processingDocument(text)

tf()
df()
tfidf()


#query("We the citizens of America are now joined in a great national effort to rebuild our country and restore its promise for all of our people. Together")