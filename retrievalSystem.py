import nltk
import math

from porterStemmer import PorterStemmer
from nltk.corpus import stopwords

documents = {}
termFrequency = {}
documentFrequency = {}
porter = PorterStemmer() 
tokens = []

class RetrievalSystem:

    def processingQuery(self, query):

        processedQuery = porter.stem(query) # we use porter stemmer for the query

        print(processedQuery)

        return processedQuery

    def processingDocument(self, document):

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
            # needs optimization
            a = list(documents)[-1]
            b = [int(s) for s in a.split() if s.isdigit()]
            lastKey = b[0] + 1
        except IndexError:
            lastKey = 0

        documents["doc " + str(lastKey)] = tokens # adding new document to our document dict

        return tokens


    def tf(self): # function that calculates term frequency (tf) for all terms for every document -> key = (term, document) -> value = number of occurrences of term in document

        for doc in documents.keys():
            for term in documents[doc]:
                if (term, doc) not in termFrequency.keys():
                    termFrequency[term, doc] = 1
                else:
                    termFrequency[term, doc] += 1
        return termFrequency


    def df(self): # function that calculates document frequency (df) for all terms
                  # needs update

        for doc in documents.keys():
            for term in documents[doc]:
                if term not in documentFrequency.keys():
                    documentFrequency[term] = []
                documentFrequency[term].append(doc)
            
        return documentFrequency

    def getDocFreq(self, term): # returns document frequency for a specific term
        return len(set(documentFrequency[term]))

    def getTermFreq(self, term, document): # returns term frequency for a specific term and document
        return termFrequency[term, document]

    def idf(self, term): # function that calculates inverse document frequency (idf) of term t

        numOfDocs = len(documents.keys())
        docFreq = len(set(documentFrequency[term]))

        idf = math.log(numOfDocs/docFreq)

        return idf


    def tfidf(self, term, document): # function that calculates tfidf-weights
        return self.getTermFreq(term, document) * self.idf(term)


    '''
    def vectorSimilarity(): # function that calculates vector distance
        return ""


    We can try w2v, standard approach, BoW, etc. to compare the effectivity 
    '''
    