from retrievalSystem import RetrievalSystem

system = RetrievalSystem()


#EXAMPLE PROGRAM

system.processingDocument("Hello, my name is Josh and I own a new credit card!")
system.processingDocument("What is up my friend?")
system.processingDocument("Tokenization is often used to protect credit card data, bank account information and other sensitive data handled by a payment processor. Payment processing use cases that tokenize sensitive credit card information include")

system.tf()
system.df()
system.idf('card')
system.tfidf('inform', 'doc 2')
