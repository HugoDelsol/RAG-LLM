import spacy
import json
import numpy as np

with open('corpus.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

nlp = spacy.load('fr_core_news_md')
textes = [item['texte'] for item in data] 

print(textes)

docs_test = nlp(textes[0])

docs = nlp.pipe(textes)

print(docs)
# vectors  = [doc.vector for doc in docs]
# np.save('embeddings_spacy.npy', vectors)

contenu_spacy = np.load('embeddings_spacy.npy')

print(contenu_spacy.shape)



for index, doc in enumerate(docs):
    
    nb_tokens = 0
    is_stop = 0
    is_punct = 0
    has_vector = 0

    for token in doc:

        nb_tokens += 1

        if (token.is_stop):
            is_stop += 1
        if (token.is_punct):
            is_punct += 1
        if (token.has_vector):
            has_vector += 1

    print(f"Phrase: {index} | {nb_tokens} tokens | {is_stop} vides | {is_punct} ponctuation | {has_vector} vectors")