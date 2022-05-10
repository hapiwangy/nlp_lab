import nltk
# 先引入文本
file_text = ""
with open("example.txt","r+", encoding="utf-8") as fp:
    file_text += fp.read()
# print(file_text)

# 斷句
sentence = nltk.sent_tokenize(file_text)
# print(sentence)

# 斷詞
tokens = [nltk.tokenize.word_tokenize(sent) for sent in sentence]
# for token in tokens:
#     print(token)

# POS
pos = [nltk.pos_tag(token) for token in tokens]
# print(pos)

# lemmatization(先簡化詞性再還原)
# 這裡把標點符號歸類為n
wordnet_pos = []
for p in pos:
    for word, tag in p:
        if tag.startswith('J'):
            wordnet_pos.append(nltk.corpus.wordnet.ADJ)
        elif tag.startswith('V'):
            wordnet_pos.append(nltk.corpus.wordnet.VERB)
        elif tag.startswith('N'):
            wordnet_pos.append(nltk.corpus.wordnet.NOUN)
        elif tag.startswith('R'):
            wordnet_pos.append(nltk.corpus.wordnet.ADV)
        else:
            wordnet_pos.append(nltk.corpus.wordnet.NOUN)
lemmatizer = nltk.stem.wordnet.WordNetLemmatizer()
tokens = [lemmatizer.lemmatize(p[n][0], pos=wordnet_pos[n]) for p in pos for n in range(len(p))]
# for token in tokens:
#     print(token)

# stopwords
# 引入nltk的stopwords表，再用for迴圈把非停用詞存起來就好了
nltk_stopwords = nltk.corpus.stopwords.words("english")
tokens = [token for token in tokens if token not in nltk_stopwords]
# for token in tokens:
#     print(token)

# NER(命名實體辨識)
# 把n和屬於人名、地名...
nltk.download('words')
nltk.download('maxent_ne_chunker')
ne_chunked_sents = [nltk.ne_chunk(tag) for tag in pos]
named_entities = []

for ne_tagged_sentence in ne_chunked_sents:
    for tagged_tree in ne_tagged_sentence:
        if hasattr(tagged_tree, 'label'):
            entity_name = ' '.join(c[0] for c in tagged_tree.leaves())
            entity_type = tagged_tree.label()
            named_entities.append((entity_name, entity_type))
            named_entities = list(set(named_entities))
for ner in named_entities:
    print(ner)

