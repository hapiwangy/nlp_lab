import gensim
import os
from LDA.LDA import preprocess
current = os.getcwd() + "\\LDA\\model"
lda = gensim.models.LdaModel.load(current)

# 印出model裡面的主題
topics = lda.print_topics(num_words = 10) # 每一類要有幾個字(10)
for i,topic in enumerate(topics):
    print(f"第{i + 1}個主題:{topic}")

# # 將之前的dictionary載入
# dictionary = gensim.corpora.Dictionary.load('dictionary.gensim')
# new_doc ='nasa'
# new_doc = preprocess(new_doc)
# new_doc_bow = dictionary.doc2bow(new_doc)
# print(new_doc_bow)
# print(lda.get_document_topics(new_doc_bow))
