from concurrent.futures import process
from sklearn.datasets import fetch_20newsgroups
from nltk.stem import PorterStemmer
from nltk import *
import gensim
from gensim.test.utils import datapath
import os
import pickle
# 對動詞還原詞性
def lemmtaizaion_stemming(text):
    stemmer = PorterStemmer() # ?
    return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos = 'v'))

def preprocess(text)->list:
    result =[]
    for token in gensim.utils.simple_preprocess(text):
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
            result.append(lemmtaizaion_stemming(token))
    return result

# 進行語料庫的下載和分類
# 下載完成後是屬於bunch的資料型態，繼承自dict，所以要用dict的方法去做存取
newsgroups_train = fetch_20newsgroups(subset='train', shuffle = True)
newsgroups_test = fetch_20newsgroups(subset='test', shuffle = True)
# 透過下列指令將分類好的新聞主題印出
# print(list(newsgroups_train.target_names))


if __name__ == '__main__':
    # 進行文本的pre-process
    # 使用nltk和genism來執行
    processed_docs = []
    for y in newsgroups_train['data']:
        processed_docs.append(preprocess(y))

    # 將結果轉換為bag-of-word
    dictionary = gensim.corpora.Dictionary(processed_docs)
    bow_corpus = [dictionary.doc2bow(doc) for doc in processed_docs]

    # 運行LDA
    lda_model = gensim.models.LdaMulticore(bow_corpus,
    num_topics = 8,# 假設分成8個主題
    id2word = dictionary,
    passes = 10,
    workers = 2 
    )
    # 若使用下面這種方法的話會把model存放到gensim下面的目錄
    # 我的話是C:\Users\user\AppData\Local\Programs\Python\Python39\lib\site-packages\gensim\test\test_data\model
    # temp_file = datapath("model")
    # lda_model.save(temp_file)
    # 如果要存到自己的目錄的話就是用model.save(路徑)
    current = os.getcwd() + "\\LDA\\model"
    lda_model.save(current)
    pickle.dump(bow_corpus,open('corpus.pkl','wb+')) 
    dictionary.save('dictionary.gensim')
    