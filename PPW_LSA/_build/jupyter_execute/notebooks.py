#!/usr/bin/env python
# coding: utf-8

# # Latent Semantic Analysis (LSA)

# # Crawling Data Berita

# sebelum melakukan proses crawling data, pastikan anda sudah menginstall library Scrapy dari python. Jika anda belum menginstall Scrapy anda dapat menginstall nya dengan cara ketikkan "pip install Scrapy" pada cmd

# ## Crawling pertama

# pada proses crawling yang pertama ini, kita akan mengambil link yang ada pada halaman kumpulan judul berita. cara untuk melakukan crawling adalah:
# 1. buat file python (.py) misalkan "crawling1.py".
# 2. copy paste code yang ada dibawah ini. (anda dapat memodifikasi kode ini sesuai dengan link berita yang anda inginkan).
# 3. jalankan file "crawling1.py" dengan cara mengetikkan "scrapy runspider crawling1.py -O link.csv" , untuk yang bagian "link.csv" ini merupakan output file yang anda crawling, karena disini saya menggunakan contoh "link.csv" maka hasil outputnya dalam bentuk file csv.

# In[1]:


import scrapy


class QuotesSpider(scrapy.Spider):
    name = "quotes"

    def start_requests(self):

        arrayData = []
        for i in range(1, 6):
            inArray = 'https://indeks.kompas.com/?site=news&page=' + str(i)
            arrayData.append(inArray)
        for url in arrayData:
            yield scrapy.Request(url=url, callback=self.parse)

    def parse(self, response):
        for i in range(1,16):
            yield {
                'link': response.css('body > div.wrap > div.container.clearfix > div:nth-child(3) > div.col-bs10-7 > div.latest--indeks.mt2.clearfix > div:nth-child(' + str(i) +') > div.article__list__title > h3 > a::attr(href)').extract(),
            }


# ## Crawling kedua

# Untuk proses crawling yang kedua ini, saya mengambil link website berita hasil dari crawling pertama yang sudah di export dalam bentuk csv. untuk membaca file csv ini saya menggunakan library pandas. lalu setelah file dibaca, saya masukkan kedalam array. setelah itu masing masing link akan dilakukan proses crawling.
# Pada proses cawling kedua ini kita akan menuju website beritanya langsung, untuk mendapatkan data judul, label dan isi dari masing-masing berita.
# jalankan file ini dengan cara yang sama seperti yang pertama, akan tetapi sesuaikan nama filenya. cnothnya seperti "scrapy runspider crawling2.py -O isi_berita.csv"

# In[2]:


import scrapy
import pandas as pd


class QuotesSpider(scrapy.Spider):
    name = "quotes"

    def start_requests(self):
        dataCSV = pd.read_csv('link.csv')
        dataCSV.head()
        indexData = dataCSV.iloc[:, [0]].values
        arrayData = []
        for i in indexData:
            ambil = i[0]
            arrayData.append(ambil)
        print(arrayData)


        for url in arrayData:
            yield scrapy.Request(url=url, callback=self.parse)

    def parse(self, response):
        yield {
            'judul': response.css('body > div.wrap > div.container.clearfix > div:nth-child(3) > div > h1::text').extract(),
            'label': response.css('body > div.wrap > div.container.clearfix > div:nth-child(3) > div > h3 > ul > li:nth-child(3) > a > span::text').extract(),
            'isi': response.css('body > div.wrap > div.container.clearfix > div.row.col-offset-fluid.clearfix.js-giant-wp-sticky-parent > div.col-bs10-7.js-read-article > div.read__article.mt2.clearfix.js-tower-sticky-parent > div.col-bs9-7 > div.read__content > div > p::text').extract(),
           
        }


# # Latent Semantic Analysis (LSA)

# sebelum kita berpindah ke LSA, ada beberapa hal yang perlu dipersiapkan terlebih dahulu.
# beberapa library yang perlu di siapkan yaitu nltk, pandas, numpy dan scikit-learn.
# jika anda menggunakan google colab anda bisa mengetikan syntax dibawah ini untuk melakukan instalasi library yang dibutuhkan.
# 
# !pip install nltk <br>
# !pip install pandas <br>
# !pip install numpy <br>
# !pip install scikit-learn <br>
# 

# ## preprocessing data

# ### import libray
# 
# import library yang dibutuhkan untuk preprocessing data

# In[3]:


# import library
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk import word_tokenize
import numpy as np


# export file "isi_berita.csv" dalam bentuk data frame pandas.

# In[4]:


#import data frame
dataCSV = pd.read_csv('isi_berita.csv')
dataCSV.head()


# ### Cleansing dan Stopword
# disini kita melakukan cleansing data, yang artinya kita membersihkan data dari simbol, angka dan spasi. <br>
# lalu untuk stopword ini untuk membuang kata yang tidak mempunyai makna seperti:
# 1. "dan"
# 2. "yang" 
# 3. "atau"
# 4. "adalah"

# In[5]:


# cleansing & stopword
index_iloc = 0
len_df = len(dataCSV.index)
array_stopwords = []
for kata in range(len_df):
    # indexData itu ambil tiap bagian dari data frame dengan nama dataCSV
    indexData = dataCSV.iloc[index_iloc, [2]].values
    clean_words = []
    for w in word_tokenize(indexData[0].lower()):
        if w.isalpha() and w not in stopwords.words('indonesian'):
            clean_words.append(w)
    array_stopwords.append(clean_words)
    index_iloc += 1

# membuat kata-kata 1 dokumen di list yang sama
NewArray_stopwords = []
for j in array_stopwords:
    # proses stem per kalimat
    temp = ""
    for i in j:
        # print(i)
        temp = temp +" "+ i

    NewArray_stopwords.append(temp)
print(NewArray_stopwords[0])


# diatas ini adalah contoh isi dari salah satu berita yang sudah dilakukan cleansing dan stopword.

# dibawah ini adalah proses memasukkan data yang sudah dilakukan preprocessing ke dalam data frame yang mempunyai nama "dataSCV"

# In[6]:


dataCSV = dataCSV.drop('isi', axis=1)
dataCSV = dataCSV.drop('judul', axis=1)
dataCSV = dataCSV.drop('label', axis=1)
dataCSV['isi_berita_final'] = np.array(NewArray_stopwords)
dataCSV.head()


# ## Term Frequency - Inverse Document Frequency (TF-IDF)

# setelah melakukan pre-processing data, selanjutnya dilakukan proses TF-IDF <br>
# TF-IDF adalah suatu metode algoritma untuk menghitung bobot setiap kata di setiap dokumen dalam korpus. Metode ini juga terkenal efisien, mudah dan memiliki hasil yang akurat. <br>
# Term Frequency (TF) merupakan jumlah kemunculan kata pada setiap dokumen. dirumuskan dengan jumlah frekuensi kata terpilih / jumlah kata <br>
# Inverse Document Matrix (IDF) dirumuskan dengan log((jumlah dokumen / jumlah frekuensi kata terpilih). <br>
# untuk menghasilkan TF-IDF maka hasil dari TF dikalikan dengan IDF, seperti rumus dibawah ini:
# 
# $$
# W_{i, j}=\frac{n_{i, j}}{\sum_{j=1}^{p} n_{j, i}} \log _{2} \frac{D}{d_{j}}
# $$
# 
# Dengan:
# 
# $
# {W_{i, j}}\quad\quad\>: \text { pembobotan tf-idf untuk term ke-j pada dokumen ke-i } \\
# {n_{i, j}}\quad\quad\>\>: \text { jumlah kemunculan term ke-j pada dokumen ke-i }\\
# {p} \quad\quad\quad\>\>: \text { banyaknya term yang terbentuk }\\
# {\sum_{j=1}^{p} n_{j, i}}: \text { jumlah kemunculan seluruh term pada dokumen ke-i }\\
# {d_{j}} \quad\quad\quad: \text { banyaknya dokumen yang mengandung term ke-j }\\
# $
# 
# 

# ### import Library TF-IDF

# import library yang dibutuhkan dalam melakukan pemrosesan TF-IDF dan juga ambil data dari data hasil preprocessing yang sudah dilakukan diatas.

# In[7]:


from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
df = dataCSV


# ### Term Frequency

# ubah data menjadi bentuk list, lalu lakukan proses tf dengan cara memanggil library CountVectorizer dari scikit-learn.

# In[8]:


#mengubah fitur dalam bentuk list
list_isi_berita = []
for i in range(len(df.iloc[:, -1])):
    list_isi_berita.append(df.iloc[i, -1])

# proses term frequency
count_vectorizer = CountVectorizer(min_df=1)
tf = count_vectorizer.fit_transform(list_isi_berita)

#get fitur
fitur = count_vectorizer.get_feature_names_out()

# menampilkan data TF
show_tf = count_vectorizer.fit_transform(list_isi_berita).toarray()
df_tf =pd.DataFrame(data=show_tf,index=list(range(1, len(show_tf[:,1])+1, )),columns=[fitur])
df_tf = df_tf.T

df_tf.head(8)


# ## TF-IDF

# setelah melakukan proses TF, lakukan proses TF-IDF dan kemudian simpan hasilnya dalam bentuk data frame.

# In[9]:


#tfidf dengan tfidf transformer
tfidf_transform = TfidfTransformer(use_idf=True,norm='l2',smooth_idf=True)
tfidf=tfidf_transform.fit_transform(count_vectorizer.fit_transform(list_isi_berita)).toarray()
df_tfidf =pd.DataFrame(data=tfidf,index=list(range(1, len(tfidf[:,1])+1, )),columns=[fitur])
df_tfidf.head(8)


# ## Latent Simantic Analysis (LSA)

# Algoritma LSA (Latent Semantic Analysis) adalah salah satu algoritma yang dapat digunakan untuk menganalisa hubungan antara sebuah frase/kalimat dengan sekumpulan dokumen.
# Dalam pemrosesan LSA ada tahap yang dinamakan Singular Value Decomposition (SVD), SVD adalah salah satu teknik reduksi dimensi yang bermanfaat untuk memperkecil nilai kompleksitas dalam pemrosesan term-document matrix. berikut adalah rumus SVD:
# 
# $$
# A_{m n}=U_{m m} x S_{m n} x V_{n n}^{T}
# $$
# 
# Dengan:
# 
# $
# {A_{m n}}: \text { Matrix Awal } \\
# {U_{m m}}: \text { Matrix ortogonal U }\\
# {S_{m n}}\>: \text { Matrix diagonal S }\\
# {V_{n n}^{T}}\>\>: \text { Transpose matrix ortogonal V }\\
# $

# In[10]:


from sklearn.decomposition import TruncatedSVD


# ### proses LSA dengan library TruncatedSVD dari scikit

# In[11]:


lsa = TruncatedSVD(n_components=10, random_state=36)
lsa_matrix = lsa.fit_transform(tfidf)


# ## proporsi topik pada tiap dokumen

# In[12]:


# menampilkan proporsi tiap topic pada masing-masing dokumen
df_topicDocument =pd.DataFrame(data=lsa_matrix,index=list(range(1, len(lsa_matrix[:,1])+1)))
df_topicDocument.head(6)


# ## proporsi term terhadap topik

# In[13]:


# menampilkan proporsi tiap topic pada masing-masing dokumen
df_termTopic =pd.DataFrame(data=lsa.components_,index=list(range(1, len(lsa.components_[:,1])+1)), columns=[fitur])
df_termTopic.head(100)

