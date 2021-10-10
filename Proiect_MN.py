import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download('punkt')
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import TruncatedSVD

i=0
ok=1
D=[]
print("Introdu numele fisierelor si scrie no dupa ce ai finalizat.")
while ok:
    name_file=input()
    if name_file=="no":
        ok=0
    else:
        with open(name_file) as file:                   
            d=[word.lower() for line in file for word in line.split()]
            D.append(d)
# for i in range(len(D)):
#     print(D[i]) 
i=len(D)


termeni=['living','room','bathroom','aircraft','teren','minge','fotbal','airplanes','sport','football','garden','sports','parts']

TermDoc=np.zeros((i,len(termeni)))

for j in range(i):
    for k in range(len(termeni)):
        idx=0
        
        for z in range(len(D[j])):
            if D[j][z]==termeni[k]:   
                idx=idx+1
        TermDoc[j][k]=idx
#print(D)
#D=np.array(D, dtype="object")
U,S,V=np.linalg.svd(TermDoc)

print(TermDoc)
print(U)
print(V)
print(S)

fig0=plt.figure(0)
plt.imshow(V)
plt.colorbar()


stop_words = ['an','a','on','to','from','are','as','in','the','they','this']


for i in range(4):
    D[i]=[word for word in D[i] if word not in stopwords.words()]
#print(D)
with open('text.txt', 'r') as file:
    data = file.read().replace('\n', '')
#print(data)
with open('text2.txt', 'r') as file:
    data2 = file.read().replace('\n', '')
#print(data2)
with open('text3.txt', 'r') as file:
    data3 = file.read().replace('\n', '')
#print(data3)
with open('text4.txt', 'r') as file:
    data4 = file.read().replace('\n', '')
#print(data4)

data_cuvinte=word_tokenize(data)
data_cuvinte_curat= [word for word in data_cuvinte if not word in stopwords.words()]
data= (" ").join(data_cuvinte_curat)

data_cuvinte2=word_tokenize(data2)
data_cuvinte_curat2= [word for word in data_cuvinte2 if not word in stopwords.words()]
data2= (" ").join(data_cuvinte_curat2)

data_cuvinte3=word_tokenize(data3)
data_cuvinte_curat3= [word for word in data_cuvinte3 if not word in stopwords.words()]
data3= (" ").join(data_cuvinte_curat3)

data_cuvinte4=word_tokenize(data4)
data_cuvinte_curat4= [word for word in data_cuvinte4 if not word in stopwords.words()]
data4 =(" ").join(data_cuvinte_curat4)


D2=[data,data2,data3,data4]
D2=[word for word in D2 if word not in stopwords.words()]
#print(D2)
copieD=[]

copieD=D[0]+D[1]+D[2]+D[3]
print("Cele mai comune concepte in toate fisierele")

vectorizer_total2 = CountVectorizer()
punga_cuvinte2=vectorizer_total2.fit_transform(D2)
#print(punga_cuvinte2.todense())
vectorizer_total = CountVectorizer()
punga_cuvinte=vectorizer_total.fit_transform(copieD)
svd_model_total = TruncatedSVD(n_components=4)
lsa = svd_model_total.fit_transform(punga_cuvinte2)
print(punga_cuvinte2.todense())
topic_encoded_df=pd.DataFrame(lsa, columns = ["concept 1","concept 2","concept 3","concept 4"])
topic_encoded_df["copieD"]=D2
#print(topic_encoded_df[["copieD","concept 1","concept 2","concept 3","concept 4"]])
dictionar=vectorizer_total2.get_feature_names()
print(svd_model_total.components_)
print(lsa)
encoding_matrix = pd.DataFrame(svd_model_total.components_,index=["concept 1","concept 2","concept 3","concept 4"]).T
encoding_matrix['concept 1']=np.abs(encoding_matrix['concept 1'])
encoding_matrix['concept 2']=np.abs(encoding_matrix['concept 2'])
encoding_matrix['concept 3']=np.abs(encoding_matrix['concept 3'])
encoding_matrix['concept 4']=np.abs(encoding_matrix['concept 4'])
encoding_matrix.sort_values('concept 1', ascending=False)
encoding_matrix.sort_values('concept 2', ascending=False)
encoding_matrix.sort_values('concept 3', ascending=False)
encoding_matrix.sort_values('concept 4', ascending=False)
encoding_matrix['termeni']=dictionar
#print(encoding_matrix)
print(encoding_matrix.sort_values('concept 1', ascending=False))
print(encoding_matrix.sort_values('concept 2', ascending=False))
print(encoding_matrix.sort_values('concept 3', ascending=False))
print(encoding_matrix.sort_values('concept 4', ascending=False))

fig, ax= plt.subplots()

for i in range(1000):
    concept_1=topic_encoded_df['concept 1'].values
    concept_2=topic_encoded_df['concept 2'].values
    concept_3=topic_encoded_df['concept 3'].values
    concept_4=topic_encoded_df['concept 4'].values

plt.scatter(concept_1,concept_2,concept_3,concept_4)
ax.axvline(linewidth=0.5)
ax.axhline(linewidth=0.5)

plt.show()

U1,S1,V1=np.linalg.svd(punga_cuvinte2.todense())
fig1=plt.figure(1)
print(U1)
plt.imshow(V1)
plt.colorbar()
fig2=plt.figure(2)
plt.imshow(U1)
plt.show()



