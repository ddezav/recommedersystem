from gensim.models import KeyedVectors
from gensim.models import Word2Vec
import pandas as pd

w2v= KeyedVectors.load_word2vec_format("modelowithpretrainded.vec",binary=False)
def normalize(s):
    replace = (("á","a"),
        ("é","e"),
        ("í","i"),
        ("ó","o"),
        ("ú","u"),
        ("ñ","n"))
    for a, b in replace:
        s= s.replace(a,b).replace(a.upper(),b.upper())
    return s

def analizarOA(idOA,df_oas,w2v):
    #CONSEGUIR NOMBRE
    nombre = df_oas.iloc[idOA,1]
    #print("nombre oa=",nombre)
    #CONSEGUIR KEYWORD (pueden ser varias palabras)
    kw = df_oas.iloc[idOA,7]
    #print("kw=",kw)
    list_kw = normalize(str(kw)).lower().split(",")
    list_kw = (' '.join(list_kw)).split(" ")

    #BUSCAR EN EL WORDVOCAB
    list_result=list()
    estaenwv = False    
    for kw in list_kw:
        try:            
            result = w2v.wv.similar_by_word(kw)
            list_result+=result
            estaenwv = True
        except:
            #print("no se encontró",kw)
            continue
    return list_result,estaenwv

df_oas =pd.read_csv('./oas.csv',encoding='latin-1',delimiter=',')
#df_oas = df_oas[['keyword','competencia_id']]

count_estan = 0
for i in range(0,150):
    result,estaenwv = analizarOA(i,df_oas,w2v)
    if estaenwv:
        count_estan+=1
    print("i=",i," , ",estaenwv,", size=",len(result))
print("found ",count_estan)

#calculate similarity between 2 oa's name
def calculateDistance(idOA1,idOA2,df_oas,w2v,column):
    nombreOA1 = normalize(str(df_oas.iloc[idOA1,column]).lower())
    nombreOA2 = normalize(str(df_oas.iloc[idOA2,column]).lower())
    
    return w2v.wmdistance(nombreOA1,nombreOA2)

#for i in range(0,150):
#    mostsimilaroa = -1
#    minim_distance = 100;
#    for j in range(0,150):
#        d= calculateDistance(i,j,df_oas,w2v,7)
#        if(d<minim_distance):
#            minim_distance =d
#            mostsimilaroa = j
    #print(df_oas.iloc[i,1])
    #print("most similar=",df_oas.iloc[mostsimilaroa,1],minim_distance)



    
