import pandas as pd
import gensim
import logging
import re
import spacy as spacy
from collections import defaultdict

df = pd.read_csv('./oasv2.csv',encoding="latin-1",delimiter=";")

data = df[['keyword','cc']]

print(data.head())

nlp = spacy.load('es')

nlp.Defaults.stop_words |= {'algún',
'alguna',
'algunas',
'alguno',
'algunos',
'ambos',
'ampleamos',
'ante',
'antes',
'aquel',
'aquellas',
'aquellos',
'aqui',
'arriba',
'atras',
'bajo',
'bastante',
'bien',
'cada',
'cierta',
'ciertas',
'cierto',
'ciertos',
'como',
'con',
'conseguimos',
'conseguir',
'consigo',
'consigue',
'consiguen',
'consigues',
'cual',
'cuando',
'dentro',
'desde',
'donde',
'dos',
'el',
'ellas',
'ellos',
'empleais',
'emplean',
'emplear',
'empleas',
'empleo',
'en',
'encima',
'entonces',
'entre',
'era',
'eramos',
'eran',
'eras',
'eres',
'es',
'esta',
'estaba',
'estado',
'estais',
'estamos',
'estan',
'estoy',
'fin',
'fue',
'fueron',
'fui',
'fuimos',
'gueno',
'ha',
'hace',
'haceis',
'hacemos',
'hacen',
'hacer',
'haces',
'hago',
'incluso',
'intenta',
'intentais',
'intentamos',
'intentan',
'intentar',
'intentas',
'intento',
'ir',
'la',
'largo',
'las',
'lo',
'los',
'mientras',
'mio',
'modo',
'muchos',
'muy',
'nos',
'nosotros',
'otro',
'para',
'pero',
'podeis',
'podemos',
'poder',
'podria',
'podriais',
'podriamos',
'podrian',
'podrias',
'por',
'por qué',
'porque',
'primero',
'puede',
'pueden',
'puedo',
'quien',
'sabe',
'sabeis',
'sabemos',
'saben',
'saber',
'sabes',
'ser',
'si',
'siendo',
'sin',
'sobre',
'sois',
'solamente',
'solo',
'somos',
'soy',
'su',
'sus',
'también',
'teneis',
'tenemos',
'tener',
'tengo',
'tiempo',
'tiene',
'tienen',
'todo',
'trabaja',
'trabajais',
'trabajamos',
'trabajan',
'trabajar',
'trabajas',
'trabajo',
'tras',
'tuyo',
'ultimo',
'un',
'una',
'unas',
'uno',
'unos',
'usa',
'usais',
'usamos',
'usan',
'usar',
'usas',
'uso',
'va',
'vais',
'valor',
'vamos',
'van',
'vaya',
'verdad',
'verdadera',
'VERDADERO',
'vosotras',
'vosotros',
'voy',
'yo',
'él',
'ésta',
'éstas',
'éste',
'éstos',
'última',
'últimas',
'último',
'últimos',
'a',
'añadió',
'aún',
'actualmente',
'adelante',
'además',
'afirmó',
'agregó',
'ahí',
'ahora',
'al',
'algo',
'alrededor',
'anterior',
'apenas',
'aproximadamente',
'aquí',
'así',
'aseguró',
'aunque',
'ayer',
'buen',
'buena',
'buenas',
'bueno',
'buenos',
'cómo',
'casi',
'cerca',
'cinco',
'comentó',
'conocer',
'consideró',
'considera',
'contra',
'cosas',
'creo',
'cuales',
'cualquier',
'cuanto',
'cuatro',
'cuenta',
'da',
'dado',
'dan',
'dar',
'de',
'debe',
'deben',
'debido',
'decir',
'dejó',
'del',
'demás',
'después',
'dice',
'dicen',
'dicho',
'dieron',
'diferente',
'diferentes',
'dijeron',
'dijo',
'dio',
'durante',
'e',
'ejemplo',
'ella',
'ello',
'embargo',
'encuentra',
'esa',
'esas',
'ese',
'eso',
'esos',
'está',
'están',
'estaban',
'estar',
'estará',
'estas',
'este',
'esto',
'estos',
'estuvo',
'ex',
'existe',
'existen',
'explicó',
'expresó',
'fuera',
'gran',
'grandes',
'había',
'habían',
'haber',
'habrá',
'hacerlo',
'hacia',
'haciendo',
'han',
'hasta',
'hay',
'haya',
'he',
'hecho',
'hemos',
'hicieron',
'hizo',
'hoy',
'hubo',
'igual',
'indicó',
'informó',
'junto',
'lado',
'le',
'les',
'llegó',
'lleva',
'llevar',
'luego',
'lugar',
'más',
'manera',
'manifestó',
'mayor',
'me',
'mediante',
'mejor',
'mencionó',
'menos',
'mi',
'misma',
'mismas',
'mismo',
'mismos',
'momento',
'mucha',
'muchas',
'mucho',
'nada',
'nadie',
'ni',
'ningún',
'ninguna',
'ningunas',
'ninguno',
'ningunos',
'no',
'nosotras',
'nuestra',
'nuestras',
'nuestro',
'nuestros',
'nueva',
'nuevas',
'nuevo',
'nuevos',
'nunca',
'o',
'ocho',
'otra',
'otras',
'otros',
'parece',
'parte',
'partir',
'pasada',
'pasado',
'pesar',
'poca',
'pocas',
'poco',
'pocos',
'podrá',
'podrán',
'podría',
'podrían',
'poner',
'posible',
'próximo',
'próximos',
'primer',
'primera',
'primeros',
'principalmente',
'propia',
'propias',
'propio',
'propios',
'pudo',
'pueda',
'pues',
'qué',
'que',
'quedó',
'queremos',
'quién',
'quienes',
'quiere',
'realizó',
'realizado',
'realizar',
'respecto',
'sí',
'sólo',
'se',
'señaló',
'sea',
'sean',
'según',
'segunda',
'segundo',
'seis',
'será',
'serán',
'sería',
'sido',
'siempre',
'siete',
'sigue',
'siguiente',
'sino',
'sola',
'solas',
'solos',
'son',
'tal',
'tampoco',
'tan',
'tanto',
'tenía',
'tendrá',
'tendrán',
'tenga',
'tenido',
'tercera',
'toda',
'todas',
'todavía',
'todos',
'total',
'trata',
'través',
'tres',
'tuvo',
'usted',
'varias',
'varios',
'veces',
'ver',
'vez',
'y',
'ya'
}


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


def cleaning(doc):
    txt = [token.lemma_ for token in doc if not token.is_stop]
    if len(txt)>2:
        return ' '.join(txt)

#brief_prev = (normalize(row) for row in df['nombreOA'])

brief_cleaning = (re.sub("[^A-Za-z']+",' ',normalize(str(row).lower())) for row in df['cc'])

txt_elors = [cleaning(doc) for doc in nlp.pipe(brief_cleaning,batch_size=50,n_threads=-1)]

def clean_line(line):
    list_words = normalize(str(line)).lower().split(",")
    return (' '.join(list_words))

#read file and return list of sentences    
def createListSentences(file):
    f = open(file,"r",encoding='utf-8')
    lines = f.readlines()
    list_sentences = list()
    for l in lines:
        cleaned_line = clean_line(l)
        list_sentences.append(cleaned_line)
    return list_sentences    
        
#df['libro']

file="./actividades.csv"
filelibro="./libro1.txt"

data_sentences = {"f":createListSentences(file)+createListSentences(filelibro)}

df_file = pd.DataFrame(data_sentences)

brieflibro_cleaning = (re.sub("[^A-Za-z']+",' ',normalize(str(row).lower())) for row in df_file['f'])
txt_libro = [cleaning(doc) for doc in nlp.pipe(brieflibro_cleaning,batch_size=50,n_threads=-1)]

df_clean= pd.DataFrame({'clean':txt_elors+txt_libro})
df_clean = df_clean.dropna().drop_duplicates()
df_clean.to_csv("out.csv")
print("clean df:",df_clean.shape)

from gensim.models.phrases import Phrases, Phraser
sent = [row.split() for row in df_clean['clean']]

phrases = Phrases(sent,min_count=1,threshold=10,progress_per=10000)
sentences = phrases[sent]

word_freq = defaultdict(int)
for sent in sentences:
    for i in sent:
        word_freq[i] += 1
#print(len(word_freq))
print(sorted(word_freq,key=word_freq.get,reverse=True)[:10])

import multiprocessing
from gensim.models import Word2Vec

from gensim.models import KeyedVectors
cores = multiprocessing.cpu_count()

w2v = Word2Vec(min_count=20,window=3,
                     size=300,
                     sample=1e-5, 
                     alpha=0.003, 
                     min_alpha=0.0007, 
                     negative=20,
                     workers=cores-1)

w2v.build_vocab(sentences)

model= KeyedVectors.load_word2vec_format("SBW-vectors-300-min5.txt",binary=False)
w2v.build_vocab([list(model.vocab.keys())],progress_per=10000,update=True)
w2v.intersect_word2vec_format("SBW-vectors-300-min5.txt",binary=False,lockf=1.0)
w2v.train(sentences,total_examples=w2v.corpus_count,epochs=60,report_delay=1)
w2v.init_sims(replace=True)
print(len(w2v.wv.vocab))
#print((w2v.wv.vocab.keys()))
#print(w2v.wv.most_similar(positive=["angulo"]))
#print(w2v.wv.most_similar(positive=["fraccion"]))
#print(w2v.wv.most_similar(positive=["numeros"]))
#print(w2v.wv.most_similar(positive=["unidades"]))
#print(w2v.wv.most_similar(positive=["conversiones"]))
#print(w2v.wv.most_similar(positive=["variable"]))
i=0
#print(w2v.wv.most_similar(positive=["representacion"]))

df_kw = pd.read_csv('./kw.csv')
df_kw = df_kw[['kw']]

brief_cleaning_kw = (re.sub("[^A-Za-z']+",' ',normalize(str(row).lower())) for row in df_kw['kw'])
txt_kw = [cleaning(doc) for doc in nlp.pipe(brief_cleaning_kw,batch_size=50,n_threads=-1)]

df_clean_kw = pd.DataFrame({'clean':txt_kw})
df_clean_kw = df_clean_kw.dropna().drop_duplicates()
              
sent_kw = [row.split() for row in df_clean_kw['clean']]

phrases_kw = Phrases(sent_kw,min_count=1,threshold=10,progress_per=10000)
sentences_kw = phrases_kw[sent_kw]


print("tamaño wordvocab model:" ,len(w2v.wv.vocab))
print("tamaño keyword:" ,len(sentences_kw))

#set_kw = set(sentences_kw)
#set_vocab = set(w2v.wv.vocab)
#set_diff = set_kw-set_vocab
#print("diff",len(set_diff))


#for image
#from sklearn.decomposition import PCA
#from matplotlib import pyplot as plt
#X = w2v[w2v.wv.vocab]
#pca = PCA(n_components=2)
#result = pca.fit_transform(X)

w2v.wv.save_word2vec_format('modelowithpretrainded.vec',binary=False)

#plt.scatter(result[:,0],result[:,1])
#words= list(w2v.wv.vocab)
#for i,word in enumerate(words):
#    plt.annotate(word,xy=(result[i,0],result[i,1]))
#plt.show()






#while i<1000:
#  input1=input()
  #input2=input()
#  print(w2v.wv.most_similar(positive=[input1]))
  #print(w2v.wv.similarity(input1, input2))


