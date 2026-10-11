import PyPDF2 
import nltk
from nltk import sent_tokenize, word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
import re
from sentence_transformers import SentenceTransformer
from sentence_transformers import CrossEncoder
from sklearn.metrics.pairwise import cosine_similarity
import faiss
import numpy as np

import os
from openai import OpenAI


# nltk.download('punkt')
# nltk.download('punkt_tab')
#nltk.download('stopwords')

#--------Read and Extract data from PDF Doc---------------
file_path = ["D:\\Abbees\\Python_Class_Doc\\Class_Notes\\Leave_Policy.pdf","D:\\Abbees\\Python_Class_Doc\\Class_Notes\\OfficeTime.pdf"]

text = ""
document = []

for file in file_path:
  #PyPDF2, read this PDF file and give me an object through which I can access its contents.
  #it is class datatype
  reader = PyPDF2.PdfReader(file) 
  for page in reader.pages:
    #from the class reader we are extracting the pages using reader.pages
    # += means text = text + new content
    text += page.extract_text()
  document.append(text)


#print(len(document))
#print(type(document))

# NLP Natural Language Processing - 
# NLP is the process of enabling computers to understand and work with human language.
# What we do in NLP -- Tokenisation (Sentence and Words), Remove stop words, Stemming, Lemmatization

#--------Sentence Tokenization---------------
sentences = []

for doc in document:
  sent = sent_tokenize(doc)
  #print(type(sent))
  for i in sent:
    sentences.append(i)

#print(type(sentences))
#print(len(sentences)) #137
#print(sentences)

#--------Removing Special Character and \n---------------

clean_sentences = []
for sent in sentences:
  pattern1 = r"/n"
  pattern2 = r"[^0-9a-zA-Z]"
  text = re.sub(pattern1, " ", sent)
  text = re.sub(pattern2, " ", text)
  clean_sentences.append(text)

#print(len(clean_sentences)) #137

#--------Word Tokenization---------------
# words = []

# for i in sentences:
#   word = word_tokenize(i)
#   for i in word:
#     words.append(i)

#print(len(words)) #3525 -- this is the count BEFORE clean sentence

words = []

for i in clean_sentences:
  word = word_tokenize(i)
  for i in word:
    words.append(i)

#print(len(words)) # 3291 -- this is the count AFTER clean sentence

#--------Stop Words Removal---------------

stopwords = nltk.corpus.stopwords.words('english')

# print(len(stopwords)) #198
# print(stopwords) 
swords = []

for word in words:
  if word.lower() not in stopwords:
    swords.append(word) 

#print(len(swords)) #2303  -- this is the count BEFORE clean sentence
#print(len(swords)) #2032  -- this is the count AFTER clean sentence

#--------Fixed Size Chuncking---------------

chunk_size = 20
chunks = []

for i in range(0, len(swords), chunk_size):
  words = swords[i : i+chunk_size]  
  chunks.append(" ".join(words))

#print(len(chunks)) #102

#--------Fixed Size with Overlap Chuncking---------------

chunk_size = 20
overlap = 5
step = chunk_size - overlap
overlap_chunks = []

for i in range(0, len(swords), step):
  #print(i,i + chunk_size)
  words = swords[i : i + chunk_size]  
  overlap_chunks.append(" ".join(words))
#print(len(overlap_chunks)) #136

#--------Embedding Chuncks---------------

model = SentenceTransformer("all-MiniLM-L6-v2")
emb_chunks = []
for i in chunks:
  chunk_embedding = model.encode(i)
  emb_chunks.append(chunk_embedding) 

#print(len(emb_chunks)) #102

#--------Semmatic Chuncking---------------

threshold = 0.6
current_chunk = [chunks[0]]
semantic_chunks = []

for i in range(0,len(emb_chunks)-1):
  sim_score = cosine_similarity([emb_chunks[i]], [emb_chunks[i+1]])[0][0]

  if sim_score >= threshold:
    current_chunk.append(chunks[i+1])
    
  else :
    semantic_chunks.append(" ".join(current_chunk))
    current_chunk = []
    current_chunk.append(chunks[i])

semantic_chunks.append(" ".join(current_chunk))

#print(len(semantic_chunks)) #76

#print(semantic_chunks)

#--------Embedding of Semmatic Chunking---------------

#We are using the same model to embed the semmatic chunks 
emb_sem_chunks = model.encode(semantic_chunks , convert_to_numpy = True , normalize_embeddings=True)
#print(len(emb_sem_chunks)) #76
#print(emb_sem_chunks)  
# this will have embedded data like [[ 0.01822293 -0.01908587 -0.00178533 ... -0.03153628 -0.03156529
#    0.01575828]

#--------uploading the embedding into Vector DB---------------

# FAISS generally expects vectors in float32 format, and this format is also memory-efficient.
es_chunks = emb_sem_chunks.astype("float32")
# print(es_chunks) # this will have same data as emb_sem_chunks

# Magnitude = how long the arrow is.
# Direction = where the arrow points.
# Normalization = make every arrow the same length so we can focus on where it points.
faiss.normalize_L2(es_chunks)

emb_dim = es_chunks.shape[1]
emb_num = es_chunks.shape[0]
# print(emb_num , emb_dim) #76 , 384 this represents there are 76 chunks and each chunks have 384 dimensions

# creates index for each vectors
index = faiss.IndexFlatL2(emb_dim)
# print(index)
index.add(es_chunks)

# print("Vector Dimension : " ,emb_dim) #384
# print("Number of vectors : " ,index.ntotal) #76

# Get the query from the user and convert them into embeddings using the same model
#user_query = input("Ask your question : ")
#how many sick leave i can take in a year
user_query = "how many casual leave i can take in a year"

query_embedding = model.encode(user_query , convert_to_numpy = True , normalize_embeddings=True)

#--------Comparing Query emb with Chunks emb---------------
"""
threshold = 0.5
retrived_chunks = []

for i in range(0,len(emb_sem_chunks)):  
# learn the difference for i in (0,4) (0 1 2 3 4) and for i in range (0,4) (0 1 2 3)
  similarity_score = cosine_similarity([query_embedding], [emb_sem_chunks[i]])[0]

  if similarity_score >= threshold:
    retrived_chunks.append(semantic_chunks[i])

#print(len(retrived_chunks)) #10
# print(retrived_chunks) #10
"""

def top_k_retrival(query_emb, chunk_emb, chunk , top_k):
  score = cosine_similarity([query_emb] , chunk_emb)[0]
  top_indices = np.argsort(score)[::-1][:top_k]
  return [(chunk[i], score[i])  for i in top_indices]

top_k_result =  top_k_retrival(query_embedding,emb_sem_chunks, semantic_chunks, 3)
#print (top_k_result)

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-12-v2") # load the model into the variable it predicts how relevant that chunk is to the query.

def rerank_chunks(query, candidate_chunks, top_k = 3):
  pair = [(query, chunk) for chunk in candidate_chunks] # it pairs query with each chunk query,chunk1 query,chunk2
  scores =  reranker.predict(pair) # gives score to each query chunk pair
  ranked_indices = np.argsort(scores)[::-1][:top_k] # gives index to score, sort the index and take top 3 values
  return  [(candidate_chunks[i] , scores[i]) for i in ranked_indices] # gives chunk, score pair for top 3

top_k_candidate = [chunk for chunk, score in top_k_result] # get only the chunk from chunk,score pair
reranker_result = rerank_chunks(user_query,top_k_candidate,3) 
# print(reranker_result)

prompt = f"""
Your an intelligent HR assistanct
The user asked a question : {user_query}
and for this question i have the retrived data as : {reranker_result}
answer only from the retrived data
if data not available then give a grounded response "No Information available for this query"
"""


client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key="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"
)

response = client.chat.completions.create(
    model="openai/gpt-oss-120b",
    messages=[
        {
            "role": "user",
            "content": prompt
        }
    ],
    temperature=0.2
)

answer = response.choices[0].message.content

print(answer)
