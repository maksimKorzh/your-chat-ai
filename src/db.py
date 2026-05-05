# Packages
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_ollama.llms import OllamaLLM
from langchain_chroma import Chroma
import os

# Generate context DB
def create_context_db(data):
  global retriever, prompt, chain

  documents = []
  ids = []
  
  for i, row in enumerate(data):
    document = Document(
      page_content=row,
      id=str(i)
    )
    
    documents.append(document)
    ids.append(str(i))
  
  db_path = './context_db'
  try: os.system('rm -r context_db')
  except: pass
  
  vector_store = Chroma(
    collection_name='context_db',
    persist_directory=db_path,
    embedding_function=embeddings
  )
  
  vector_store.add_documents(documents=documents, ids=ids)
  retriever = vector_store.as_retriever(search_kwargs={"k": 5})
  prompt = ChatPromptTemplate.from_template(template)
  chain = prompt | model

# Generate AI answer
def generate_response(question):
  reviews = retriever.invoke(question)
  result = chain.invoke({"reviews": reviews, "question": question})
  return result

# Load embeddings model
embeddings = OllamaEmbeddings(model="mxbai-embed-large")

# Load LLM model
model = OllamaLLM(model="qwen:0.5b")

# RAG retriever
retriever = None

# Current prompt
prompt = None

# Conversation
chain = None

# Prompt template
template = '''
You are an exeprt in answering questions based on given context

Here are some relevant reviews: {reviews}

Here is the question to answer: {question}
'''
