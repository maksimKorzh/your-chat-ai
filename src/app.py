# Packages
from flask import Flask, Response, request, jsonify, render_template, stream_with_context
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_ollama.llms import OllamaLLM
from langchain_chroma import Chroma
import os

# Load embeddings model
embeddings = OllamaEmbeddings(model="mxbai-embed-large")

# Load LLM model
model = OllamaLLM(model="llama3.2")

# RAG retriever
retriever = None

# Current prompt
prompt = None

# Conversation
chain = None

# Prompt template
template = '''
You are an exeprt in answering questions about real estate properties

Here are some relevant properties: {properties}

Here is the question to answer: {question}
'''

# Generate context DB
def create_context_db():
  global retriever, prompt, chain

  no_db =  not os.path.exists('./context_db')
  
  if no_db:
    print('Creating context database...')
    context = []
    with open('context.txt') as f:
      context = f.read().split('\n')[:-1]
  
    documents = []
    ids = []
  
    for i, row in enumerate(context):
      print(f'ADDING DOCUMENT: #{i}', row[:50] + '...')
      document = Document(
        page_content=row,
        id=str(i)
      )
    
      documents.append(document)
      ids.append(str(i))
  
  print('Building vector database...')
  vector_store = Chroma(
    collection_name=f'context_db',
    persist_directory=f'./context_db',
    embedding_function=embeddings
  )
  
  if no_db: vector_store.add_documents(documents=documents, ids=ids)
  retriever = vector_store.as_retriever(search_kwargs={"k": 1})
  prompt = ChatPromptTemplate.from_template(template)
  chain = prompt | model

# Streaming generator
def generate_response_stream(question):
  properties = retriever.invoke(question)

  for chunk in chain.stream({
    "properties": properties,
    "question": question
  }):
    # Adjust depending on what chunk is
    if isinstance(chunk, str): yield chunk
    elif hasattr(chunk, "content"): yield chunk.content
    elif isinstance(chunk, dict): yield chunk.get("answer", "")
    else: yield str(chunk)

# Create vector database
create_context_db()

# Create app
app = Flask(__name__)

# Home page
@app.route('/')
def index():
  return render_template('chat.html')

# Talk to model
@app.route('/chat', methods=['POST'])
def chat():
  data = request.json
  message = data.get('message', '')
  def generate():
    for piece in generate_response_stream(message):
      yield piece

  return Response(
    stream_with_context(generate()),
    mimetype="text/plain"
  )

# Main driver
if __name__ == '__main__':
  app.run(debug=True)
