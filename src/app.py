# Packages
from flask import Flask, request, jsonify, render_template
import db

# Create app
app = Flask(__name__)

# Home page
@app.route('/')
def index():
  return render_template('your-chat-ai.html')

# Create database
@app.route('/generate', methods=['POST'])
def generate():
  data = request.json
  context = data.get('context', '').split('\n')
  print('Creating context database...')
  db.create_context_db(context)
  print('Done')
  return jsonify({'status': 'Context database has been generated!'})

# Talk to model
@app.route('/chat', methods=['POST'])
def chat():
  data = request.json
  message = data.get('message', '')
  if db.retriever is None: return jsonify({'reply': 'Generate Knowledge Base first'})
  reply = f'{db.generate_response(message)}'
  return jsonify({'reply': reply})

# Main driver
if __name__ == '__main__':
  app.run(debug=True)
