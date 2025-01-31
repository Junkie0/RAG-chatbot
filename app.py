from flask import Flask, render_template, request, jsonify
from RAG import SimpleRAG

app = Flask(__name__)
rag = SimpleRAG()

# Load the data file once at startup
print("Loading data.txt please wait")
if rag.load_file('src/data.txt'):
    print("Data loaded successfully!")
else:
    print("Failed to load data.")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask():
    data = request.json
    question = data.get('question', '')

    if not question.strip():
        return jsonify({"error": "Question cannot be empty"}), 400

    response = rag.query(question)
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=False)
