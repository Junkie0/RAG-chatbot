import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from typing import List, Dict
import re

class SimpleRAG:
    def __init__(self):
        print("Initializing models...")
        model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True
        )

        print("Loading embedding model...")
        self.embed_model = SentenceTransformer('paraphrase-MiniLM-L3-v2')

        self.dimension = 384
        self.index = faiss.IndexFlatL2(self.dimension)
        self.documents = []

        # Set maximum context length
        self.max_context_length = 300  # Reduced context length
        print("System initialized!")

    def process_text(self, text: str, chunk_size: int = 150) -> List[str]:  # Reduced chunk size
        """Simple text chunking"""
        words = text.split()
        chunks = []

        for i in range(0, len(words), chunk_size):
            chunk = ' '.join(words[i:i + chunk_size])
            chunks.append(chunk)

        return chunks

    def load_file(self, file_path: str = "src/data.txt"):
        """Load and process a text file"""
        try:
            print(f"Loading {file_path}...")
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()

            print(f"Processing text ({len(text)} characters)...")
            chunks = self.process_text(text)
            print(f"Created {len(chunks)} chunks")

            if chunks:
                print("Generating embeddings...")
                embeddings = self.embed_model.encode(chunks)

                self.index = faiss.IndexFlatL2(self.dimension)
                self.index.add(np.array(embeddings).astype('float32'))

                self.documents = chunks
                print("File processed successfully!")
                return True

            return False

        except Exception as e:
            print(f"Error: {str(e)}")
            return False

    def prepare_prompt(self, question: str, contexts: List[str]) -> str:
        """Prepare a more structured and clear prompt."""
        combined_context = "\n\n".join(contexts)

        return f"""You are an AI assistant. Use the following context to answer the question concisely. 

Context:
{combined_context}

Question: {question}

Provide a clear and accurate answer based on the given context. Do not repeat the context. If the answer is not found in the context, say "I don't know."

Answer:
"""

    def query(self, question: str, k: int = 2) -> Dict:
        """Query the system and return only the answer without contexts."""
        if not self.documents:
            return {"answer": "Please load a document first."}
    
        question_embedding = self.embed_model.encode([question])[0]
        distances, indices = self.index.search(
            np.array([question_embedding]).astype('float32'), k
        )
    
        contexts = [self.documents[i] for i in indices[0]]
        prompt = self.prepare_prompt(question, contexts)
    
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=250,
            temperature=0.5,  
            top_p=0.9,
            num_return_sequences=1,
            pad_token_id=self.tokenizer.eos_token_id
        )
    
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # More robust extraction of the answer
        answer_parts = response.split("Answer:")
        answer = answer_parts[-1].strip() if len(answer_parts) > 1 else response.strip()
    
        # If the model couldn't find an answer, return a default message
        if not answer or answer.lower() in ["i don't know.", "i don't know", ""]:
            answer = "Sorry, I'm unable to answer this question."

        return {"answer": answer, "contexts": contexts}
