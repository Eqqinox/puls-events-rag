"""Test des imports"""

# Test Faiss
import faiss
print("faiss importé")

# Test LangChain + FAISS
from langchain_community.vectorstores import FAISS
print("langchain FAISS importé")

# Test Embeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
print("HuggingFaceEmbeddings importé")

# Test Mistral
from mistralai import Mistral
print("Mistral importé")

print("\nTous les imports fonctionnent !")