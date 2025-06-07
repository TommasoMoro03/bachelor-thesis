import sys
import os
import random
import numpy as np
from typing import List, Dict, Any

# Ensure NLTK punkt tokenizer is downloaded if not already present
try:
    import nltk

    nltk.data.find('tokenizers/punkt')
except (nltk.downloader.DownloadError, LookupError):
    print("NLTK 'punkt' tokenizer not found, attempting download...")
    nltk.download('punkt', quiet=True)
    print("NLTK 'punkt' tokenizer downloaded.")
except ImportError:
    print("NLTK not installed. Please install it: pip install nltk")
    sys.exit(1)

# LlamaIndex Imports
try:
    from llama_index.core import VectorStoreIndex, QueryBundle
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    from llama_index.core.schema import TextNode
    # Import Settings instead of ServiceContext
    from llama_index.core import Settings  # NEW IMPORT
except ImportError:
    print(
        "LlamaIndex or its dependencies not installed. Please install: pip install llama-index llama-index-embeddings-huggingface")
    sys.exit(1)

# --- Configuration ---
# IMPORTANT: This embedding model name MUST be the same as the one used in your Django app
# for chunking and actual retrieval.
EMBED_MODEL_NAME = "BAAI/bge-small-en-v1.5"

# The top-k value for retrieval
TEST_K_RETRIEVED = 5

# --- Mock Data (simulating your Django Chunks and Question) ---
# In a real scenario, these would come from your Django ORM
mock_document_content = """
The quick brown fox jumps over the lazy dog. This is a very common sentence used for testing.
It contains several common English words and good sentence structure.

The dog was indeed quite lazy, preferring to sleep all day in the sun.
It rarely moved unless it was time for food or a short walk.

Artificial intelligence is rapidly advancing, with large language models showing impressive capabilities.
These models can understand natural language, generate coherent text, and answer a wide range of questions.
However, they often lack access to specific, private, or real-time information.

Retrieval-Augmented Generation (RAG) addresses these limitations by connecting LLMs to external knowledge sources at inference time.
RAG enhances LLMs by using a retrieval system to find relevant document passages from an external corpus.
This retrieved context is then used to generate more accurate and up-to-date responses.
"""

# Simulate getting chunks from your chunking process.
# Here, we'll manually split them roughly, but in your app, these are real Chunk objects.
mock_chunks_raw = [
    "The quick brown fox jumps over the lazy dog. This is a very common sentence used for testing. It contains several common English words and good sentence structure.",
    "The dog was indeed quite lazy, preferring to sleep all day in the sun. It rarely moved unless it was time for food or a short walk.",
    "Artificial intelligence is rapidly advancing, with large language models showing impressive capabilities. These models can understand natural language, generate coherent text, and answer a wide range of questions.",
    "However, they often lack access to specific, private, or real-time information. Retrieval-Augmented Generation (RAG) addresses these limitations by connecting LLMs to external knowledge sources at inference time.",
    "RAG enhances LLMs by using a retrieval system to find relevant document passages from an external corpus. This retrieved context is then used to generate more accurate and up-to-date responses."
]

# Simulate your Django Chunk objects with Pks and text
mock_django_chunks = []
current_char_idx = 0
for i, text in enumerate(mock_chunks_raw):
    # This simulates how your Chunk objects would be structured
    mock_django_chunks.append({
        'pk': i + 1,  # Simulate primary key
        'text': text,
        'chunk_index': i,
        'start_char': current_char_idx,
        'end_char': current_char_idx + len(text)
    })
    current_char_idx += len(text) + 2  # Add some space for newlines/separators

mock_question = "What is RAG and why is it important?"  # Your test question

# --- Retrieval Logic (Simplified Version of your service function) ---

# 1. Initialize the embedding model (globally or once)
_embed_model_instance = None


def get_test_embed_model():
    global _embed_model_instance
    if _embed_model_instance is None:
        _embed_model_instance = HuggingFaceEmbedding(model_name=EMBED_MODEL_NAME)
    return _embed_model_instance


# Configure global LlamaIndex settings
print(f"Configuring LlamaIndex global settings with embed_model: {EMBED_MODEL_NAME}...")
embed_model = get_test_embed_model()
Settings.embed_model = embed_model
Settings.llm = None  # Set LLM to None as it's not needed for just retrieval embedding

print("LlamaIndex global settings configured.")

# 2. Convert mock Django chunks to LlamaIndex TextNodes
llama_nodes: List[TextNode] = []
for chunk_data in mock_django_chunks:
    llama_nodes.append(TextNode(
        text=chunk_data['text'],
        id_=str(chunk_data['pk']),  # Crucial: Use Django PK as Node ID
        metadata={
            'chunk_pk': chunk_data['pk'],
            'chunk_index': chunk_data['chunk_index'],
            'start_char': chunk_data['start_char'],
            'end_char': chunk_data['end_char'],
        }
    ))

if not llama_nodes:
    print("No nodes to index. Exiting test.")
    sys.exit(1)

# 3. Create an in-memory VectorStoreIndex
print("\n--- Creating VectorStoreIndex ---")
# FIX: No need to pass service_context directly if using global Settings
index = VectorStoreIndex(
    nodes=llama_nodes,
    # embed_model=embed_model # No longer directly passed here if using Settings
)
print("VectorStoreIndex created successfully.")

# 4. Get the retriever
retriever = index.as_retriever(similarity_top_k=TEST_K_RETRIEVED)
print(f"Retriever initialized with top_k={TEST_K_RETRIEVED}.")

# 5. Perform the retrieval
print(f"\n--- Performing Retrieval for Query: '{mock_question}' ---")
query_bundle = QueryBundle(query_str=mock_question)
retrieved_results = retriever.retrieve(query_bundle)
print(f"Retriever returned {len(retrieved_results)} results.")

# --- Display Results ---
print("\n--- Retrieved Chunks ---")
if retrieved_results:
    for i, node_with_score in enumerate(retrieved_results):
        node = node_with_score.node
        score = node_with_score.score

        # Access original Django Chunk PK from the node's ID or metadata
        original_pk = node.id_  # Or node.metadata['chunk_pk']

        print(f"Rank {i + 1}: (Chunk PK: {original_pk}, Score: {score:.4f})")
        print(f"  Text: {node.text[:150]}{'...' if len(node.text) > 150 else ''}")
        print("-" * 30)
else:
    print("No chunks retrieved.")

print("\n--- Test Complete ---")