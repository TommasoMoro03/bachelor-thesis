import json
from typing import List, Dict, Any

# --- LlamaIndex Imports ---
from llama_index.core import Document as LlamaDocument
from llama_index.core.node_parser import (
    TokenTextSplitter,
    SentenceSplitter,
    SemanticSplitterNodeParser,
)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from experiments.models import ChunkingStrategy
from experiments.service import structure_utils # Importa il tuo modulo con le funzioni custom

import nltk

try:
    nltk.data.find('tokenizers/punkt')
    print("NLTK 'punkt' tokenizer già scaricato.")
except LookupError:
    print("NLTK 'punkt' tokenizer non trovato, avvio il download...")
    nltk.download('punkt')
    nltk.download('wordnet')
    nltk.download('omw-1.4')
    nltk.download('punkt_tab')
    print("NLTK 'punkt' tokenizer scaricato.")

def apply_chunking_strategy(strategy: ChunkingStrategy, content: str, source_doc_id: str = "doc") -> List[Dict[str, Any]]:

    llama_document = LlamaDocument(
        text=content,
        doc_id=str(source_doc_id), # Es. source_text.id o un UUID
        metadata={'source_doc_id': str(source_doc_id)} # Esempio di metadato
    )

    params = strategy.parameters
    if not isinstance(params, dict): # Assicura che params sia un dizionario
        try:
            params = json.loads(params) if isinstance(params, str) else {}
        except json.JSONDecodeError:
            print(f"WARN: Parametri per strategia '{strategy.name}' non sono JSON valido. Usati defaults.")
            params = {}

    chunks_data_list: List[Dict[str, Any]] = [] # Inizializza qui, sarà popolata in ogni ramo

    try:
        if strategy.method_type == 'length':
            print(f"LlamaIndex: Configurazione TokenTextSplitter con params: {params}")
            node_parser = TokenTextSplitter(
                chunk_size=int(params.get('chunk_size', 512)),
                chunk_overlap=int(params.get('chunk_overlap', 50)),
                separator=params.get('separator', " "),
            )
            # Esegui il parsing e popola chunks_data_list QUI per 'length'
            nodes = node_parser.get_nodes_from_documents([llama_document])
            for node in nodes:
                start_char = node.start_char_idx if node.start_char_idx is not None else -1
                end_char = node.end_char_idx if node.end_char_idx is not None else -1
                if start_char == -1 or end_char == -1 or start_char >= end_char:
                    print(f"WARN: Nodo LlamaIndex (ID: {node.node_id}) con indici non validi/mancanti. Saltato.")
                    continue
                chunks_data_list.append({
                    'text': node.text,
                    'start_char': start_char,
                    'end_char': end_char,
                    'metadata': node.metadata if node.metadata else None
                })


        elif strategy.method_type == 'structure':
            structure_type = params.get('structure_type') # Recupera structure_type qui
            if structure_type == 'pure_paragraph':
                paragraph_separator = params.get('paragraph_separator', '\n\n')
                chunks_data_list = structure_utils._pure_paragraph_split(content, paragraph_separator)
            elif structure_type == 'n_sentence_chunking':
                sentences_per_chunk = int(params.get('sentences_per_chunk', 5))
                sentence_overlap = int(params.get('sentence_overlap', 1))
                chunks_data_list = structure_utils._n_sentence_chunking(content, sentences_per_chunk, sentence_overlap)
            elif structure_type == 'sentence_window':
                min_chars_per_chunk = int(params.get('min_chars_per_chunk', 200))
                max_chars_per_chunk = int(params.get('max_chars_per_chunk', 500))
                sentence_overlap_chars = int(params.get('sentence_overlap_chars', 50))
                chunks_data_list = structure_utils._sentence_window_chunking(content, min_chars_per_chunk, max_chars_per_chunk,
                                                             sentence_overlap_chars)
            else:  # Fallback al SentenceSplitter di LlamaIndex se non specificato un structure_type custom
                print(
                    f"LlamaIndex: Configurazione SentenceSplitter con params: {params} (No custom structure_type specificato)")
                node_parser = SentenceSplitter(
                    chunk_size=int(params.get('chunk_size', 1024)),
                    chunk_overlap=int(params.get('chunk_overlap', 200)),
                    separator=params.get('separator', " "),
                    paragraph_separator=params.get('paragraph_separator', "\n\n\n"),
                )
                # Esegui il parsing e popola chunks_data_list QUI per il fallback 'structure'
                nodes = node_parser.get_nodes_from_documents([llama_document])
                for node in nodes:
                    start_char = node.start_char_idx if node.start_char_idx is not None else -1
                    end_char = node.end_char_idx if node.end_char_idx is not None else -1
                    if start_char == -1 or end_char == -1 or start_char >= end_char:
                        print(f"WARN: Nodo LlamaIndex (ID: {node.node_id}) con indici non validi/mancanti. Saltato.")
                        continue
                    chunks_data_list.append({
                        'text': node.text,
                        'start_char': start_char,
                        'end_char': end_char,
                        'metadata': node.metadata if node.metadata else None
                    })

        elif strategy.method_type == 'semantic':
            print(f"LlamaIndex: Configurazione SemanticSplitterNodeParser con params: {params}")
            embed_model_name = params.get("embed_model_name")
            if not embed_model_name:
                raise ValueError("Per 'semantic' chunking, 'embed_model_name' è richiesto nei parametri.")

            try:
                print(f"Caricamento embedding model: {embed_model_name}")
                embed_model = HuggingFaceEmbedding(model_name=embed_model_name)
            except Exception as e_embed:
                print(f"ERRORE CRITICO: Impossibile caricare embedding model '{embed_model_name}': {e_embed}")
                raise ValueError(f"Impossibile caricare embedding model: {embed_model_name}. Dettagli: {e_embed}") from e_embed

            node_parser = SemanticSplitterNodeParser(
                embed_model=embed_model,
                breakpoint_percentile_threshold=int(params.get("breakpoint_percentile_threshold", 95)), # Converti a float
                buffer_size=int(params.get("buffer_size", 1)),
            )
            # Esegui il parsing e popola chunks_data_list QUI per 'semantic'
            nodes = node_parser.get_nodes_from_documents([llama_document])
            for node in nodes:
                start_char = node.start_char_idx if node.start_char_idx is not None else -1
                end_char = node.end_char_idx if node.end_char_idx is not None else -1
                if start_char == -1 or end_char == -1 or start_char >= end_char:
                    print(f"WARN: Nodo LlamaIndex (ID: {node.node_id}) con indici non validi/mancanti. Saltato.")
                    continue
                chunks_data_list.append({
                    'text': node.text,
                    'start_char': start_char,
                    'end_char': end_char,
                    'metadata': node.metadata if node.metadata else None
                })
        else:
            raise ValueError(f"Tipo di metodo di chunking '{strategy.method_type}' non supportato.")

    except ValueError as ve:
        print(f"Errore di configurazione strategia '{strategy.name}': {ve}")
        raise
    except ImportError as ie:
        print(f"Errore di import LlamaIndex (manca una libreria?): {ie}")
        raise ValueError(f"Dipendenza LlamaIndex mancante: {ie}") from ie
    except Exception as e:
        print(f"ERRORE CRITICO durante l'inizializzazione o l'uso del parser: {e}")
        raise Exception(f"Errore generale per strategia '{strategy.name}': {e}") from e

    print(f"ChunkImplementations: Restituiti {len(chunks_data_list)} chunk data per strategia '{strategy.name}'.")
    return chunks_data_list
