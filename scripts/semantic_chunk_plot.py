import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.node_parser import SentenceSplitter

# 1. Configurazione
DOCUMENT_PATH = "../media/source_texts/ted_talk_2.txt"  # <--- MODIFICA QUESTO PER IL TUO FILE TXT
EMBED_MODEL_NAME = "BAAI/bge-small-en-v1.5"  # Lo stesso modello che usi per il chunking
BREAKPOINT_PERCENTILE_THRESHOLD = 95  # Soglia di percentile per i breakpoint (es. 95 per "high cohesion")

# 2. Carica il documento
try:
    with open(DOCUMENT_PATH, 'r', encoding='utf-8') as f:
        content = f.read()
except FileNotFoundError:
    print(f"Errore: File non trovato a {DOCUMENT_PATH}")
    exit()
except Exception as e:
    print(f"Errore durante la lettura del file: {e}")
    exit()

# 3. Tokenizzazione in frasi (usiamo SentenceSplitter di LlamaIndex per coerenza)
# SentenceSplitter è un NodeParser che può anche essere usato solo per splittare in frasi
sentence_splitter = SentenceSplitter(chunk_size=1024,
                                     chunk_overlap=0)  # chunk_size e overlap non influiscono qui, è solo per ottenere le frasi
# Per ottenere solo le frasi, possiamo usare nltk.sent_tokenize() se preferiamo,
# ma SentenceSplitter gestisce anche preprocessing interno.
# nodes = sentence_splitter.get_nodes_from_documents([LlamaDocument(text=content)])
# sentences = [node.text for node in nodes]
# Un approccio più diretto e spesso sufficiente:
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)
sentences = nltk.sent_tokenize(content)

if not sentences:
    print("Nessuna frase estratta dal documento. Impossibile procedere.")
    exit()
print(f"Estratte {len(sentences)} frasi.")

# 4. Generazione degli embeddings
print(f"Caricamento modello di embedding: {EMBED_MODEL_NAME}...")
try:
    embed_model = HuggingFaceEmbedding(model_name=EMBED_MODEL_NAME)
    # embed_model.get_text_embedding_batch restituisce una lista di vettori (list of list o list of np.ndarray)
    raw_embeddings = embed_model.get_text_embedding_batch(sentences)

    # CONVERSIONE IMPORTANTE: Assicurati che sia un array NumPy 2D
    # np.array() convertirà la lista di liste/array in un singolo array NumPy 2D
    sentence_embeddings = np.array(raw_embeddings)

    print("Embeddings generati.")
    print(f"Shape degli embeddings: {sentence_embeddings.shape}")  # Debug per verificare la shape
except Exception as e:
    print(f"Errore durante la generazione degli embeddings: {e}")
    exit()

# 5. Calcolo della similarità coseno tra frasi adiacenti
similarities = []
for i in range(
        len(sentences) - 1):  # Itera sulla lunghezza delle frasi, non degli embeddings (anche se dovrebbero essere uguali)
    # Non è più necessario reshape se sentence_embeddings è già un array 2D
    # sklearn.metrics.pairwise.cosine_similarity può prendere vettori 1D direttamente
    # per la similarità tra due vettori singoli, o vettori 2D (n_samples, n_features) per batch.
    # In questo caso, passiamo i vettori 1D estratti da sentence_embeddings.

    # Assicurati che i singoli elementi siano trattati come array 1D
    embedding1 = sentence_embeddings[i]
    embedding2 = sentence_embeddings[i + 1]

    # cosine_similarity può accettare due array 1D se li si passa come liste di array
    # o, più semplicemente, usare la formula del prodotto scalare normalizzato manualmente
    # per la similarità tra due singoli vettori 1D per evitare reshape complicati.

    # Option 1: Using sklearn's cosine_similarity with 1D inputs (reshaping internally)
    sim = cosine_similarity(embedding1.reshape(1, -1), embedding2.reshape(1, -1))[0][0]

    # Option 2: Manual cosine similarity for 1D vectors (often cleaner for single pairs)
    # dot_product = np.dot(embedding1, embedding2)
    # norm_a = np.linalg.norm(embedding1)
    # norm_b = np.linalg.norm(embedding2)
    # sim = dot_product / (norm_a * norm_b) if (norm_a * norm_b) != 0 else 0

    similarities.append(sim)

if not similarities:
    print("Meno di due frasi, impossibile calcolare le similarità.")
    exit()

# 6. Identificazione dei breakpoint (soglia di percentile)
# Un breakpoint è dove la similarità scende al di sotto di una certa soglia.
# La soglia è un percentile delle similarità calcolate.
threshold_value = np.percentile(similarities,
                                100 - BREAKPOINT_PERCENTILE_THRESHOLD)  # Il 95° percentile dei breakpoint significa il 5° percentile della similarità

# Identifica gli indici (frasi) dove la similarità è sotto la soglia
breakpoint_indices = [i for i, sim in enumerate(similarities) if sim < threshold_value]

print(f"Soglia di similarità (breakpoint percentile {BREAKPOINT_PERCENTILE_THRESHOLD}): {threshold_value:.4f}")
print(f"Trovati {len(breakpoint_indices)} breakpoint.")

# 7. Plotting
plt.figure(figsize=(15, 7))
plt.plot(similarities, marker='o', linestyle='-', color='skyblue', label='Similarità Coseno Frasi Adiacenti')

# Aggiungi i breakpoint sul grafico
for bp_idx in breakpoint_indices:
    plt.axvline(x=bp_idx, color='red', linestyle='--', linewidth=1.5,
                label='Breakpoint Semantico' if bp_idx == breakpoint_indices[0] else "")
    # Opzionale: aggiungi del testo per evidenziare il breakpoint
    # plt.text(bp_idx + 0.1, np.min(similarities) * 0.9, f'BP {bp_idx}', rotation=90, va='bottom', color='red')

plt.title(f'Similarità Semantica tra Frasi Adiacenti e Breakpoint ({BREAKPOINT_PERCENTILE_THRESHOLD}th Percentile)')
plt.xlabel('Indice della Frase (x vs x+1)')
plt.ylabel('Similarità Coseno')
plt.grid(True, linestyle=':', alpha=0.7)
plt.legend()
plt.tight_layout()

# Salva l'immagine
output_image_path = "semantic_similarity_plot.png"
plt.savefig(output_image_path)
print(f"Plot salvato come {output_image_path}")

plt.show()  # Mostra il plot