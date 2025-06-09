# experiments/service/retrieval_simulation.py
import math

from django.db import transaction

# LlamaIndex Imports
from llama_index.core import VectorStoreIndex, QueryBundle, Settings  # Import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.schema import TextNode

# Import your Django models
from evaluation.models import ExperimentChunkAnalysis, RetrievedChunk, RetrievalSimulation
from evaluation.service.relevant_chunks import initialize_analysis

GLOBAL_EMBED_MODEL_NAME = "BAAI/bge-small-en-v1.5"
_embed_model = None


def get_global_embed_model():
    """Loads and returns the embedding model, memoizing it for reuse."""
    global _embed_model
    if _embed_model is None:
        print(f"Loading embedding model for retrieval: {GLOBAL_EMBED_MODEL_NAME}...")
        try:
            _embed_model = HuggingFaceEmbedding(model_name=GLOBAL_EMBED_MODEL_NAME)
            print("Embedding model loaded for retrieval.")
        except Exception as e:
            print(f"CRITICAL ERROR: Could not load embedding model '{GLOBAL_EMBED_MODEL_NAME}': {e}")
            raise ValueError(f"Could not load embedding model: {GLOBAL_EMBED_MODEL_NAME}. Details: {e}") from e
    return _embed_model


@transaction.atomic
def run_retrieval_simulation(analysis: ExperimentChunkAnalysis):
    """
    Executes a retrieval simulation using LlamaIndex Vector Retriever for a given ExperimentChunkAnalysis.
    Creates a RetrievalSimulation object and its associated RetrievedChunk objects.
    Returns the created RetrievalSimulation object.
    """
    print(f"Starting REAL retrieval simulation for Analysis ID: {analysis.id}")

    # Set the retriever and embedding model names for the simulation record
    retriever_name = "LlamaIndexVectorRetriever"
    embedding_model_name = GLOBAL_EMBED_MODEL_NAME  # Use the global model name

    # 1. Determine k_retrieved (as before)
    if analysis.k_relevant is None:
        # If k_relevant is not calculated, call the function to initialize it
        print("k_relevant is not calculated for this analysis. Calling initialize_analysis...")
        initialize_analysis(analysis)  # Initialize k_relevant based on relevant sentences
        # Reload the analysis object to get the updated k_relevant value
        analysis = ExperimentChunkAnalysis.objects.get(pk=analysis.pk)
        print(f"k_relevant re-calculated: {analysis.k_relevant}")

    k_retrieved_target = max(10, 2 * analysis.k_relevant) if analysis.k_relevant > 0 else 10
    print(f"Target k_retrieved: {k_retrieved_target}")

    # 2. Get all chunks from the chunk_set associated with the analysis
    # Convert them to LlamaIndex TextNodes for indexing
    all_chunks_in_set = analysis.chunk_set.chunks.all().order_by('chunk_index')

    if not all_chunks_in_set:
        print("No chunks in the chunk_set. Cannot perform simulation.")
        simulation = RetrievalSimulation.objects.create(
            analysis=analysis,
            retriever_name=retriever_name,
            embedding_model_name=embedding_model_name,
            k_retrieved=0,
            rdsg_score=None
        )
        return simulation

    # Convert your Django Chunk objects into LlamaIndex TextNodes
    llama_nodes = []
    for chunk_obj in all_chunks_in_set:
        # It's CRUCIAL that the LlamaIndex node ID is the PK of your Django Chunk object.
        # This allows mapping retriever results back to your original Chunk objects.
        llama_nodes.append(TextNode(
            text=chunk_obj.text,
            id_=str(chunk_obj.pk),
            metadata={
                'chunk_pk': chunk_obj.pk,
                'chunk_index': chunk_obj.chunk_index,
                'start_char': chunk_obj.start_char,
                'end_char': chunk_obj.end_char,
                # You can add other useful metadata here for the retriever or debugging
            }
        ))

    # 3. Initialize the embedding model and configure LlamaIndex global settings
    embed_model = get_global_embed_model()

    # Configure LlamaIndex global settings using Settings
    print(f"Configuring LlamaIndex global settings with embed_model: {GLOBAL_EMBED_MODEL_NAME}...")
    Settings.embed_model = embed_model
    Settings.llm = None  # Set LLM to None as it's not needed for just retrieval embedding operations
    print("LlamaIndex global settings configured.")

    # 4. Create an in-memory VectorStoreIndex
    # The index will automatically use the embed_model configured in Settings.
    print("Creating VectorStoreIndex with chunks...")
    index = VectorStoreIndex(nodes=llama_nodes)
    print("VectorStoreIndex created.")

    # 5. Get the retriever
    # similarity_top_k determines how many top similar chunks to retrieve.
    retriever = index.as_retriever(similarity_top_k=k_retrieved_target)
    print(f"Retriever created with top_k={k_retrieved_target}.")

    # 6. Execute the query
    query_text = analysis.experiment.question.text  # The question text from the experiment
    print(f"Executing query: '{query_text[:50]}...'")

    # QueryBundle encapsulates the query string for LlamaIndex
    query_bundle = QueryBundle(query_str=query_text)

    # The retriever returns a list of NodeWithScore objects
    retrieved_results = retriever.retrieve(query_bundle)
    print(f"Retriever returned {len(retrieved_results)} results.")

    # 7. Process the retriever results
    retrieved_chunks_data = []  # List of {'chunk': Chunk, 'score': float, 'rank': int}

    # Map Django Chunk PKs to their objects for quick lookup
    chunk_pk_to_obj_map = {str(c.pk): c for c in all_chunks_in_set}

    for i, node_with_score in enumerate(retrieved_results):
        node = node_with_score.node
        score = node_with_score.score

        # Retrieve the original Django Chunk object using the PK stored in node.id_
        original_chunk_pk = node.id_
        original_chunk_obj = chunk_pk_to_obj_map.get(original_chunk_pk)

        if original_chunk_obj:
            retrieved_chunks_data.append({
                'chunk': original_chunk_obj,
                'score': score,
                'rank': i + 1  # Rank 1-based
            })
        else:
            print(
                f"WARN: Chunk with PK {original_chunk_pk} from retriever results not found in the original chunk_set.")

    # 8. Create the RetrievalSimulation object in the DB
    simulation = RetrievalSimulation.objects.create(
        analysis=analysis,
        retriever_name=retriever_name,
        embedding_model_name=embedding_model_name,
        k_retrieved=len(retrieved_chunks_data)  # Actual number of chunks retrieved
        # rdsg_score will be calculated in a subsequent step
    )
    print(f"Created RetrievalSimulation ID: {simulation.id}.")

    # 9. Bulk create the RetrievedChunk objects
    retrieved_chunk_objects_to_create = []
    for data in retrieved_chunks_data:
        retrieved_chunk_objects_to_create.append(
            RetrievedChunk(
                simulation=simulation,
                chunk=data['chunk'],
                retrieved_rank=data['rank'],
                similarity_score_s=data['score']
            )
        )

    if retrieved_chunk_objects_to_create:
        RetrievedChunk.objects.bulk_create(retrieved_chunk_objects_to_create)
        print(
            f"Created {len(retrieved_chunk_objects_to_create)} RetrievedChunk objects for simulation ID: {simulation.id}.")
    else:
        print("No RetrievedChunk objects to create.")

    return simulation


@transaction.atomic
def calculate_rdsg_and_ndcg(simulation: RetrievalSimulation): # Renamed function
    """
    Calculates the RDSG, Ideal RDSG, and NDCG scores for a given RetrievalSimulation.
    Updates and saves simulation.rdsg_score, simulation.ideal_rdsg_score, and simulation.ndcg_score.
    """
    print(f"Calculating RDSG and NDCG for Simulation ID: {simulation.id}")

    retrieved_chunks = simulation.retrieved_chunks.select_related('chunk').order_by('retrieved_rank')
    analysis = simulation.analysis

    # Fetch all RankedRelevantChunk objects for the analysis, ordered by ideal_rank
    ranked_relevant_chunks_qs = analysis.ranked_relevant_chunks.select_related('chunk').filter(
        ideal_rank__isnull=False, effective_relevance_w_prime__isnull=False
    ).order_by('ideal_rank')

    # Create a map for effective relevance (w') for quick lookup by chunk_id
    w_prime_map = {rrc.chunk_id: rrc.effective_relevance_w_prime for rrc in ranked_relevant_chunks_qs}

    # --- Calculate RDSG ---
    rdsg_sum = 0.0
    if not retrieved_chunks.exists():
        print("No chunks retrieved in this simulation. RDSG = 0.")
        rdsg_sum = 0.0
    else:
        for retrieved_chunk_item in retrieved_chunks:
            chunk_id = retrieved_chunk_item.chunk_id
            retrieved_rank_i = retrieved_chunk_item.retrieved_rank
            similarity_score_s_i = retrieved_chunk_item.similarity_score_s

            w_prime_c_i = w_prime_map.get(chunk_id, 0.0) # 0 if not relevant or w' not calculated

            denominator = math.log2(retrieved_rank_i + 1)
            term_value = (w_prime_c_i * similarity_score_s_i) / denominator if denominator > 0 else 0.0
            rdsg_sum += term_value
            # print(f"  RDSG Term: Rank {retrieved_rank_i}, ChunkID {chunk_id}, w'={w_prime_c_i:.3f}, s={similarity_score_s_i:.3f}, term={term_value:.3f}")

    # --- Calculate Ideal RDSG ---
    ideal_rdsg_sum = 0.0
    if not ranked_relevant_chunks_qs.exists():
        print("No ranked relevant chunks for ideal RDSG calculation. Ideal RDSG = 0.")
        ideal_rdsg_sum = 0.0
    else:
        # We use enumerate for the ideal rank (i) from 1, based on the ordering
        for idx, rrc_item in enumerate(ranked_relevant_chunks_qs):
            ideal_rank_i = idx + 1 # Ideal rank is 1-based, matches (i) in the formula
            ideal_w_c_i = rrc_item.intrinsic_importance_w

            ideal_denominator = math.log2(ideal_rank_i + 1)
            ideal_term_value = (ideal_w_c_i * 1.0) / ideal_denominator if ideal_denominator > 0 else 0.0
            ideal_rdsg_sum += ideal_term_value

    # --- Calculate NDCG ---
    ndcg_score = 0.0
    if ideal_rdsg_sum > 0:
        ndcg_score = rdsg_sum / ideal_rdsg_sum
    else:
        print("Ideal RDSG is zero, NDCG cannot be calculated (defaulting to 0).")
        # If ideal_rdsg_sum is 0, it means there are no relevant chunks or their w' values are all 0.
        # In this case, NDCG is undefined or 0.

    # Save results to the simulation object
    simulation.rdsg_score = rdsg_sum
    simulation.ideal_rdsg_score = ideal_rdsg_sum
    simulation.ndcg_score = ndcg_score
    simulation.save(update_fields=['rdsg_score', 'ideal_rdsg_score', 'ndcg_score'])

    print(f"RDSG calculated: {simulation.rdsg_score:.4f}")
    print(f"Ideal RDSG calculated: {simulation.ideal_rdsg_score:.4f}")
    print(f"NDCG calculated: {simulation.ndcg_score:.4f}")