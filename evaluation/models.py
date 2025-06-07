from django.db import models

from experiments.models import Chunk, ChunkSet, Experiment


class ExperimentChunkAnalysis(models.Model):
    """Stores the analysis results for a specific Experiment and ChunkSet."""
    experiment = models.ForeignKey(Experiment, on_delete=models.CASCADE)
    chunk_set = models.ForeignKey(ChunkSet, on_delete=models.CASCADE)
    # Calculated number of relevant chunks for this experiment/chunkset combo
    k_relevant = models.PositiveIntegerField(null=True, blank=True, help_text="|C_relevant|")
    analysis_time = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = ('experiment', 'chunk_set')


class RankedRelevantChunk(models.Model):
    """Stores properties of a chunk identified as relevant (C_relevant) for an analysis,
       including its manually assigned ideal rank and calculated scores."""
    analysis = models.ForeignKey(ExperimentChunkAnalysis, related_name='ranked_relevant_chunks', on_delete=models.CASCADE)
    chunk = models.ForeignKey(Chunk, on_delete=models.CASCADE)
    # Manually assigned rank based on perceived importance for the specific experiment
    ideal_rank = models.PositiveIntegerField(help_text="Position in Rank_expected (1-based)", null=True, blank=True)
    # Calculated scores based on ideal_rank and relevant sentences within the chunk
    intrinsic_importance_w = models.FloatField(null=True, blank=True, help_text="w(c)")
    relevance_density = models.FloatField(null=True, blank=True, help_text="Density(c)")
    effective_relevance_w_prime = models.FloatField(null=True, blank=True, help_text="w'(c) = w(c) * sqrt(Density(c))")

    class Meta:
        unique_together = ('analysis', 'chunk') # A chunk can only be relevant once per analysis
        ordering = ['ideal_rank']


class RetrievalSimulation(models.Model):
    """Stores the results of a single retrieval simulation run."""
    # Links to the specific analysis this simulation is based on
    analysis = models.ForeignKey(ExperimentChunkAnalysis, related_name='simulations', on_delete=models.CASCADE)
    # Details of the fixed retriever setup used for this simulation
    retriever_name = models.CharField(max_length=100, help_text="e.g., FAISS")
    embedding_model_name = models.CharField(max_length=150, help_text="e.g., all-MiniLM-L6-v2")
    # Parameters used for retrieval
    k_retrieved = models.PositiveIntegerField(help_text="Number of chunks retrieved")
    # The final calculated evaluation score
    rdsg_score = models.FloatField(null=True, blank=True, help_text="Calculated RDSG score")
    ran_at = models.DateTimeField(auto_now_add=True)
    ideal_rdsg_score = models.FloatField(null=True, blank=True, help_text="Ideal RDSG score for normalization")
    ndcg_score = models.FloatField(null=True, blank=True, help_text="Normalized DCG score (NDCG)")


class RetrievedChunk(models.Model):
    """Represents a single chunk retrieved in a simulation run, with its rank and score."""
    simulation = models.ForeignKey(RetrievalSimulation, related_name='retrieved_chunks', on_delete=models.CASCADE)
    chunk = models.ForeignKey(Chunk, on_delete=models.CASCADE)
    retrieved_rank = models.PositiveIntegerField(help_text="Position i in the retrieved list (1-based)")
    similarity_score_s = models.FloatField(help_text="Similarity score s_i from retriever")

    class Meta:
        unique_together = ('simulation', 'retrieved_rank') # Rank must be unique per simulation
        ordering = ['retrieved_rank']
