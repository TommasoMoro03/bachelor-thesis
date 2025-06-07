
from django.db import models

from corpus.models import SourceText, Question


class Experiment(models.Model):
    """Links a SourceText and a Question, forming the basis for analysis."""
    source_text = models.ForeignKey(SourceText, on_delete=models.CASCADE)
    question = models.ForeignKey(Question, on_delete=models.CASCADE)
    description = models.TextField(blank=True, null=True) # Optional notes for the experiment
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        # Ensure each Question-SourceText pair is unique for experiments
        unique_together = ('source_text', 'question')


class RelevantSentence(models.Model):
    """Stores manually annotated relevant sentences (S_relevant) for an Experiment."""
    experiment = models.ForeignKey(Experiment, related_name='relevant_sentences', on_delete=models.CASCADE)
    text = models.TextField()
    # Store location within the original source text for density calculation
    start_char = models.PositiveIntegerField()
    end_char = models.PositiveIntegerField()
    annotation_time = models.DateTimeField(auto_now_add=True)

    class Meta:
        # Ensure each sentence is unique within the context of an experiment
        unique_together = ('experiment', 'start_char', 'end_char')
        ordering = ['start_char']


class ChunkingStrategy(models.Model):
    """Defines a specific chunking configuration (method and parameters)."""
    METHOD_CHOICES = [
        ('length', 'Length-based'),
        ('structure', 'Structure-based'),
        ('semantic', 'Semantic-based'),
    ]
    name = models.CharField(max_length=100, unique=True, help_text="Unique name, e.g., 'Fixed Size 512/50'")
    method_type = models.CharField(max_length=20, choices=METHOD_CHOICES)
    # Store parameters flexibly, e.g., {'chunk_size': 512, 'overlap': 50}
    parameters = models.JSONField(help_text="Parameters for the chunking method")

    def __str__(self):
        return self.name


class ChunkSet(models.Model):
    """Represents the collection of chunks produced by applying a Strategy to a SourceText."""
    source_text = models.ForeignKey(SourceText, on_delete=models.CASCADE)
    strategy = models.ForeignKey(ChunkingStrategy, on_delete=models.CASCADE)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        # A source_text + strategy combination should only produce one set of chunks
        unique_together = ('source_text', 'strategy')


class Chunk(models.Model):
    """Represents a single chunk of text within a ChunkSet."""
    chunk_set = models.ForeignKey(ChunkSet, related_name='chunks', on_delete=models.CASCADE)
    text = models.TextField()
    chunk_index = models.PositiveIntegerField(help_text="Order of the chunk within the set (0-based)")
    # Store location within the original source text for analysis
    start_char = models.PositiveIntegerField()
    end_char = models.PositiveIntegerField()
    metadata = models.JSONField(null=True, blank=True, help_text="Optional metadata from the chunker")

    class Meta:
        ordering = ['chunk_index']
        unique_together = ('chunk_set', 'chunk_index') # Ensure unique index per set

    @property
    def length(self):
        """Calcola la lunghezza del chunk basata su start_char e end_char."""
        if self.end_char is not None and self.start_char is not None:
            return self.end_char - self.start_char
        return 0