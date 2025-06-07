# evaluation/services.py
from django.db import transaction
from experiments.models import Chunk # Import Chunk
from evaluation.models import ExperimentChunkAnalysis, RankedRelevantChunk # Importa modelli evaluation

@transaction.atomic # Assicura che tutte le operazioni nel DB avvengano o nessuna
def initialize_analysis(analysis: ExperimentChunkAnalysis):
    """
    Esegue l'analisi iniziale per un ExperimentChunkAnalysis:
    1. Identifica i Chunk rilevanti (C_relevant) confrontandoli con RelevantSentence.
    2. Calcola e salva k_relevant (|C_relevant|).
    3. Crea gli oggetti RankedRelevantChunk iniziali (senza rank o score).
    Restituisce il numero di chunk rilevanti trovati (k_relevant).
    """
    print(f"Inizializzazione Analisi per Analysis ID: {analysis.id}") # Log di debug

    experiment = analysis.experiment
    chunk_set = analysis.chunk_set

    # Recupera tutte le frasi rilevanti per l'esperimento
    relevant_sentences = experiment.relevant_sentences.all()
    if not relevant_sentences.exists():
        print("Nessuna frase rilevante definita per questo esperimento. k_relevant = 0.")
        analysis.k_relevant = 0
        analysis.save(update_fields=['k_relevant'])
        # Non creiamo RankedRelevantChunk se non ci sono frasi rilevanti
        return 0

    # Recupera tutti i chunk per questo chunk set
    all_chunks = chunk_set.chunks.all()
    if not all_chunks.exists():
        print("Nessun chunk trovato per questo chunk set. k_relevant = 0.")
        analysis.k_relevant = 0
        analysis.save(update_fields=['k_relevant'])
        return 0

    relevant_chunk_pks = set() # Usiamo un set per memorizzare i PK dei chunk rilevanti trovati

    # Itera per trovare le sovrapposizioni
    for chunk in all_chunks:
        chunk_start = chunk.start_char
        chunk_end = chunk.end_char
        is_relevant = False
        for sentence in relevant_sentences:
            sent_start = sentence.start_char
            sent_end = sentence.end_char
            # Condizione di sovrapposizione: (inizio_A < fine_B) AND (fine_A > inizio_B)
            if sent_start < chunk_end and sent_end > chunk_start:
                relevant_chunk_pks.add(chunk.pk)
                is_relevant = True
                # Trovata una sovrapposizione per questo chunk, possiamo passare al prossimo chunk
                break
        # if is_relevant: # Debug log
        #     print(f"Chunk ID {chunk.pk} ({chunk_start}-{chunk_end}) è RILEVANTE.")
        # else:
        #     print(f"Chunk ID {chunk.pk} ({chunk_start}-{chunk_end}) non è rilevante.")


    # Calcola k_relevant
    k_relevant = len(relevant_chunk_pks)
    print(f"k_relevant calcolato: {k_relevant}")

    # Aggiorna l'oggetto analysis nel DB
    analysis.k_relevant = k_relevant
    analysis.save()

    # Crea gli oggetti RankedRelevantChunk (solo se k_relevant > 0)
    # Usiamo get_or_create per sicurezza, nel caso questa funzione venga chiamata più volte
    created_count = 0
    if k_relevant > 0:
        relevant_chunks_queryset = Chunk.objects.filter(pk__in=relevant_chunk_pks)
        print(f"Creazione di {k_relevant} oggetti RankedRelevantChunk per l'analisi ID: {analysis.id}")
        for relevant_chunk in relevant_chunks_queryset:
            ranked_chunk, created = RankedRelevantChunk.objects.get_or_create(
                analysis=analysis,
                chunk=relevant_chunk,
                # Non impostiamo rank o score qui, verranno aggiunti dopo
                defaults={
                    'ideal_rank': None,
                    'intrinsic_importance_w': None,
                    'relevance_density': None,
                    'effective_relevance_w_prime': None
                }
            )
            if created:
                created_count += 1

    print(f"Creati {created_count} nuovi oggetti RankedRelevantChunk.")

    # Pulisci eventuali RankedRelevantChunk orfani (se un chunk non è più rilevante?)
    # Questo è più complesso, per ora assumiamo che la rilevanza non cambi una volta calcolata.
    # Potremmo aggiungere:
    # RankedRelevantChunk.objects.filter(analysis=analysis).exclude(chunk_id__in=relevant_chunk_pks).delete()

    return k_relevant