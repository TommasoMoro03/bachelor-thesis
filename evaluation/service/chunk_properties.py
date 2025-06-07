from django.db import transaction
import math

from evaluation.models import RankedRelevantChunk, ExperimentChunkAnalysis


@transaction.atomic
def calculate_chunk_properties(analysis: ExperimentChunkAnalysis):
    """
    Calcola e salva w, Density, e w' per tutti i RankedRelevantChunk
    associati a questa analisi, assumendo che ideal_rank sia già impostato.
    """
    print(f"Calcolo proprietà per Analysis ID: {analysis.id}") # Log

    ranked_chunks = analysis.ranked_relevant_chunks.select_related('chunk').filter(ideal_rank__isnull=False)
    relevant_sentences = analysis.experiment.relevant_sentences.all()
    k_relevant = analysis.k_relevant

    if k_relevant is None or k_relevant == 0:
        print("k_relevant non definito o zero, nessun calcolo necessario.")
        return # Non c'è nulla da calcolare

    if not ranked_chunks.exists():
        print("Nessun chunk rilevante rankato trovato per l'analisi.")
        return

    if not relevant_sentences.exists():
        print("Nessuna frase rilevante trovata per l'esperimento, Density sarà 0.")
        # Imposta w' a 0 per tutti? O solleva errore? Impostiamo a 0.
        for rrc in ranked_chunks:
            rrc.intrinsic_importance_w = k_relevant - rrc.ideal_rank + 1
            rrc.relevance_density = 0.0
            rrc.effective_relevance_w_prime = 0.0
        RankedRelevantChunk.objects.bulk_update(ranked_chunks, ['intrinsic_importance_w', 'relevance_density', 'effective_relevance_w_prime'])
        return

    # Prepara lista per bulk update
    chunks_to_update = []
    for rrc in ranked_chunks:
        # 1. Calcola w(c) (Intrinsic Importance)
        # Assicurati che ideal_rank sia valido (dovrebbe esserlo data la query filter)
        if rrc.ideal_rank is None or rrc.ideal_rank < 1 or rrc.ideal_rank > k_relevant:
             print(f"WARN: Rank non valido ({rrc.ideal_rank}) per RRC ID {rrc.id}. Salto calcolo proprietà.")
             continue # Salta questo chunk se il rank non è valido

        rrc.intrinsic_importance_w = float(k_relevant - rrc.ideal_rank + 1)

        # 2. Calcola Density(c)
        chunk = rrc.chunk
        chunk_start = chunk.start_char
        chunk_end = chunk.end_char
        chunk_length = float(chunk.length) # Usa float per la divisione
        overlapping_sentences_length = 0

        if chunk_length > 0: # Evita divisione per zero
            for sentence in relevant_sentences:
                sent_start = sentence.start_char
                sent_end = sentence.end_char
                # Check overlap
                if sent_start < chunk_end and sent_end > chunk_start:
                    # Calcola la lunghezza della *parte* della frase che cade nel chunk
                    overlap_start = max(sent_start, chunk_start)
                    overlap_end = min(sent_end, chunk_end)
                    overlapping_sentences_length += (overlap_end - overlap_start)

            rrc.relevance_density = overlapping_sentences_length / chunk_length
        else:
            rrc.relevance_density = 0.0 # Densità zero se il chunk ha lunghezza zero

        # 3. Calcola w'(c) (Effective Relevance)
        # Gestisci densità negativa o > 1 (non dovrebbe accadere ma per sicurezza)
        density = max(0.0, min(1.0, rrc.relevance_density))
        # Usa math.sqrt, assicurati che l'argomento non sia negativo (già fatto con max)
        rrc.effective_relevance_w_prime = rrc.intrinsic_importance_w * math.sqrt(density)

        chunks_to_update.append(rrc)
        # print(f"RRC ID {rrc.id}: Rank={rrc.ideal_rank}, w={rrc.intrinsic_importance_w:.2f}, Density={rrc.relevance_density:.3f}, w'={rrc.effective_relevance_w_prime:.3f}") # Debug log

    # Salva tutti gli aggiornamenti in un colpo solo
    if chunks_to_update:
        updated_fields = ['intrinsic_importance_w', 'relevance_density', 'effective_relevance_w_prime']
        RankedRelevantChunk.objects.bulk_update(chunks_to_update, updated_fields)
        print(f"Aggiornate proprietà per {len(chunks_to_update)} oggetti RankedRelevantChunk.")