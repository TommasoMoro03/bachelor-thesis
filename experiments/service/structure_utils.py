import nltk

from typing import List, Dict, Any
import re


def _pure_paragraph_split(content: str, paragraph_separator: str) -> List[Dict[str, Any]]:
    """Splitta il contenuto in chunk basati su paragrafi, indipendentemente dalla lunghezza."""
    escaped_sep = paragraph_separator.replace('\n', '\\n')
    print(f"Custom: Esecuzione Pure Paragraph Split con separatore: '{escaped_sep}'")
    paragraphs = re.split(r'\n\s*\n+', content)

    chunks_data_list = []
    current_char_offset = 0
    for para in paragraphs:
        cleaned_para = para.strip()
        # Calcola la lunghezza del paragrafo originale + separatore per avanzare l'offset
        # Questo è importante anche per paragrafi vuoti o solo spazi
        full_para_len = len(para) + len(paragraph_separator)

        if not cleaned_para:
            current_char_offset += full_para_len
            continue

        # Trova la posizione esatta del paragrafo pulito nel testo originale
        # Questo è più robusto che incrementare un offset fisso se ci sono variazioni di spazio
        start_idx = content.find(cleaned_para, current_char_offset)

        if start_idx == -1:
            # Fallback se non si trova (es. encoding diverso o pulizia aggressiva)
            # Potrebbe significare che il 'cleaned_para' non è esattamente una sottostringa di 'content'
            print(
                f"WARN: Paragrafo non trovato nel testo originale a partire da offset {current_char_offset}: '{cleaned_para[:50]}...'")
            # In un caso reale, potresti voler gestire questo errore in modo più robusto
            # Per ora, stimiamo l'offset in base alla lunghezza del testo originale processato
            if chunks_data_list:
                start_idx = chunks_data_list[-1]['end_char'] + len(paragraph_separator)
            else:
                start_idx = current_char_offset  # Ultimo offset noto
            end_idx = start_idx + len(cleaned_para)
        else:
            end_idx = start_idx + len(cleaned_para)

        chunks_data_list.append({
            'text': cleaned_para,
            'start_char': start_idx,
            'end_char': end_idx,
            'metadata': {'type': 'pure_paragraph'}
        })
        # Avanza l'offset per la prossima ricerca, includendo il separatore dopo il paragrafo trovato
        current_char_offset = end_idx + len(paragraph_separator)

    return chunks_data_list


def _n_sentence_chunking(content: str, sentences_per_chunk: int, sentence_overlap: int) -> List[Dict[str, Any]]:
    """Splitta il contenuto in chunk basati su un numero fisso di frasi con overlap."""
    print(f"Custom: Esecuzione N-Sentence Chunking: {sentences_per_chunk} frasi, {sentence_overlap} overlap")
    sentences = nltk.sent_tokenize(content)

    chunks_data_list = []
    i = 0
    while i < len(sentences):
        chunk_sentences = sentences[i: i + sentences_per_chunk]
        if not chunk_sentences:
            break

        chunk_text = " ".join(chunk_sentences)

        # Trova la posizione esatta del chunk nel testo originale
        # Questo richiede di trovare l'inizio della prima frase e la fine dell'ultima frase del chunk
        start_char = content.find(chunk_sentences[0])
        if start_char == -1:
            print(f"WARN: Prima frase del chunk non trovata: '{chunk_sentences[0][:50]}...'")
            # Fallback approssimativo
            if chunks_data_list:
                start_char = chunks_data_list[-1]['end_char']
            else:
                start_char = 0
            end_char = start_char + len(chunk_text)
        else:
            # Trova la fine dell'ultima frase nel content originale, a partire dalla posizione della prima
            # Questo è più robusto se ci sono spazi extra o normalizzazioni di nltk
            last_sentence = chunk_sentences[-1]
            end_of_last_sentence_in_content = content.find(last_sentence, start_char + len(
                chunk_sentences[0]))  # Cerca da dopo la prima frase
            if end_of_last_sentence_in_content == -1:
                end_of_last_sentence_in_content = content.find(last_sentence,
                                                               start_char)  # Se non trovata dopo, cerca da inizio chunk
                if end_of_last_sentence_in_content == -1:
                    print(f"WARN: Ultima frase del chunk non trovata: '{last_sentence[:50]}...'")
                    end_char = start_char + len(chunk_text)  # Fallback approssimativo
                else:
                    end_char = end_of_last_sentence_in_content + len(last_sentence)
            else:
                end_char = end_of_last_sentence_in_content + len(last_sentence)

        chunks_data_list.append({
            'text': chunk_text,
            'start_char': start_char,
            'end_char': end_char,
            'metadata': {'type': f'{sentences_per_chunk}-sentence-chunk'}
        })

        i += (sentences_per_chunk - sentence_overlap)
        if i < 0:
            i = 0  # Assicura che l'indice non diventi negativo

    return chunks_data_list


def _sentence_window_chunking(content: str, min_chars_per_chunk: int, max_chars_per_chunk: int,
                              sentence_overlap_chars: int) -> List[Dict[str, Any]]:
    """Splitta il contenuto in chunk basati su una finestra di caratteri, rispettando i confini delle frasi."""
    print(
        f"Custom: Esecuzione Sentence Window Chunking: min={min_chars_per_chunk}, max={max_chars_per_chunk}, overlap={sentence_overlap_chars}")
    sentences = nltk.sent_tokenize(content)

    chunks_data_list = []
    current_sentence_idx = 0

    while current_sentence_idx < len(sentences):
        current_chunk_sentences = []
        current_chunk_len = 0
        start_chunk_sentence_idx = current_sentence_idx  # Per calcolare l'overlap

        # Accumula frasi finché non si raggiunge il minimo o si esauriscono le frasi
        while (current_chunk_len < min_chars_per_chunk or not current_chunk_sentences) and current_sentence_idx < len(
                sentences):
            sentence_to_add = sentences[current_sentence_idx]
            current_chunk_sentences.append(sentence_to_add)
            current_chunk_len += len(sentence_to_add) + 1  # +1 per lo spazio implicito
            current_sentence_idx += 1

        # Continua ad aggiungere frasi finché non si supera il massimo, senza spezzare l'ultima frase
        while current_chunk_len < max_chars_per_chunk and current_sentence_idx < len(sentences):
            next_sentence = sentences[current_sentence_idx]
            # Aggiungi solo se non sfora eccessivamente il massimo
            if current_chunk_len + len(next_sentence) + 1 <= max_chars_per_chunk * 1.1:  # Piccolo buffer flessibile
                current_chunk_sentences.append(next_sentence)
                current_chunk_len += len(next_sentence) + 1
                current_sentence_idx += 1
            else:
                break

        if not current_chunk_sentences:  # Evita chunk vuoti
            break

        chunk_text = " ".join(current_chunk_sentences)

        # Calcolo robusto degli indici (similare a _n_sentence_chunking)
        start_char = content.find(current_chunk_sentences[0])
        if start_char == -1:
            print(f"WARN: Prima frase del chunk non trovata: '{current_chunk_sentences[0][:50]}...'")
            if chunks_data_list:
                start_char = chunks_data_list[-1]['end_char']
            else:
                start_char = 0
            end_char = start_char + len(chunk_text)
        else:
            last_sentence = current_chunk_sentences[-1]
            end_of_last_sentence_in_content = content.find(last_sentence, start_char + len(current_chunk_sentences[0]))
            if end_of_last_sentence_in_content == -1:
                end_of_last_sentence_in_content = content.find(last_sentence, start_char)
                if end_of_last_sentence_in_content == -1:
                    print(f"WARN: Ultima frase del chunk non trovata: '{last_sentence[:50]}...'")
                    end_char = start_char + len(chunk_text)
                else:
                    end_char = end_of_last_sentence_in_content + len(last_sentence)
            else:
                end_char = end_of_last_sentence_in_content + len(last_sentence)

        chunks_data_list.append({
            'text': chunk_text,
            'start_char': start_char,
            'end_char': end_char,
            'metadata': {'type': 'sentence_window'}
        })

        overlap_sentences_count = 0
        current_overlap_length = 0
        # Itera all'indietro per vedere quante frasi coprono l'overlap desiderato
        for k in range(len(current_chunk_sentences) - 1, -1, -1):
            if current_overlap_length + len(current_chunk_sentences[k]) + 1 <= sentence_overlap_chars:
                current_overlap_length += len(current_chunk_sentences[k]) + 1
                overlap_sentences_count += 1
            else:
                break

        # Il prossimo chunk dovrebbe iniziare da:
        # l'inizio del chunk corrente + (numero di frasi nel chunk corrente - numero di frasi in overlap)
        next_start_sentence_idx = start_chunk_sentence_idx + (len(current_chunk_sentences) - overlap_sentences_count)

        # ASSICURATI CHE L'INDICE AVANZI SEMPRE DI ALMENO UNA POSIZIONE REALE
        # Se next_start_sentence_idx non è maggiore di start_chunk_sentence_idx, avanziamo di 1
        if next_start_sentence_idx <= start_chunk_sentence_idx:
            # Avanza di almeno 1 frase per garantire che il loop termini
            current_sentence_idx = start_chunk_sentence_idx + 1
        else:
            current_sentence_idx = next_start_sentence_idx

        # Se siamo arrivati alla fine delle frasi ma c'è ancora un pezzo da aggiungere,
        # lo gestirà il prossimo ciclo o uscirà.
        if current_sentence_idx >= len(sentences) and len(chunks_data_list) > 0 and len(current_chunk_sentences) > 0:
            # Ultimo chunk, assicurati di non creare un loop infinito se l'overlap impedisce l'avanzamento
            # Questo break è un fail-safe per gli ultimi frammenti.
            break

    return chunks_data_list