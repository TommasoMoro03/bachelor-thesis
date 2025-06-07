from corpus.models import SourceText

# recupero MEDIA DIR
from chunking_thesis import settings

import os

def get_media_dir() -> str:
    """
    Recupera la directory dei media.
    """
    return settings.MEDIA_ROOT

def get_full_text(source_text: SourceText) -> str:
    file_path = source_text.file
    media_dir = get_media_dir()
    full_path = os.path.join(media_dir, file_path.name)
    # Leggi il file e restituisci il contenuto
    with open(full_path, 'r', encoding='utf-8') as file:
        content = file.read()
    return content
