# corpus/models.py
import os
from django.db import models
from django.core.exceptions import ValidationError
from django.conf import settings


# Validator per controllare l'estensione del file
def validate_txt_extension(value):
    ext = os.path.splitext(value.name)[1] # Ottiene l'estensione .txt
    valid_extensions = ['.txt']
    if not ext.lower() in valid_extensions:
        raise ValidationError('Estensione file non supportata. Sono permessi solo file .txt.')


# Funzione per definire il percorso di upload relativo a MEDIA_ROOT
def source_text_upload_path(instance, filename):
    # Il file verrà caricato in MEDIA_ROOT/source_texts/<nome_originale_file>
    # Assicurati che MEDIA_ROOT sia definito in settings.py
    return f'source_texts/{filename}'


class SourceText(models.Model):
    """Rappresenta un file di testo sorgente originale."""
    title = models.CharField(
        max_length=255,
        unique=True,
        help_text="Titolo univoco per identificare il testo."
    )
    file = models.FileField(
        upload_to=source_text_upload_path,
        validators=[validate_txt_extension],
        help_text='Carica un file .txt.',
        default=None
    )
    metadata = models.JSONField(
        null=True,
        blank=True,
        help_text='Opzionale: metadati come fonte, autore, ecc. in formato JSON.'
    )
    uploaded_at = models.DateTimeField(auto_now_add=True)

    # Metodo helper per leggere il contenuto del file facilmente
    def read_content(self):
        try:
            # Assicurati che il file esista prima di tentare di aprirlo
            if self.file and hasattr(self.file, 'path') and os.path.exists(self.file.path):
                 # Usiamo 'with' per assicurare che il file venga chiuso correttamente
                with self.file.open('r', encoding='utf-8') as f:
                    return f.read()
            else:
                return "Errore: File non trovato sul percorso specificato."
        except Exception as e:
            # Logga l'errore per debug futuro se necessario
            # logger.error(f"Errore durante la lettura del file {self.file.name}: {e}")
            return f"Errore durante la lettura del file: {e}"

    # Metodo per ottenere il nome del file senza il percorso
    @property
    def filename(self):
        return os.path.basename(self.file.name)


class Question(models.Model):
    """Rappresenta una domanda relativa a un SourceText."""
    # Collegamento al SourceText a cui si riferisce la domanda
    source_text = models.ForeignKey(
        SourceText,
        related_name='questions', # Permette accesso inverso: source_text.questions.all()
        on_delete=models.CASCADE
    )
    text = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = ('source_text', 'text')