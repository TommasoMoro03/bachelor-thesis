from django import forms
from .models import SourceText, Question
import json


class SourceTextForm(forms.ModelForm):
    class Meta:
        model = SourceText
        fields = ['title', 'file', 'metadata']
        widgets = {
            'metadata': forms.Textarea(attrs={
                'rows': 3,
                'placeholder': 'Opzionale: Inserisci metadati in formato JSON (es. {"source": "Example Source", "author": "John Doe"})'
            }),
        }
        help_texts = {
            'file': 'Sono permessi solo file con estensione .txt.',
        }

    # Aggiungi validazione per il campo metadata JSON
    def clean_metadata(self):
        metadata = self.cleaned_data.get('metadata')
        # Se l'utente ha inserito qualcosa ma non è un JSON valido
        if metadata and isinstance(metadata, str): # Viene passato come stringa dal Textarea
             try:
                 # Tentiamo di caricarlo per vedere se è JSON valido
                 json.loads(metadata)
             except json.JSONDecodeError:
                 # Se non è valido, solleviamo un errore di validazione
                 raise forms.ValidationError("Il formato dei metadati non è JSON valido.")
        # Se è vuoto o già un dict (improbabile da Textarea ma per sicurezza), va bene
        return metadata


class QuestionForm(forms.ModelForm):
    class Meta:
        model = Question
        fields = ['text'] # Il source_text verrà impostato nella view
        widgets = {
            'text': forms.Textarea(attrs={
                'rows': 3,
                'placeholder': 'Inserisci qui la domanda relativa a questo testo...'
            }),
        }
        labels = {
            'text': 'Nuova Domanda' # Etichetta più chiara per il campo
        }