# experiments/forms.py
from django import forms
from .models import ChunkingStrategy
import json

class ChunkingStrategyForm(forms.ModelForm):
    parameters = forms.CharField(
        widget=forms.Textarea(attrs={'rows': 5}),
        help_text='Inserisci i parametri in formato JSON valido. Es: {"chunk_size": 512, "chunk_overlap": 50}',
        required=True
    )

    class Meta:
        model = ChunkingStrategy
        fields = ['name', 'method_type', 'parameters']
        labels = {
            'name': 'Nome Strategia',
            'method_type': 'Tipo di Metodo',
            'parameters': 'Parametri (JSON)',
        }

    def clean_parameters(self):
        """Valida che il campo parameters contenga JSON valido."""
        params_str = self.cleaned_data.get('parameters')
        try:
            # Tenta di parsare il JSON per validarlo
            parsed_params = json.loads(params_str)
            if not isinstance(parsed_params, dict):
                 raise forms.ValidationError("I parametri devono essere un oggetto JSON valido (es. { ... }).")
            # Potresti aggiungere qui validazioni specifiche per i parametri
            # basate sul method_type selezionato, se necessario.
            # Esempio:
            # method_type = self.cleaned_data.get('method_type')
            # if method_type == 'length' and 'chunk_size' not in parsed_params:
            #     raise forms.ValidationError("Per il tipo 'length', 'chunk_size' è obbligatorio.")
            return parsed_params # Ritorna il dizionario Python parsato
        except json.JSONDecodeError:
            raise forms.ValidationError("Formato JSON non valido.")
        except TypeError:
             raise forms.ValidationError("Input parametri non valido.")

    # Se stai modificando, potresti voler visualizzare il JSON formattato
    # def __init__(self, *args, **kwargs):
    #     super().__init__(*args, **kwargs)
    #     if self.instance and self.instance.pk and isinstance(self.instance.parameters, dict):
    #         self.initial['parameters'] = json.dumps(self.instance.parameters, indent=2)