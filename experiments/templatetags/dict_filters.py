# experiments/templatetags/dict_filters.py
from django import template

register = template.Library()

@register.filter(name='get_item')
def get_item(dictionary, key):
    """Permette di fare dictionary.get(key) nei template Django"""
    # Controlla che sia un dizionario
    if isinstance(dictionary, dict):
        return dictionary.get(key)
    return None # o '' o gestisci l'errore come preferisci