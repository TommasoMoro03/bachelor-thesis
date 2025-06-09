# evaluation/templatetags/dict_filters.py

from django import template

register = template.Library()

@register.filter
def get_item(dictionary, key):
    """
    Allows dictionary lookups in Django templates using dictionary|get_item:key.
    Example: {{ my_dict|get_item:my_key }}
    """
    return dictionary.get(key)

# You can add more custom filters here if needed in the future