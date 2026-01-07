from django import template
import json

register = template.Library()

@register.filter
def get_item(dictionary, key):
    """Obtiene un item de un diccionario por clave"""
    if isinstance(dictionary, dict):
        return dictionary.get(key, '')
    return ''

@register.filter
def to_json(value):
    """Convierte a JSON string"""
    try:
        return json.dumps(value)
    except:
        return '{}'
