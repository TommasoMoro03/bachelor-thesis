# chunking_thesis/views.py
from django.shortcuts import render


def dashboard_view(request):
    # Per ora, la vista rende solo il template.
    # In futuro potresti passare dati riassuntivi.
    context = {}
    return render(request, 'dashboard.html', context)