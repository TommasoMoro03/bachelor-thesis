# experiments/urls.py
from django.urls import path
from experiments.views import relevant_sentences_views as r_views
from experiments.views import chunking_views as c_views

app_name = 'experiments'

urlpatterns = [
    path('', r_views.list_experiments, name='list_experiments'),
    # Vista principale per annotazione (gestirà GET e POST)
    path('<int:experiment_pk>/annotate/', r_views.annotate_experiment_view, name='annotate_experiment'),

    # Strategie di Chunking (Nuovi)
    path('strategies/', c_views.list_strategies, name='list_strategies'),
    path('strategies/create/', c_views.create_strategy, name='create_strategy'),
    path('strategies/<int:pk>/edit/', c_views.edit_strategy, name='edit_strategy'),
    path('strategies/<int:pk>/delete/', c_views.delete_strategy, name='delete_strategy'),

    # Gestione Chunking per Documento (Nuovi)
    path('source-text/<int:source_text_pk>/chunking/', c_views.manage_document_chunking, name='manage_document_chunking'),
    path('source-text/<int:source_text_pk>/apply-strategy/<int:strategy_pk>/', c_views.apply_strategy_to_document, name='apply_strategy_to_document'), # POST request
    path('chunk-set/<int:chunk_set_pk>/view/', c_views.view_chunk_set, name='view_chunk_set'),
    path('chunk_sets/<int:chunk_set_pk>/delete/', c_views.delete_chunk_set, name='delete_chunk_set'),
]