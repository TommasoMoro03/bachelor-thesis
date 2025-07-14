# evaluation/urls.py
from django.urls import path
from . import views

app_name = 'evaluation'

urlpatterns = [
    # URL principale per visualizzare/gestire l'analisi di un Exp/ChunkSet specifico
    path(
        'analyze/experiment/<int:experiment_pk>/chunkset/<int:chunk_set_pk>/',
        views.evaluation_detail_view, # Nome della view che creeremo
        name='detail'
    ),
    path(
        'results/',
        views.view_evaluation_results, # This is the new view function
        name='view_results'
    ),
    path(
        'statistical_analysis/',
        views.run_statistical_analysis_view,
        name='statistical_analysis'
    ),
]