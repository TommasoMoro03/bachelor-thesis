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
    # Aggiungeremo altri URL qui in seguito, se necessario
]