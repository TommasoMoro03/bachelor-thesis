from django.urls import path
from . import views

app_name = 'corpus' # Namespace per evitare conflitti di nomi URL

urlpatterns = [
    # Pagina principale dell'app: lista dei testi caricati
    path('', views.list_source_texts, name='list_source_texts'),
    # Pagina per caricare un nuovo testo
    path('upload/', views.upload_source_text, name='upload_source_text'),
    # Pagina di dettaglio per un testo specifico (mostra domande, permette aggiunta)
    path('<int:pk>/', views.source_text_detail, name='source_text_detail'),
]