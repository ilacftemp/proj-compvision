from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),           # Upload + seleção de pontos + processamento inicial
    path('ajustar/', views.ajustar, name='ajustar'),  # AJAX para ajustar a máscara com sliders em tempo real
]
