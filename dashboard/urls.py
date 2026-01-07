from django.urls import path
from . import views

app_name = 'dashboard'

urlpatterns = [
    path('', views.introduccion, name='introduccion'),
    path('visualizacion/', views.visualizacion, name='visualizacion'),
    path('division/', views.division, name='division'),
    path('preparacion/', views.preparacion, name='preparacion'),
    path('pipelines/', views.pipelines, name='pipelines'),
    path('evaluacion/', views.evaluacion, name='evaluacion'),
]
