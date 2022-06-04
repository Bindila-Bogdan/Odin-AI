from django.urls import path

from . import views

urlpatterns = [path('', views.index, name='index'),
               path('train/', views.train, name='train'),
               path('test/', views.test, name='test'),
               path('delete_data/', views.delete_data, name='delete_data')]
