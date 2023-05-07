from django.urls import path

from .views import *

urlpatterns = [
    path('', index, name='index'),
    path('ajax_request/', ajax_request, name='my_ajax_view'),
]
