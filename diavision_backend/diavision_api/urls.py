from django.urls import path
from . import views

urlpatterns = [
    path('hera/chat/', views.hera_chat, name='hera_chat'),
    path('predict/tabular/', views.predict_tabular, name='predict_tabular'),
    path('predict/vision/', views.predict_vision, name='predict_vision'),
    path('diagnostics/shap-dependence/', views.shap_dependence_diagnostic, name='shap_dependence'),
]