from django.urls import path

from . import views

urlpatterns = [
    path("", views.index, name="index"),
    path("update", views.update, name="update"),
    path("<int:entry_id>/", views.detail, name="detail"),
    # ex: /rssfeed/5/results/
    path("<int:entry_id>/results/", views.results, name="results"),
]
