from django.urls import path

from . import views

urlpatterns = [
    path("", views.index, name="index"),
    path("update", views.update, name="update"),
    path("<int:entry_id>/", views.detail, name="detail"),
    # ex: /rssfeed/5/results/
    path("<int:entry_id>/results/", views.results, name="results"),
    path("home", views.home, name="tasks"),
    path("tasks/<task_id>/", views.get_status, name="get_status"),
    path("tasks/", views.run_task, name="run_task"),
    path('hello-world/', views.apitest, name='apitest'),

]
