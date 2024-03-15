from django.shortcuts import render

# Create your views here.
from django.http import HttpResponse

from django.http import Http404
from django.template import loader

from .models import RSSSource

# context = {"latest_question_list": latest_question_list}
#     return render(request, "polls/index.html", context)

def index(request):
    feeds_list = RSSSource.objects.order_by("-pub_date")[:5]
    context = {
        "feeds_list": feeds_list,
    }
    return render(request, "feeds/index.html", context)

def detail(request, entry_id):
    return HttpResponse("You're looking at entry %s." % entry_id)


def results(request, entry_id):
    response = "You're looking at the results of question %s."
    return HttpResponse(response % entry_id)


def update(request):
    response = "Updating feeds."
    feeds_list = RSSSource.objects.order_by("-pub_date")
    for feed in feeds_list:
        print(feed)
    print("response test")
    return HttpResponse(response)
    
