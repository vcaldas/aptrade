from django.shortcuts import render
import feedparser
# Create your views here.
from django.http import HttpResponse

from django.http import Http404
from django.template import loader

from .models import Feed, Publication

# context = {"latest_question_list": latest_question_list}
#     return render(request, "polls/index.html", context)

def index(request):
    feeds_list = Feed.objects.order_by("-pub_date")[:5]
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
    feeds_list = Feed.objects.order_by("-pub_date")
    print(feeds_list)
    for feed in feeds_list:
        url = feed.url
        NewsFeed = feedparser.parse(url)
        entries = NewsFeed.get('entries', {})
        
        for entry in entries:
            p = Publication(
                title=entry['title'],
                link=entry['link'],
                summary=entry['summary'],
                Feed = feed
            )
            print(p)
            # p.save()
    print("response test")
    return HttpResponse(f"Finish Update example")
    

from celery.result import AsyncResult
from django.http import JsonResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt

from rssfeed.sample_tasks import create_task


def home(request):
    return render(request, "home.html")


@csrf_exempt
def run_task(request):
    if request.POST:
        task_type = request.POST.get("type")
        task = create_task.delay(int(task_type))
        return JsonResponse({"task_id": task.id}, status=202)


@csrf_exempt
def get_status(request, task_id):
    task_result = AsyncResult(task_id)
    result = {
        "task_id": task_id,
        "task_status": task_result.status,
        "task_result": task_result.result
    }
    return JsonResponse(result, status=200)