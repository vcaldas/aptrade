import feedparser
import logging
from celery.result import AsyncResult
from django.http import JsonResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
logger = logging.getLogger(__name__)
from rssfeed.sample_tasks import create_task

# Create your views here.
from django.http import HttpResponse

from .models import Feed, Publication

def index(request):
    feeds_list = Feed.objects.order_by("-last_updated")[:5]
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
    print(response)
    feeds_list = Feed.objects.order_by("-last_updated")
    for feed in feeds_list:
        url = feed.url
        etag = feed.etag
        modified = feed.modified
        
        NewsFeed = feedparser.parse(url)
        
        response_code = feedparser.parse(url, modified=modified, etag=modified)
        if response_code.status == 304:
            print("Feed not modified")
        else:
            print("Feed modified")
            # feed.last_updated = NewsFeed.get('updated', '')
            feed.etag = NewsFeed.get('etag', '')
            feed.modified = NewsFeed.get('modified', '')
            feed.save()
            entries = NewsFeed.get('entries', {})
            
            for entry in entries:
                p = Publication(
                    id=entry['id'],
                    title=entry['title'],
                    link=entry['link'],
                    summary=entry['summary'],
                    rssfeed = feed
                )
                p.save()
    logger.info("response test")
    return HttpResponse(f"Finish Update example")
    


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


from django.shortcuts import render
from rest_framework.decorators import api_view
from rest_framework.response import Response

@api_view(['GET'])
def apitest(request):
    return Response({'message': 'Hello, world!'})