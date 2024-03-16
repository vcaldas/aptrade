from django.db import models
from django.utils import timezone
import datetime
# Create your models here.
class Publication(models.Model):
    title = models.CharField(max_length=200)
    link = models.CharField(max_length=200)
    description = models.CharField(max_length=200)
    summary = models.CharField(max_length=200)
    pub_date = models.DateTimeField("date published", default=timezone.now)
    rssfeed = models.ForeignKey('RSSSource', on_delete=models.CASCADE)

    def __str__(self):
        return self.title   

class RSSSource(models.Model):
    name = models.CharField(max_length=200)
    url = models.CharField(max_length=200)
    pub_date = models.DateTimeField("last update", default=timezone.now)

    
    def __str__(self):
        return self.name    
    
    def was_published_recently(self):
        return self.pub_date >= timezone.now() - datetime.timedelta(days=1)


class Question(models.Model):
    
    question_text = models.CharField(max_length=200)
    pub_date = models.DateTimeField("date published")


class Choice(models.Model):
    question = models.ForeignKey(Question, on_delete=models.CASCADE)
    choice_text = models.CharField(max_length=200)
    votes = models.IntegerField(default=0)