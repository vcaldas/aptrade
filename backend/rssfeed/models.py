from django.db import models
from django.utils import timezone
import datetime
# Create your models here.
class Publication(models.Model):
    id = models.CharField(max_length=1024, primary_key=True, unique=True)
    title = models.CharField(max_length=512)
    link = models.CharField(max_length=1024)
    description = models.CharField(max_length=200)
    summary = models.TextField(max_length=512)
    pub_date = models.DateTimeField("date published", default=timezone.now)
    rssfeed = models.ForeignKey('Feed', on_delete=models.CASCADE)

    def __str__(self):
        return self.title
    
    class Meta:
        db_table = 'Publication'

class Feed(models.Model):
    name = models.CharField(max_length=200)
    url = models.CharField(max_length=200)
    last_updated = models.DateTimeField("last update", default=timezone.now)
    modified = models.CharField("last modified", max_length=200, default="")
    etag = models.CharField(max_length=200, default="")
    
    
    def __str__(self):
        return self.name    
    
    def was_published_recently(self):
        return self.last_updated >= timezone.now() - datetime.timedelta(days=1)

    class Meta:
        db_table: 'Feed'