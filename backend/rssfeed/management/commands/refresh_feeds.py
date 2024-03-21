from datetime import timedelta, time, datetime

from django.core.mail import mail_admins
from django.core.management import BaseCommand
from django.utils import timezone
from django.utils.timezone import make_aware
import feedparser
from rssfeed.models import Publication, Feed

today = timezone.now()
tomorrow = today + timedelta(1)
today_start = make_aware(datetime.combine(today, time()))
today_end = make_aware(datetime.combine(tomorrow, time()))


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


class Command(BaseCommand):
    help = "Send Today's Orders Report to Admins"

    def handle(self, *args, **options):
        feeds_list = Feed.objects.order_by("-last_updated")

        if orders:
            message = ""

            for order in orders:
                message += f"{order} \n"

            subject = (
                f"Order Report for {today_start.strftime('%Y-%m-%d')} "
                f"to {today_end.strftime('%Y-%m-%d')}"
            )

            mail_admins(subject=subject, message=message, html_message=None)

            self.stdout.write("E-mail Report was sent.")
        else:
            self.stdout.write("No orders confirmed today.")