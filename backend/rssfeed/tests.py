from django.test import TestCase
import datetime
# Create your tests here.
from django.test import TestCase
from .models import Publication, Feed

class FeedModelTests(TestCase):
    def test_feed_creation(self):
        """
        Test that a Feed can be created with the required fields.
        """
        feed = Feed.objects.create(
            name='Test Feed',
            url='http://example.com/feed',
        )
        self.assertEqual(feed.name, 'Test Feed')
        self.assertEqual(feed.url, 'http://example.com/feed')
        self.assertEqual(type(feed.last_updated), datetime.datetime)
        self.assertEqual(feed.etag, '')
        
class PublicationModelTests(TestCase):
    def test_feed_creation(self):
        """
        Test that a Feed can be created with the required fields.
        """
        feed = Feed.objects.create(
            name='Test Feed',
            url='http://example.com/feed',
        )
        
        p = Publication.objects.create(
            title='Test Publication',
            link='http://example.com/test',
            description='Test description',
            summary='Test summary',
            rssfeed=feed
        )
        self.assertEqual(feed.name, 'Test Feed')
        self.assertEqual(feed.url, 'http://example.com/feed')
        self.assertEqual(type(feed.last_updated), datetime.datetime)
        self.assertEqual(feed.etag, '')