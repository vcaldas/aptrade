import json
import logstash


class LogstashFormatter(logstash.formatter.LogstashFormatterVersion0):
    """Helper class for formatting for Logstash"""

    @classmethod
    def serialize(cls, message):
        """Serialize the message

        Args:
            message: the message you want to serialize

        Returns:
            JSON dump of the message
        """
        return json.dumps(message)
s