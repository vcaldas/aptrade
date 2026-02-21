from abc import ABC, abstractmethod


class NotificationService(ABC):
    @abstractmethod
    def send(self, message: str) -> None:
        pass
