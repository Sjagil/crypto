"""Optional outbound notifications; never an execution authority."""

from notifications.telegram import TelegramNotifier

__all__ = ["TelegramNotifier"]
