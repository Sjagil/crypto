"""Thin executable composition root."""

from core.cli import build_parser, main, research_status_from_report

__all__ = ["build_parser", "main", "research_status_from_report"]


if __name__ == "__main__":
    raise SystemExit(main())
