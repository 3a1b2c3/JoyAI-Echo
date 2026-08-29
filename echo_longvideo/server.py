"""Command-line entry point for the Echo 1.5 local server."""

from server.app import app, main

__all__ = ["app", "main"]


if __name__ == "__main__":
    main()
