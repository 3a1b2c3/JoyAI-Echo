"""Safe HTTP download helper — no redirects, size cap, status 200 only."""

from __future__ import annotations

import urllib.error
import urllib.request
from dataclasses import dataclass

_DEFAULT_MAX_BYTES = 100 * 1024 * 1024
_DEFAULT_TIMEOUT_S = 60.0


@dataclass(frozen=True)
class HttpDownloadResult:
    data: bytes
    content_type: str
    content_length: int | None


class HttpDownloadError(RuntimeError):
    """Raised when an HTTP download fails or exceeds policy limits."""


class _NoRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise HttpDownloadError("下载失败：禁止重定向")


def download_http_bytes(
    url: str,
    *,
    max_bytes: int = _DEFAULT_MAX_BYTES,
    timeout_s: float = _DEFAULT_TIMEOUT_S,
) -> HttpDownloadResult:
    """Download *url* into memory with SSRF follow-up protections.

    - Does not follow redirects (302 → internal IP bypass).
    - Accepts HTTP 200 only.
    - Caps response body at *max_bytes* (+1 byte probe for oversize detection).
    """
    if max_bytes <= 0:
        raise HttpDownloadError("无效的大小限制")

    opener = urllib.request.build_opener(_NoRedirectHandler)
    request = urllib.request.Request(url, method="GET")
    request.add_header("Accept", "*/*")

    try:
        with opener.open(request, timeout=timeout_s) as response:
            status = getattr(response, "status", None) or response.getcode()
            if status != 200:
                raise HttpDownloadError(f"下载失败，HTTP状态码: {status}")

            content_type = response.headers.get("Content-Type") or "application/octet-stream"
            raw_length = response.headers.get("Content-Length")
            content_length: int | None = None
            if raw_length:
                try:
                    content_length = int(raw_length)
                except ValueError:
                    content_length = None

            limited = response.read(max_bytes + 1)
    except HttpDownloadError:
        raise
    except urllib.error.HTTPError as exc:
        raise HttpDownloadError(f"下载失败，HTTP状态码: {exc.code}") from exc
    except Exception as exc:
        raise HttpDownloadError("下载文件失败") from exc

    if len(limited) > max_bytes:
        raise HttpDownloadError(f"文件大小超过限制({max_bytes // (1024 * 1024)}MB)")

    return HttpDownloadResult(
        data=limited,
        content_type=content_type.split(";", 1)[0].strip() or "application/octet-stream",
        content_length=content_length,
    )
