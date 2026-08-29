"""S3 URL parsing helpers for server-side object downloads."""

from __future__ import annotations


class S3UrlError(ValueError):
    pass


def is_s3_url(url: str) -> bool:
    return url.strip().lower().startswith("s3://")


def parse_s3_bucket_and_key(s3_url: str) -> tuple[str, str]:
    """Parse ``s3://bucket/key/path`` into ``(bucket, key)``."""
    raw = s3_url.strip()
    if not is_s3_url(raw):
        raise S3UrlError(f"不是有效的S3 URL: {s3_url}")
    without_scheme = raw[5:]
    parts = without_scheme.split("/", 1)
    if len(parts) < 2 or not parts[0].strip() or not parts[1].strip():
        raise S3UrlError(f"S3 URL格式错误: {s3_url}")
    return parts[0].strip(), parts[1].strip()


def validate_s3_url(s3_url: str, *, allowed_bucket: str) -> str:
    """Return object key when *s3_url* targets *allowed_bucket*."""
    bucket, key = parse_s3_bucket_and_key(s3_url)
    if bucket != allowed_bucket:
        raise S3UrlError(f"不允许访问该存储桶: {bucket}")
    if ".." in key.split("/"):
        raise S3UrlError("非法的S3路径")
    return key
