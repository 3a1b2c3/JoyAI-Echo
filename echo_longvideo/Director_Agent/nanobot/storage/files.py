"""Config-driven local and S3-compatible file storage."""

from __future__ import annotations

import base64
import mimetypes
import re
import shutil
from hashlib import sha256
from pathlib import Path
from typing import Any
from urllib.parse import quote, unquote, urlparse


def _safe_segment(value: str, *, fallback: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip()).strip("._")
    return cleaned or fallback


def _local_root(config: Any, workspace: Path) -> Path:
    configured = Path(str(config.directory).strip()).expanduser()
    return (configured if configured.is_absolute() else workspace / configured).resolve()


class LocalFilePublisher:
    """Copy files into the configured local store and return gateway URLs."""

    def __init__(self, config: Any, *, workspace: Path, work_id: str) -> None:
        self.base_url = str(config.base_url).strip().rstrip("/")
        self.route_prefix = "/" + str(config.route_prefix).strip().strip("/")
        self.work_id = _safe_segment(work_id, fallback="work")
        self.root = _local_root(config, workspace.expanduser().resolve())
        self.delete_local_after_upload = False

    def _relative_path(self, name: str, *, digest: str) -> Path:
        parts = [
            _safe_segment(part, fallback="asset")
            for part in name.replace("\\", "/").split("/")
            if part.strip()
        ]
        relative = Path(*parts) if parts else Path("asset")
        return Path(self.work_id) / relative.with_name(
            f"{relative.stem}_{digest}{relative.suffix}"
        )

    def __call__(self, local: str, name: str) -> str:
        source = Path(local).expanduser().resolve()
        if not source.is_file() or source.stat().st_size <= 0:
            raise RuntimeError(f"file is missing or empty: {source}")
        digest = sha256(source.read_bytes()).hexdigest()[:12]
        relative = self._relative_path(name, digest=digest)
        target = self.root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if source != target.resolve(strict=False):
            shutil.copy2(source, target)
        path = quote(relative.as_posix(), safe="/-_.~")
        url = f"{self.route_prefix}/{path}"
        return f"{self.base_url}{url}" if self.base_url else url


class S3FilePublisher:
    """Upload files to explicitly configured S3-compatible storage."""

    def __init__(self, config: Any, *, work_id: str, client: Any | None = None) -> None:
        self.bucket = str(config.bucket).strip()
        self.region = str(config.region).strip()
        self.endpoint_url = str(config.endpoint_url).strip().rstrip("/")
        self.public_base_url = str(config.public_base_url).strip().rstrip("/")
        self.key_prefix = str(config.key_prefix).strip().strip("/")
        self.work_id = _safe_segment(work_id, fallback="work")
        self._config = config
        self._client = client
        access_key = str(config.access_key_id).strip()
        secret_key = str(config.secret_access_key).strip()
        if not all(
            (self.bucket, self.endpoint_url, self.public_base_url, access_key, secret_key)
        ):
            raise RuntimeError(
                "tools.fileStorage.outbound.s3 requires endpointUrl, publicBaseUrl, bucket, "
                "accessKeyId, and secretAccessKey"
            )

    def _client_for_upload(self) -> Any:
        if self._client is None:
            try:
                import boto3
                from botocore.config import Config as BotoConfig
            except ImportError as exc:
                raise RuntimeError(
                    "S3-compatible file upload requires boto3; run setup_local.sh again"
                ) from exc
            kwargs: dict[str, Any] = {
                "endpoint_url": self.endpoint_url,
                "region_name": self.region or None,
                "aws_access_key_id": str(self._config.access_key_id).strip(),
                "aws_secret_access_key": str(self._config.secret_access_key).strip(),
                "config": BotoConfig(
                    signature_version="s3v4",
                    s3={"addressing_style": str(self._config.addressing_style)},
                    request_checksum_calculation="when_required",
                    response_checksum_validation="when_required",
                ),
            }
            session_token = str(self._config.session_token).strip()
            if session_token:
                kwargs["aws_session_token"] = session_token
            self._client = boto3.client("s3", **kwargs)
        return self._client

    def _object_key(self, name: str, *, digest: str) -> str:
        parts = [
            _safe_segment(part, fallback="asset")
            for part in name.replace("\\", "/").split("/")
            if part.strip()
        ]
        relative = "/".join(parts) or "asset"
        path = Path(relative)
        relative = str(path.with_name(f"{path.stem}_{digest}{path.suffix}"))
        return "/".join(
            part for part in (self.key_prefix, self.work_id, relative) if part
        )

    def __call__(self, local: str, name: str) -> str:
        source = Path(local).expanduser().resolve()
        if not source.is_file() or source.stat().st_size <= 0:
            raise RuntimeError(f"file is missing or empty: {source}")
        digest = sha256(source.read_bytes()).hexdigest()[:12]
        object_key = self._object_key(name, digest=digest)
        content_type = mimetypes.guess_type(source.name)[0] or "application/octet-stream"
        self._client_for_upload().upload_file(
            str(source),
            self.bucket,
            object_key,
            ExtraArgs={"ContentType": content_type},
        )
        return f"{self.public_base_url}/{quote(object_key, safe='/-_.~')}"


def configured_file_publisher(
    work_id: str,
    *,
    storage: Any | None = None,
    workspace: Path | None = None,
) -> Any:
    """Build the canonical local file publisher."""
    if storage is None or workspace is None:
        from nanobot.config.loader import load_config

        config = load_config()
        storage = storage or config.tools.file_storage
        workspace = workspace or Path(config.agents.defaults.workspace)
    return LocalFilePublisher(
        storage.local,
        workspace=workspace,
        work_id=work_id,
    )


def resolve_local_asset_path(url: str, *, workspace: Path, config: Any) -> Path | None:
    """Resolve one configured local-store URL without allowing traversal."""
    parsed = urlparse(url)
    path = unquote(parsed.path or url)
    prefix = "/" + str(config.route_prefix).strip().strip("/") + "/"
    if not path.startswith(prefix):
        return None
    relative = path[len(prefix):]
    root = _local_root(config, workspace.expanduser().resolve())
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root)
    except ValueError:
        return None
    return candidate if candidate.is_file() else None


def local_asset_data_uri(url: str, *, workspace: Path, config: Any) -> str:
    """Convert a configured local-store URL to an inline payload."""
    path = resolve_local_asset_path(url, workspace=workspace, config=config)
    if path is None:
        return url
    mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def outbound_file_url(
    url: str,
    *,
    workspace: Path,
    work_id: str,
    name: str,
    storage: Any | None = None,
) -> str:
    """Prepare a locally stored file for an outbound service request."""
    if storage is None:
        from nanobot.config.loader import load_config

        storage = load_config().tools.file_storage
    path = resolve_local_asset_path(url, workspace=workspace, config=storage.local)
    if path is None:
        return url
    if storage.outbound.backend == "s3":
        return S3FilePublisher(storage.outbound.s3, work_id=work_id)(str(path), name)
    return local_asset_data_uri(url, workspace=workspace, config=storage.local)
