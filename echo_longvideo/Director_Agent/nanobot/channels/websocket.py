"""WebSocket server channel: nanobot acts as a WebSocket server and serves connected clients."""

from __future__ import annotations

import asyncio
import base64
import binascii
import email.utils
import hashlib
import hmac
import http
import json
import mimetypes
import os
import re
import secrets
import shutil
import ssl
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Self
from urllib.parse import unquote, urlparse

from loguru import logger
from pydantic import Field, field_validator, model_validator
from websockets.asyncio.server import ServerConnection, serve
from websockets.datastructures import Headers
from websockets.exceptions import ConnectionClosed
from websockets.http11 import Request as WsRequest
from websockets.http11 import Response

from nanobot.agent.tools.director import (
    _WORKFLOW_INJECTED_EVENT,
    WORKFLOW_GATE_BYPASS,
    GenerateEchoShotTool,
    MergeShotTool,
    ReviewShotTool,
    SetShotReferencesTool,
    _caption_language_validation_error,
    _shot_key,
    _story_profile_validation_error,
    resolve_echo_duration_seconds,
    rewrite_prompt_for_i2v,
    sync_shot_echo_duration,
)
from nanobot.bus.events import InboundMessage, OutboundMessage
from nanobot.bus.queue import MessageBus
from nanobot.channels.base import BaseChannel
from nanobot.channels.webui_sessions import (
    is_legacy_two_part_webui_session_key,
    webui_session_key,
    webui_wire_chat_id,
)
from nanobot.config.paths import get_media_dir
from nanobot.config.schema import Base, ToolsConfig
from nanobot.director.memory_coordinator import (
    _resolve_media_binary,
    download_video,
    extract_video_frame,
    select_manual_memory_frame,
)
from nanobot.director.memory_review import (
    MemoryReviewConflict,
    approve_memory_review,
    reselect_memory_review,
)
from nanobot.integrations.echo_admission import (
    UNAVAILABLE_MESSAGE,
    EchoAdmissionController,
    EchoGeneratorBusyError,
    EchoGeneratorUnavailableError,
    is_connection_refused,
)
from nanobot.security.http_download import HttpDownloadError, download_http_bytes
from nanobot.security.url_validator import (
    DownloadUrlPolicy,
    UrlValidationError,
    configure_download_policy,
    validate_external_url,
)
from nanobot.session.auto_generate import (
    DEFAULT_AUTO_GENERATE_DURATION_SEC,
    apply_auto_generate,
    get_auto_generate,
    locked_shot_count_from_goal,
    locked_shot_count_from_state,
    resolve_auto_generate_from_wire,
    shot_count_for_auto_generate,
)
from nanobot.session.reference_image import (
    apply_story_direction_answer,
    is_blocked_local_url,
    is_reference_image_locked,
    mark_reference_image_needs_story_rewrite,
    normalize_reference_image,
    reference_image_present,
    reference_image_url,
    story_rewrite_suppressed,
)
from nanobot.session.source import apply_source, normalize_source
from nanobot.storage.files import (
    configured_file_publisher,
    resolve_local_asset_path,
)
from nanobot.utils.helpers import write_json_atomic
from nanobot.utils.media_decode import (
    FileSizeExceeded,
    save_base64_data_url,
)

if TYPE_CHECKING:
    from nanobot.providers.base import LLMProvider
    from nanobot.session.manager import SessionManager


def _strip_trailing_slash(path: str) -> str:
    if len(path) > 1 and path.endswith("/"):
        return path.rstrip("/")
    return path or "/"


def _normalize_config_path(path: str) -> str:
    return _strip_trailing_slash(path)


_CARD_ENTER_SHOT_POLISH = "进入逐镜打磨"
_CARD_CONFIRM_AUTO_GENERATE = "确认并一键成片"


class WebSocketConfig(Base):
    """WebSocket server channel configuration.

    Clients connect with URLs like ``ws://{host}:{port}{path}?client_id=...&token=...``.
    - ``client_id``: Used for ``allow_from`` authorization; if omitted, a value is generated and logged.
    - ``token``: If non-empty, the ``token`` query param may match this static secret; short-lived tokens
      from ``token_issue_path`` are also accepted.
    - ``token_issue_path``: If non-empty, **GET** (HTTP/1.1) to this path returns JSON
      ``{"token": "...", "expires_in": <seconds>}``; use ``?token=...`` when opening the WebSocket.
      Must differ from ``path`` (the WS upgrade path). If the client runs in the **same process** as
      nanobot and shares the asyncio loop, use a thread or async HTTP client for GET—do not call
      blocking ``urllib`` or synchronous ``httpx`` from inside a coroutine.
    - ``token_issue_secret``: If non-empty, token requests must send ``Authorization: Bearer <secret>`` or
      ``X-Nanobot-Auth: <secret>``.
    - ``websocket_requires_token``: If True, the handshake must include a valid token (static or issued and not expired).
    - Each connection has its own session: a unique ``chat_id`` maps to the agent session internally.
    - ``media`` field in outbound messages contains local filesystem paths; remote clients need a
      shared filesystem or an HTTP file server to access these files.
    """

    enabled: bool = False
    host: str = "127.0.0.1"
    port: int = 8765
    path: str = "/"
    token: str = ""
    token_issue_path: str = ""
    token_issue_secret: str = ""
    token_ttl_s: int = Field(default=300, ge=30, le=86_400)
    websocket_requires_token: bool = False
    allow_from: list[str] = Field(default_factory=lambda: ["*"])
    streaming: bool = True
    # Default 36 MB, upper 40 MB: supports up to 4 images at ~6 MB each after
    # client-side Worker normalization (see webui Composer). 4 × 6 MB × 1.37
    # (base64 overhead) + envelope framing stays under 36 MB; the 40 MB ceiling
    # leaves a small margin for sender slop without opening a DoS avenue.
    max_message_bytes: int = Field(default=37_748_736, ge=1024, le=41_943_040)
    ping_interval_s: float = Field(default=20.0, ge=5.0, le=300.0)
    ping_timeout_s: float = Field(default=20.0, ge=5.0, le=300.0)
    ssl_certfile: str = ""
    ssl_keyfile: str = ""
    # Workplace final-video proxy download (SSRF + size cap).
    download_max_bytes: int = Field(default=104_857_600, ge=1024, le=1_073_741_824)
    download_timeout_s: int = Field(default=60, ge=5, le=600)
    download_allowed_domain_suffixes: list[str] = Field(
        default_factory=list
    )
    download_trusted_internal_domains: list[str] = Field(default_factory=list)

    @field_validator("path")
    @classmethod
    def path_must_start_with_slash(cls, value: str) -> str:
        if not value.startswith("/"):
            raise ValueError('path must start with "/"')
        return _normalize_config_path(value)

    @field_validator("token_issue_path")
    @classmethod
    def token_issue_path_format(cls, value: str) -> str:
        value = value.strip()
        if not value:
            return ""
        if not value.startswith("/"):
            raise ValueError('token_issue_path must start with "/"')
        return _normalize_config_path(value)

    @model_validator(mode="after")
    def token_issue_path_differs_from_ws_path(self) -> Self:
        if not self.token_issue_path:
            return self
        if _normalize_config_path(self.token_issue_path) == _normalize_config_path(self.path):
            raise ValueError("token_issue_path must differ from path (the WebSocket upgrade path)")
        return self


def _http_json_response(data: dict[str, Any] | list, *, status: int = 200) -> Response:
    body = json.dumps(data, ensure_ascii=False).encode("utf-8")
    headers = Headers(
        [
            ("Date", email.utils.formatdate(usegmt=True)),
            ("Connection", "close"),
            ("Content-Length", str(len(body))),
            ("Content-Type", "application/json; charset=utf-8"),
        ]
    )
    reason = http.HTTPStatus(status).phrase
    return Response(status, reason, headers, body)


def _read_webui_model_name() -> str | None:
    """Return the configured default model for readonly webui display."""
    try:
        from nanobot.config.loader import load_config

        model = load_config().agents.defaults.model.strip()
        return model or None
    except Exception as e:
        logger.debug("webui bootstrap could not load model name: {}", e)
        return None


def _parse_request_path(path_with_query: str) -> tuple[str, dict[str, list[str]]]:
    """Parse normalized path and query parameters in one pass."""
    parsed = urlparse("ws://x" + path_with_query)
    path = _strip_trailing_slash(parsed.path or "/")
    query: dict[str, list[str]] = {}
    if parsed.query:
        for part in parsed.query.split("&"):
            if not part:
                continue
            if "=" in part:
                key, value = part.split("=", 1)
            else:
                key, value = part, ""
            key = unquote(key, errors="replace")
            value = unquote(value, errors="replace")
            query.setdefault(key, []).append(value)
    return path, query


def _normalize_http_path(path_with_query: str) -> str:
    """Return the path component (no query string), with trailing slash normalized (root stays ``/``)."""
    return _parse_request_path(path_with_query)[0]


def _parse_query(path_with_query: str) -> dict[str, list[str]]:
    return _parse_request_path(path_with_query)[1]


def _query_first(query: dict[str, list[str]], key: str) -> str | None:
    """Return the first value for *key*, or None."""
    values = query.get(key)
    return values[0] if values else None


def _parse_inbound_payload(raw: str) -> str | None:
    """Parse a client frame into text; return None for empty or unrecognized content."""
    text = raw.strip()
    if not text:
        return None
    if text.startswith("{"):
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            return text
        if isinstance(data, dict):
            for key in ("content", "text", "message"):
                value = data.get(key)
                if isinstance(value, str) and value.strip():
                    return value
            return None
        return None
    return text


# Accept UUIDs and short scoped keys like "unified:default". Keeps the capability
# namespace small enough to rule out path traversal / quote injection tricks.
_CHAT_ID_RE = re.compile(r"^[A-Za-z0-9_:-]{1,64}$")


def _is_valid_chat_id(value: Any) -> bool:
    return isinstance(value, str) and _CHAT_ID_RE.match(value) is not None


def _parse_envelope(raw: str) -> dict[str, Any] | None:
    """Return a typed envelope dict if the frame is a new-style JSON envelope, else None.

    A frame qualifies when it parses as a JSON object with a string ``type`` field.
    Legacy frames (plain text, or ``{"content": ...}`` without ``type``) return None;
    callers should fall back to :func:`_parse_inbound_payload` for those.
    """
    text = raw.strip()
    if not text.startswith("{"):
        return None
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None
    t = data.get("type")
    if not isinstance(t, str):
        return None
    return data


# Per-message image limits. The server-side guard is a touch looser than the
# client's ``Worker`` normalization target (6 MB) — tolerate client slop, but
# still cap total ingress at ``_MAX_IMAGES_PER_MESSAGE * _MAX_IMAGE_BYTES``
# which fits comfortably inside ``max_message_bytes``.
_MAX_IMAGES_PER_MESSAGE = 4
_MAX_IMAGE_BYTES = 8 * 1024 * 1024
_MAX_MEMORY_AUDIO_BYTES = 20 * 1024 * 1024
_MAX_MEMORY_WORKSPACE_ASSETS = 100
_MAX_MEMORY_SLOTS = 7
_MEMORY_REFERENCE_TYPES = frozenset({"character", "scene", "style", "object", "other"})

# Image MIME whitelist — matches the Composer's ``accept`` list. SVG is
# explicitly excluded to avoid the XSS surface inside embedded scripts.
_IMAGE_MIME_ALLOWED: frozenset[str] = frozenset({
    "image/png",
    "image/jpeg",
    "image/webp",
    "image/gif",
})

_AUDIO_MIME_ALLOWED: frozenset[str] = frozenset({
    "audio/aac",
    "audio/flac",
    "audio/mp4",
    "audio/mpeg",
    "audio/m4a",
    "audio/ogg",
    "audio/vnd.wave",
    "audio/wave",
    "audio/wav",
    "audio/webm",
    "audio/x-m4a",
    "audio/x-wav",
})

_DATA_URL_MIME_RE = re.compile(r"^data:([^;]+);base64,", re.DOTALL)


def _extract_data_url_mime(url: str) -> str | None:
    """Return the MIME type of a ``data:<mime>;base64,...`` URL, else ``None``."""
    if not isinstance(url, str):
        return None
    m = _DATA_URL_MIME_RE.match(url)
    if not m:
        return None
    return m.group(1).strip().lower() or None


_LOCALHOSTS = frozenset({"127.0.0.1", "::1", "localhost"})

# Matches the legacy chat-id pattern but allows file-system-safe stems too,
# so the API can address sessions whose keys came from non-WebSocket channels.
_API_KEY_RE = re.compile(r"^[A-Za-z0-9_:.-]{1,128}$")


def _decode_api_key(raw_key: str) -> str | None:
    """Decode a percent-encoded API path segment, then validate the result."""
    key = unquote(raw_key)
    if _API_KEY_RE.match(key) is None:
        return None
    return key


def _is_localhost(connection: Any) -> bool:
    """Return True if *connection* originated from the loopback interface."""
    addr = getattr(connection, "remote_address", None)
    if not addr:
        return False
    host = addr[0] if isinstance(addr, tuple) else addr
    if not isinstance(host, str):
        return False
    # ``::ffff:127.0.0.1`` is loopback in IPv6-mapped form.
    if host.startswith("::ffff:"):
        host = host[7:]
    return host in _LOCALHOSTS


def _http_response(
    body: bytes,
    *,
    status: int = 200,
    content_type: str = "text/plain; charset=utf-8",
    extra_headers: list[tuple[str, str]] | None = None,
) -> Response:
    headers = [
        ("Date", email.utils.formatdate(usegmt=True)),
        ("Connection", "close"),
        ("Content-Length", str(len(body))),
        ("Content-Type", content_type),
    ]
    if extra_headers:
        headers.extend(extra_headers)
    reason = http.HTTPStatus(status).phrase
    return Response(status, reason, Headers(headers), body)


def _http_error(status: int, message: str | None = None) -> Response:
    body = (message or http.HTTPStatus(status).phrase).encode("utf-8")
    return _http_response(body, status=status)


def _bearer_token(headers: Any) -> str | None:
    """Pull a Bearer token out of standard or query-style headers."""
    auth = headers.get("Authorization") or headers.get("authorization")
    if auth and auth.lower().startswith("bearer "):
        return auth[7:].strip() or None
    return None


def _is_websocket_upgrade(request: WsRequest) -> bool:
    """Detect an actual WS upgrade; plain HTTP GETs to the same path should fall through."""
    upgrade = request.headers.get("Upgrade") or request.headers.get("upgrade")
    connection = request.headers.get("Connection") or request.headers.get("connection")
    if not upgrade or "websocket" not in upgrade.lower():
        return False
    if not connection or "upgrade" not in connection.lower():
        return False
    return True


def _b64url_encode(data: bytes) -> str:
    """URL-safe base64 without padding — compact + friendly in URL paths."""
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _b64url_decode(s: str) -> bytes:
    """Reverse of :func:`_b64url_encode`; caller handles ``ValueError``."""
    pad = "=" * (-len(s) % 4)
    return base64.urlsafe_b64decode(s + pad)


# Allowed MIME types we actually serve from the media endpoint. Anything
# outside this set is degraded to ``application/octet-stream`` so an
# attacker who somehow gets a signed URL for an unexpected file type can't
# trick the browser into sniffing executable content.
_MEDIA_ALLOWED_MIMES: frozenset[str] = frozenset({
    "image/png",
    "image/jpeg",
    "image/webp",
    "image/gif",
    "audio/wav",
    "audio/x-wav",
    "audio/mpeg",
    "audio/mp4",
    "audio/ogg",
    "video/mp4",
    "video/webm",
    "video/ogg",
    "video/quicktime",
})

_REMOTE_MEDIA_SCHEMES: frozenset[str] = frozenset({"http", "https"})


def _guess_allowed_media_mime(name: str) -> str | None:
    """Return the guessed MIME only when it is safe to serve inline."""
    mime, _ = mimetypes.guess_type(name)
    if mime in _MEDIA_ALLOWED_MIMES:
        return mime
    return None


def _media_display_name(ref: str) -> str | None:
    """Best-effort filename label for a local path or remote URL."""
    raw = ref.strip()
    if not raw:
        return None
    parsed = urlparse(raw)
    if parsed.scheme in (_REMOTE_MEDIA_SCHEMES | {"file"}):
        path = unquote(parsed.path or "")
        return Path(path).name or None
    return Path(raw).name or None


def _issue_route_secret_matches(headers: Any, configured_secret: str) -> bool:
    """Return True if the token-issue HTTP request carries credentials matching ``token_issue_secret``."""
    if not configured_secret:
        return True
    authorization = headers.get("Authorization") or headers.get("authorization")
    if authorization and authorization.lower().startswith("bearer "):
        supplied = authorization[7:].strip()
        return hmac.compare_digest(supplied, configured_secret)
    header_token = headers.get("X-Nanobot-Auth") or headers.get("x-nanobot-auth")
    if not header_token:
        return False
    return hmac.compare_digest(header_token.strip(), configured_secret)


class WebSocketChannel(BaseChannel):
    """Run a local WebSocket server; forward text/JSON messages to the message bus."""

    name = "websocket"
    display_name = "WebSocket"

    def __init__(
        self,
        config: Any,
        bus: MessageBus,
        *,
        session_manager: "SessionManager | None" = None,
        provider: "LLMProvider | None" = None,
        model: str | None = None,
        static_dist_path: Path | None = None,
        tools_config: ToolsConfig | None = None,
        gateway_debug: bool = False,
        memory_review_runner: Callable[..., Any] | None = None,
    ):
        if isinstance(config, dict):
            config = WebSocketConfig.model_validate(config)
        super().__init__(config, bus)
        self.config: WebSocketConfig = config
        # chat_id -> connections subscribed to it (fan-out target).
        self._subs: dict[str, set[Any]] = {}
        # connection -> chat_ids it is subscribed to (O(1) cleanup on disconnect).
        self._conn_chats: dict[Any, set[str]] = {}
        # connection -> default chat_id for legacy frames that omit routing.
        self._conn_default: dict[Any, str] = {}
        # Single-use tokens consumed at WebSocket handshake.
        self._issued_tokens: dict[str, float] = {}
        # Multi-use tokens for the embedded webui's REST surface; checked but not consumed.
        self._api_tokens: dict[str, float] = {}
        self._stop_event: asyncio.Event | None = None
        self._server_task: asyncio.Task[None] | None = None
        self._session_manager = session_manager
        self._provider = provider
        self._model = model or (provider.get_default_model() if provider is not None else None)
        self._tools_config = tools_config or ToolsConfig()
        self._gateway_debug = gateway_debug
        if memory_review_runner is None:
            from nanobot.director.memory_coordinator import (
                run_memory_review_from_config,
            )

            memory_review_runner = run_memory_review_from_config
        self._memory_review_runner = memory_review_runner
        self._static_dist_path: Path | None = (
            static_dist_path.resolve() if static_dist_path is not None else None
        )
        # Process-local secret used to HMAC-sign media URLs. The signed URL is
        # the capability — anyone who holds a valid URL can fetch that one
        # file, nothing else. The secret regenerates on restart so links
        # become self-expiring (callers just refresh the session list).
        self._media_secret: bytes = secrets.token_bytes(32)
        self._auto_generate_inflight: set[str] = set()
        self._loop: asyncio.AbstractEventLoop | None = None

    # -- Subscription bookkeeping -------------------------------------------

    def _attach(self, connection: Any, chat_id: str) -> None:
        """Idempotently subscribe *connection* to *chat_id*."""
        self._subs.setdefault(chat_id, set()).add(connection)
        self._conn_chats.setdefault(connection, set()).add(chat_id)

    def _cleanup_connection(self, connection: Any) -> None:
        """Remove *connection* from every subscription set; safe to call multiple times."""
        chat_ids = self._conn_chats.pop(connection, set())
        for cid in chat_ids:
            subs = self._subs.get(cid)
            if subs is None:
                continue
            subs.discard(connection)
            if not subs:
                self._subs.pop(cid, None)
        self._conn_default.pop(connection, None)

    def _webui_session_key_for_connection(self, connection: Any, chat_id: str) -> str:
        return webui_session_key("local", chat_id)

    def _session_visible_to_caller(self, session_key: str, request: WsRequest) -> bool:
        return self._is_webui_session_key(session_key)

    async def _send_event(self, connection: Any, event: str, **fields: Any) -> None:
        """Send a control event (attached, error, ...) to a single connection."""
        payload: dict[str, Any] = {"event": event}
        payload.update(fields)
        raw = json.dumps(payload, ensure_ascii=False)
        try:
            await connection.send(raw)
        except ConnectionClosed:
            self._cleanup_connection(connection)
        except Exception as e:
            logger.warning("websocket: failed to send {} event: {}", event, e)

    @classmethod
    def default_config(cls) -> dict[str, Any]:
        return WebSocketConfig().model_dump(by_alias=True)

    def _expected_path(self) -> str:
        return _normalize_config_path(self.config.path)

    def _build_ssl_context(self) -> ssl.SSLContext | None:
        cert = self.config.ssl_certfile.strip()
        key = self.config.ssl_keyfile.strip()
        if not cert and not key:
            return None
        if not cert or not key:
            raise ValueError(
                "websocket: ssl_certfile and ssl_keyfile must both be set for WSS, or both left empty"
            )
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ctx.minimum_version = ssl.TLSVersion.TLSv1_2
        ctx.load_cert_chain(certfile=cert, keyfile=key)
        return ctx

    _MAX_ISSUED_TOKENS = 10_000

    def _purge_expired_issued_tokens(self) -> None:
        now = time.monotonic()
        for token_key, expiry in list(self._issued_tokens.items()):
            if now > expiry:
                self._issued_tokens.pop(token_key, None)

    def _take_issued_token_if_valid(self, token_value: str | None) -> bool:
        """Validate and consume one issued token (single use per connection attempt).

        Uses single-step pop to minimize the window between lookup and removal;
        safe under asyncio's single-threaded cooperative model.
        """
        if not token_value:
            return False
        self._purge_expired_issued_tokens()
        expiry = self._issued_tokens.pop(token_value, None)
        if expiry is None:
            return False
        if time.monotonic() > expiry:
            return False
        return True

    def _handle_token_issue_http(self, connection: Any, request: Any) -> Any:
        secret = self.config.token_issue_secret.strip()
        if secret:
            if not _issue_route_secret_matches(request.headers, secret):
                return connection.respond(401, "Unauthorized")
        else:
            logger.warning(
                "websocket: token_issue_path is set but token_issue_secret is empty; "
                "any client can obtain connection tokens — set token_issue_secret for production."
            )
        self._purge_expired_issued_tokens()
        if len(self._issued_tokens) >= self._MAX_ISSUED_TOKENS:
            logger.error(
                "websocket: too many outstanding issued tokens ({}), rejecting issuance",
                len(self._issued_tokens),
            )
            return _http_json_response({"error": "too many outstanding tokens"}, status=429)
        token_value = f"nbwt_{secrets.token_urlsafe(32)}"
        self._issued_tokens[token_value] = time.monotonic() + float(self.config.token_ttl_s)

        return _http_json_response(
            {"token": token_value, "expires_in": self.config.token_ttl_s}
        )

    # -- HTTP dispatch ------------------------------------------------------

    async def _dispatch_http(self, connection: Any, request: WsRequest) -> Any:
        """Route an inbound HTTP request to a handler or to the WS upgrade path."""
        got, query = _parse_request_path(request.path)

        # 1. Token issue endpoint (legacy, optional, gated by configured secret).
        if self.config.token_issue_path:
            issue_expected = _normalize_config_path(self.config.token_issue_path)
            if got == issue_expected:
                return self._handle_token_issue_http(connection, request)

        # 2. WebUI bootstrap: mints short-lived local transport tokens.
        if got == "/webui/bootstrap":
            return self._handle_webui_bootstrap(connection, request)

        # 3. REST surface for the embedded UI.
        if got == "/api/sessions":
            return self._handle_sessions_list(request)

        m = re.match(r"^/api/sessions/([^/]+)/messages$", got)
        if m:
            return self._handle_session_messages(request, m.group(1))

        # NOTE: websockets' HTTP parser only accepts GET, so we cannot expose a
        # true ``DELETE`` verb. The action is folded into the path instead.
        m = re.match(r"^/api/sessions/([^/]+)/delete$", got)
        if m:
            return self._handle_session_delete(request, m.group(1))

        m = re.match(r"^/api/sessions/([^/]+)/generation-settings/save$", got)
        if m:
            return self._handle_generation_settings_save(request, m.group(1))

        m = re.match(r"^/api/sessions/([^/]+)/generation-settings$", got)
        if m:
            return self._handle_generation_settings_get(request, m.group(1))

        m = re.match(r"^/api/workplace/([^/]+)$", got)
        if m:
            return self._handle_workplace_status(request, m.group(1))

        m = re.match(r"^/api/workplace/([^/]+)/story/save$", got)
        if m:
            return self._handle_workplace_story_save(request, m.group(1))

        m = re.match(r"^/api/workplace/([^/]+)/story-profile/save$", got)
        if m:
            return self._handle_workplace_story_profile_save(request, m.group(1))

        m = re.match(r"^/api/workplace/([^/]+)/reference-image/save$", got)
        if m:
            return self._handle_workplace_reference_image_save(request, m.group(1))

        m = re.match(r"^/api/workplace/([^/]+)/reference-image/delete$", got)
        if m:
            return self._handle_workplace_reference_image_delete(request, m.group(1))

        # Alias: GET /reference-image with X-Nanobot-Body is save (websockets is GET-only).
        m = re.match(r"^/api/workplace/([^/]+)/reference-image$", got)
        if m:
            return self._handle_workplace_reference_image_save(request, m.group(1))

        m = re.match(r"^/api/workplace/([^/]+)/shots/accept-all$", got)
        if m:
            return self._handle_workplace_shot_accept_all(request, m.group(1))

        m = re.match(
            r"^/api/workplace/([^/]+)/shots/(\d+)/memory-review/(approve|reselect|manual-select|select-mode)$", got
        )
        if m:
            return self._handle_memory_review_action(
                request, m.group(1), int(m.group(2)), m.group(3)
            )
        m = re.match(r"^/api/workplace/([^/]+)/shots/(\d+)/accept$", got)
        if m:
            return self._handle_workplace_shot_accept(request, m.group(1), int(m.group(2)))

        m = re.match(r"^/api/workplace/([^/]+)/shots/(\d+)/revise$", got)
        if m:
            return self._handle_workplace_shot_revise(request, m.group(1), int(m.group(2)))

        m = re.match(r"^/api/workplace/([^/]+)/shots/(\d+)/merge-up$", got)
        if m:
            return self._handle_workplace_shot_merge_up(request, m.group(1), int(m.group(2)))

        m = re.match(r"^/api/workplace/([^/]+)/shots/(\d+)/remove-shot$", got)
        if m:
            return self._handle_workplace_shot_remove_shot(request, m.group(1), int(m.group(2)))

        m = re.match(r"^/api/workplace/([^/]+)/shots/(\d+)/split-shot$", got)
        if m:
            return self._handle_workplace_shot_split_shot(request, m.group(1), int(m.group(2)))

        m = re.match(r"^/api/workplace/([^/]+)/workflow/confirm-story$", got)
        if m:
            return self._handle_workplace_workflow_confirm_story(request, m.group(1))

        m = re.match(r"^/api/workplace/([^/]+)/workflow/start-generation$", got)
        if m:
            return self._handle_workplace_workflow_start_generation(request, m.group(1))

        m = re.match(r"^/api/workplace/([^/]+)/workflow/abort-generation$", got)
        if m:
            return self._handle_workplace_workflow_abort_generation(request, m.group(1))

        m = re.match(r"^/api/workplace/([^/]+)/workflow/generate-all$", got)
        if m:
            return await self._handle_workplace_workflow_generate_all(request, m.group(1))

        m = re.match(r"^/api/workplace/([^/]+)/workflow/auto-generate$", got)
        if m:
            return self._handle_workplace_workflow_auto_generate(request, m.group(1))

        m = re.match(r"^/api/workplace/([^/]+)/shots/(\d+)/generate$", got)
        if m:
            return await self._handle_workplace_shot_generate(
                request, m.group(1), int(m.group(2))
            )

        m = re.match(r"^/api/workplace/([^/]+)/shots/(\d+)/continuous-generate$", got)
        if m:
            return self._handle_workplace_shot_continuous_generate(request, m.group(1), int(m.group(2)))

        m = re.match(r"^/api/workplace/([^/]+)/shots/(\d+)/continuous-mode$", got)
        if m:
            return self._handle_workplace_shot_continuous_mode(request, m.group(1), int(m.group(2)))

        m = re.match(r"^/api/workplace/([^/]+)/shots/(\d+)/duration$", got)
        if m:
            return self._handle_workplace_shot_duration(request, m.group(1), int(m.group(2)))

        m = re.match(r"^/api/workplace/([^/]+)/workflow/start-merge$", got)
        if m:
            return self._handle_workplace_workflow_start_merge(request, m.group(1))

        m = re.match(r"^/api/workplace/([^/]+)/workflow/regenerate$", got)
        if m:
            return self._handle_workplace_workflow_regenerate(request, m.group(1))

        m = re.match(r"^/api/workplace/([^/]+)/echo/like$", got)
        if m:
            return self._handle_echo_like(request, m.group(1))

        m = re.match(r"^/api/workplace/([^/]+)/echo/download-prompt$", got)
        if m:
            return self._handle_echo_download_prompt(request, m.group(1))

        m = re.match(r"^/api/workplace/([^/]+)/download/final$", got)
        if m:
            return self._handle_workplace_download_final(request, m.group(1))

        # Signed media fetch: ``<sig>`` is an HMAC over ``<payload>``; the
        # payload decodes to a path inside :func:`get_media_dir`. See
        # :meth:`_sign_media_path` for the inverse direction used to build
        # these URLs when replaying a session.
        m = re.match(r"^/api/media/([A-Za-z0-9_-]+)/([A-Za-z0-9_-]+)$", got)
        if m:
            return self._handle_media_fetch(m.group(1), m.group(2))

        asset_prefix = (
            "/"
            + self._tools_config.file_storage.local.route_prefix.strip().strip("/")
        )
        if got.startswith(f"{asset_prefix}/"):
            return self._handle_local_asset_fetch(got)

        # PromptStack API
        if got == "/api/promptstack/sessions":
            if not self._gateway_debug:
                return _http_error(404, "Not found")
            return self._handle_promptstack_sessions(request)

        m = re.match(r"^/api/promptstack/traces/([A-Za-z0-9_.-]+)$", got)
        if m:
            if not self._gateway_debug:
                return _http_error(404, "Not found")
            return self._handle_promptstack_trace(request, m.group(1))

        # EventStack API
        if got == "/api/eventstack/sessions":
            if not self._gateway_debug:
                return _http_error(404, "Not found")
            return self._handle_eventstack_sessions(request)

        m = re.match(r"^/api/eventstack/traces/([A-Za-z0-9_.-]+)$", got)
        if m:
            if not self._gateway_debug:
                return _http_error(404, "Not found")
            return self._handle_eventstack_trace(request, m.group(1))

        # PE (Prompt Engineering) set list + active
        if got == "/api/pe-sets":
            return self._handle_pe_list(request)

        # 4. WebSocket upgrade (the channel's primary purpose). Only run the
        # handshake gate on requests that actually ask to upgrade; otherwise
        # a bare ``GET /`` from the browser would be rejected as an
        # unauthorized WS handshake instead of serving the SPA's index.html.
        expected_ws = self._expected_path()
        if got == expected_ws and _is_websocket_upgrade(request):
            client_id = _query_first(query, "client_id") or ""
            if len(client_id) > 128:
                client_id = client_id[:128]
            if not self.is_allowed(client_id):
                return connection.respond(403, "Forbidden")
            return self._authorize_websocket_handshake(connection, query)

        # 5. Static SPA serving (only if a build directory was wired in).
        if self._static_dist_path is not None:
            response = self._serve_static(got)
            if response is not None:
                return response

        return connection.respond(404, "Not Found")

    # -- HTTP route handlers ------------------------------------------------

    def _check_api_token(self, request: WsRequest) -> bool:
        """Validate the short-lived local WebUI transport token."""
        self._purge_expired_api_tokens()
        token = _bearer_token(request.headers) or _query_first(
            _parse_query(request.path), "token"
        )
        if not token:
            return False
        expiry = self._api_tokens.get(token)
        if expiry is None or time.monotonic() > expiry:
            self._api_tokens.pop(token, None)
            return False
        return True

    def _purge_expired_api_tokens(self) -> None:
        now = time.monotonic()
        for token_key, expiry in list(self._api_tokens.items()):
            if now > expiry:
                self._api_tokens.pop(token_key, None)

    def _handle_webui_bootstrap(self, connection: Any, request: WsRequest) -> Response:
        from nanobot.prompts import PEManager

        active_pe = PEManager.instance().active
        if not _is_localhost(connection):
            return _http_error(403, "webui bootstrap is localhost-only")

        self._purge_expired_issued_tokens()
        self._purge_expired_api_tokens()
        if (
            len(self._issued_tokens) >= self._MAX_ISSUED_TOKENS
            or len(self._api_tokens) >= self._MAX_ISSUED_TOKENS
        ):
            return _http_json_response({"error": "too many outstanding tokens"}, status=429)

        token_value = f"nbwt_{secrets.token_urlsafe(32)}"
        expiry = time.monotonic() + float(self.config.token_ttl_s)
        self._issued_tokens[token_value] = expiry
        self._api_tokens[token_value] = expiry
        return _http_json_response(
            {
                "ws_path": self._expected_path(),
                "token": token_value,
                "expires_in": int(self.config.token_ttl_s),
                "model_name": _read_webui_model_name(),
                "user_id": "local",
                "active_pe": active_pe,
            }
        )

    def _handle_sessions_list(self, request: WsRequest) -> Response:
        if not self._check_api_token(request):
            return _http_error(401, "Unauthorized")
        if self._session_manager is None:
            return _http_error(503, "session manager unavailable")
        sessions = self._session_manager.list_sessions()
        # The webui is only meaningful for websocket-channel chats — CLI /
        # Slack / Lark / Discord sessions can't be resumed from the browser,
        # so leaking them into the sidebar is just noise. Filter to the
        # ``websocket:`` prefix and strip absolute paths on the way out.
        cleaned = [
            {
                **{k: v for k, v in s.items() if k != "path"},
                **self._session_echo_tracking_fields(s.get("key")),
            }
            for s in sessions
            if isinstance(s.get("key"), str)
            and self._is_webui_session_key(s["key"])
            and self._session_visible_to_caller(s["key"], request)
        ]
        return _http_json_response({"sessions": cleaned})

    @staticmethod
    def _is_webui_session_key(key: str) -> bool:
        """Return True when *key* belongs to the webui's websocket-only surface."""
        return key.startswith("websocket:")

    @staticmethod
    def _legacy_webui_session_key_error(raw_key: str) -> Response | None:
        decoded = _decode_api_key(raw_key)
        if decoded is not None and is_legacy_two_part_webui_session_key(decoded):
            return _http_error(400, "invalid session key: two-part format is no longer supported")
        return None

    def _resolve_webui_api_session_key(self, raw_key: str, request: WsRequest) -> str | None:
        """Decode and authorize a three-part WebUI session key."""
        decoded = _decode_api_key(raw_key)
        if decoded is None or not self._is_webui_session_key(decoded):
            return None
        if is_legacy_two_part_webui_session_key(decoded):
            return None
        if self._session_visible_to_caller(decoded, request):
            return decoded
        return None

    def _handle_session_messages(self, request: WsRequest, key: str) -> Response:
        if not self._check_api_token(request):
            return _http_error(401, "Unauthorized")
        if self._session_manager is None:
            return _http_error(503, "session manager unavailable")
        legacy_err = self._legacy_webui_session_key_error(key)
        if legacy_err is not None:
            return legacy_err
        decoded_key = self._resolve_webui_api_session_key(key, request)
        if decoded_key is None:
            if _decode_api_key(key) is None:
                return _http_error(400, "invalid session key")
            return _http_error(404, "session not found")
        data = self._session_manager.read_session_file(decoded_key)
        if data is None:
            # Brand-new chats exist only in the browser until the first turn is
            # persisted — return an empty history instead of a misleading 404.
            return _http_json_response(
                {
                    "key": decoded_key,
                    "created_at": None,
                    "updated_at": None,
                    "messages": [],
                }
            )
        # Decorate persisted media refs with browser-usable URLs so the
        # client can render previews. Raw local paths are stripped on the
        # way out — they leak server filesystem layout and the client never
        # needs them once it has a signed fetch URL.
        self._augment_media_urls(data)
        return _http_json_response(data)

    def _augment_media_urls(self, payload: dict[str, Any]) -> None:
        """Mutate *payload* in place: each message's ``media`` ref list is
        replaced by a parallel ``media_urls`` list of browser-usable URLs.

        Messages without media or with non-string path entries are left
        untouched. Missing / unsupported local files are silently skipped;
        the client falls back to text-only history in that case.
        """
        messages = payload.get("messages")
        if not isinstance(messages, list):
            return
        for msg in messages:
            if not isinstance(msg, dict):
                continue
            media = msg.get("media")
            if not isinstance(media, list) or not media:
                continue
            urls: list[dict[str, str]] = []
            for entry in media:
                if not isinstance(entry, str) or not entry:
                    continue
                public_ref = self._public_media_entry(entry)
                if public_ref is None:
                    continue
                urls.append(public_ref)
            if urls:
                msg["media_urls"] = urls
            # Always drop the raw paths from the wire payload.
            msg.pop("media", None)

    def _sign_media_path(self, abs_path: Path) -> str | None:
        """Return a ``/api/media/<sig>/<payload>`` URL for *abs_path*, or
        ``None`` when the path does not resolve inside the media root.

        The URL is self-authenticating: the signature binds the payload to
        this process's ``_media_secret``, so only paths we chose to sign can
        be fetched. The returned path is relative to the server origin; the
        client joins it against the existing webui base.
        """
        try:
            media_root = get_media_dir().resolve()
            rel = abs_path.resolve().relative_to(media_root)
        except (OSError, ValueError):
            return None
        payload = _b64url_encode(rel.as_posix().encode("utf-8"))
        mac = hmac.new(
            self._media_secret, payload.encode("ascii"), hashlib.sha256
        ).digest()[:16]
        return f"/api/media/{_b64url_encode(mac)}/{payload}"

    def _sign_local_media_path(self, abs_path: Path) -> str | None:
        """Return a signed fetch URL for an arbitrary local media file.

        Unlike :meth:`_sign_media_path`, this path may live outside the
        websocket media dir (for example director-generated video shots in
        ``/tmp``). Only files with an allow-listed image/video MIME are
        signed so the route does not become a generic file browser.
        """
        try:
            candidate = abs_path.expanduser().resolve()
        except OSError:
            return None
        if not candidate.is_file():
            return None
        if _guess_allowed_media_mime(candidate.name) is None:
            return None
        payload = _b64url_encode(candidate.as_posix().encode("utf-8"))
        mac = hmac.new(
            self._media_secret, payload.encode("ascii"), hashlib.sha256
        ).digest()[:16]
        return f"/api/media/{_b64url_encode(mac)}/{payload}"

    def _public_media_entry(self, entry: str) -> dict[str, str] | None:
        """Return a browser-usable media ref for a local path or remote URL."""
        raw = entry.strip()
        if not raw:
            return None
        parsed = urlparse(raw)
        name = _media_display_name(raw)
        if parsed.scheme in _REMOTE_MEDIA_SCHEMES:
            result = {"url": raw}
            if name:
                result["name"] = name
            return result
        if raw.startswith("/api/media/"):
            result = {"url": raw}
            if name:
                result["name"] = name
            return result
        local_asset = None
        if self._session_manager is not None:
            local_asset = resolve_local_asset_path(
                raw,
                workspace=self._session_manager.workspace,
                config=self._tools_config.file_storage.local,
            )
        if parsed.scheme == "file":
            local = Path(unquote(parsed.path or ""))
        elif local_asset is not None:
            local = local_asset
        else:
            local = Path(raw).expanduser()
        signed = self._sign_media_path(local)
        if signed is None:
            signed = self._sign_local_media_path(local)
        if signed is None:
            return None
        result = {"url": signed}
        if name:
            result["name"] = name
        return result

    def _handle_media_fetch(self, sig: str, payload: str) -> Response:
        """Serve a single media file previously signed via
        :meth:`_sign_media_path`. Validates the signature, decodes the
        payload to a relative path, and streams the file bytes with a
        long-lived immutable cache header (the URL already encodes the
        file identity, so caches can be aggressive)."""
        try:
            provided_mac = _b64url_decode(sig)
        except (ValueError, binascii.Error):
            return _http_error(401, "invalid signature")
        expected_mac = hmac.new(
            self._media_secret, payload.encode("ascii"), hashlib.sha256
        ).digest()[:16]
        if not hmac.compare_digest(expected_mac, provided_mac):
            return _http_error(401, "invalid signature")
        try:
            rel_bytes = _b64url_decode(payload)
            rel_str = rel_bytes.decode("utf-8")
        except (ValueError, binascii.Error, UnicodeDecodeError):
            return _http_error(400, "invalid payload")
        try:
            requested = Path(rel_str).expanduser()
            if requested.is_absolute():
                candidate = requested.resolve()
                if _guess_allowed_media_mime(candidate.name) is None:
                    return _http_error(404, "not found")
            else:
                # Legacy payload shape: relative to ``media_dir``.
                media_root = get_media_dir().resolve()
                candidate = (media_root / rel_str).resolve()
                candidate.relative_to(media_root)
        except (OSError, ValueError):
            return _http_error(404, "not found")
        if not candidate.is_file():
            return _http_error(404, "not found")
        try:
            body = candidate.read_bytes()
        except OSError:
            return _http_error(500, "read error")
        mime, _ = mimetypes.guess_type(candidate.name)
        if mime not in _MEDIA_ALLOWED_MIMES:
            mime = "application/octet-stream"
        return _http_response(
            body,
            content_type=mime,
            extra_headers=[
                ("Cache-Control", "private, max-age=31536000, immutable"),
                # Paired with the MIME whitelist above: prevents browsers from
                # MIME-sniffing an octet-stream fallback into executable HTML.
                ("X-Content-Type-Options", "nosniff"),
            ],
        )

    def _handle_local_asset_fetch(self, request_path: str) -> Response:
        """Serve a content-addressed Memory asset from the local workspace."""
        if self._session_manager is None:
            return _http_error(503, "session manager unavailable")
        path = resolve_local_asset_path(
            request_path,
            workspace=self._session_manager.workspace,
            config=self._tools_config.file_storage.local,
        )
        if path is None:
            return _http_error(404, "not found")
        mime = _guess_allowed_media_mime(path.name)
        if mime is None:
            return _http_error(415, "unsupported media type")
        try:
            body = path.read_bytes()
        except OSError:
            return _http_error(404, "not found")
        return _http_response(
            body,
            content_type=mime,
            extra_headers=[("Cache-Control", "public, max-age=31536000, immutable")],
        )

    def _handle_session_delete(self, request: WsRequest, key: str) -> Response:
        if not self._check_api_token(request):
            return _http_error(401, "Unauthorized")
        if self._session_manager is None:
            return _http_error(503, "session manager unavailable")
        legacy_err = self._legacy_webui_session_key_error(key)
        if legacy_err is not None:
            return legacy_err
        decoded_key = self._resolve_webui_api_session_key(key, request)
        if decoded_key is None:
            if _decode_api_key(key) is None:
                return _http_error(400, "invalid session key")
            return _http_error(404, "session not found")
        deleted = self._session_manager.delete_session(decoded_key)
        return _http_json_response({"deleted": bool(deleted)})

    def _director_root(self) -> Path | None:
        if self._session_manager is None:
            return None
        return self._session_manager.workspace / "director"

    def _workplace_paths(self, work_id: str) -> dict[str, Path] | None:
        root = self._director_root()
        if root is None:
            return None
        work_dir = root / "works" / work_id
        return {
            "work_dir": work_dir,
            "story": work_dir / "story.md",
            "story_profile": work_dir / "story_profile.json",
            "state": work_dir / "state.json",
            "shots": work_dir / "shots",
            "memory_bank": work_dir / "memory" / "memory_bank.json",
            "previous_shot_memory": work_dir / "memory" / "previous_shot.json",
            "manual_memory_workspace": work_dir / "memory" / "manual" / "workspace.json",
            "memory_asset_profiles": work_dir / "memory" / "asset_profiles.json",
        }

    @staticmethod
    def _read_json_file(path: Path, default: Any) -> Any:
        if not path.exists():
            return default
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return default

    @staticmethod
    def _write_json_file(path: Path, payload: dict[str, Any]) -> None:
        write_json_atomic(path, payload)

    @staticmethod
    def _work_id_from_session_map_entry(entry: Any) -> str | None:
        if isinstance(entry, dict):
            work_id = entry.get("active")
        elif isinstance(entry, str):
            work_id = entry
        else:
            return None
        return work_id if isinstance(work_id, str) and work_id.strip() else None

    @staticmethod
    def _session_map_lookup_keys(session_key: str) -> list[str]:
        """Candidate session_map keys, including legacy two-part aliases."""
        keys: list[str] = []
        seen: set[str] = set()

        def add(key: str) -> None:
            if key and key not in seen:
                seen.add(key)
                keys.append(key)

        add(session_key)
        chat_id = webui_wire_chat_id(session_key)
        if chat_id:
            add(f"websocket:{chat_id}")
        return keys

    def _resolve_generation_api_session(self, request: WsRequest, key: str) -> str | Response:
        if not self._check_api_token(request):
            return _http_error(401, "Unauthorized")
        if self._session_manager is None:
            return _http_error(503, "session manager unavailable")
        legacy_err = self._legacy_webui_session_key_error(key)
        if legacy_err is not None:
            return legacy_err
        decoded_key = self._resolve_webui_api_session_key(key, request)
        if decoded_key is None:
            if _decode_api_key(key) is None:
                return _http_error(400, "invalid session key")
            return _http_error(404, "session not found")
        return decoded_key

    def _echo_tracking_payload(self, state: dict[str, Any] | None) -> dict[str, Any]:
        payload = state if isinstance(state, dict) else {}
        return {
            "echo_request_id": payload.get("echo_request_id"),
            "like_status": int(payload.get("like_status") or 0),
            "prompt_downloaded": bool(payload.get("prompt_downloaded")),
            "video_downloaded": bool(payload.get("video_downloaded")),
        }

    def _session_echo_tracking_fields(self, session_key: Any) -> dict[str, Any]:
        if not isinstance(session_key, str) or self._session_manager is None:
            return self._echo_tracking_payload(None)
        work_id = self._resolve_work_id_for_session(session_key)
        if work_id:
            paths = self._workplace_paths(work_id)
            if paths is not None:
                state = self._read_json_file(paths["state"], {})
                if isinstance(state, dict):
                    return self._echo_tracking_payload(state)
        return self._echo_tracking_payload(None)

    def _resolve_echo_tracking_context(self, session_key: str) -> dict[str, Any] | None:
        work_id = self._resolve_work_id_for_session(session_key)
        if work_id:
            paths = self._workplace_paths(work_id)
            if paths is not None:
                state = self._read_json_file(paths["state"], {})
                if isinstance(state, dict):
                    return {
                        "kind": "director",
                        "state": state,
                        "work_id": work_id,
                        "session_key": session_key,
                    }
        return None

    def _save_echo_tracking_context(self, ctx: dict[str, Any]) -> None:
        state = ctx["state"]
        if ctx.get("work_id"):
            self._save_workplace_state(str(ctx["work_id"]), state)

    def _handle_echo_like(self, request: WsRequest, key: str) -> Response:
        resolved = self._resolve_generation_api_session(request, key)
        if not isinstance(resolved, str):
            return resolved
        decoded_key = resolved
        tracking_ctx = self._resolve_echo_tracking_context(decoded_key)
        if tracking_ctx is None:
            return _http_error(409, "echo tracking state is not ready")
        state = tracking_ctx["state"]
        query = _parse_query(request.path)
        raw_status = _query_first(query, "like_status")
        if raw_status is None:
            body = self._parse_json_body_payload(request)
            if isinstance(body, dict):
                raw_status = body.get("like_status")
                if raw_status is None:
                    raw_status = body.get("likeStatus")
        try:
            action = int(raw_status)
        except (TypeError, ValueError):
            return _http_error(400, "invalid like_status")
        if action not in {1, 2}:
            return _http_error(400, "like_status must be 1 or 2")
        current = int(state.get("like_status") or 0)
        next_status = 0 if current == action else action

        state["like_status"] = next_status
        state["updated_at"] = datetime.now(timezone.utc).isoformat()
        self._save_echo_tracking_context(tracking_ctx)
        workplace = self._build_workplace_payload(decoded_key)
        return _http_json_response(
            {
                "ok": True,
                "session_key": decoded_key,
                **self._echo_tracking_payload(state),
                "workplace": workplace,
            }
        )

    def _handle_echo_download_prompt(self, request: WsRequest, key: str) -> Response:
        resolved = self._resolve_generation_api_session(request, key)
        if not isinstance(resolved, str):
            return resolved
        decoded_key = resolved
        tracking_ctx = self._resolve_echo_tracking_context(decoded_key)
        if tracking_ctx is None:
            return _http_error(409, "echo tracking state is not ready")
        state = tracking_ctx["state"]
        state["prompt_downloaded"] = True
        state["updated_at"] = datetime.now(timezone.utc).isoformat()
        self._save_echo_tracking_context(tracking_ctx)
        workplace = self._build_workplace_payload(decoded_key)
        return _http_json_response(
            {
                "ok": True,
                "session_key": decoded_key,
                **self._echo_tracking_payload(state),
                "workplace": workplace,
            }
        )

    def _persist_session_source(self, session_key: str, source: str | None) -> None:
        if self._session_manager is None:
            return
        normalized = normalize_source(source)
        if not normalized:
            return
        session = self._session_manager.get_or_create(session_key)
        if apply_source(session.metadata, normalized):
            self._session_manager.save(session)
            logger.info(
                "Persisted session source={} session_key={}",
                normalized,
                session_key,
            )

    def _persist_session_pe(self, session_key: str, name: str) -> None:
        """Persist the chosen PE set to session metadata so it survives restart/reconnect."""
        if not session_key or self._session_manager is None:
            return
        session = self._session_manager.get_or_create(session_key)
        if session.metadata.get("pe_set") == name:
            return
        session.metadata["pe_set"] = name
        self._session_manager.save(session)
        logger.info("Persisted pe_set={} session_key={}", name, session_key)

    def _hydrate_session_pe(self, session_key: str) -> None:
        """Restore a persisted PE selection into the PEManager if not already bound."""
        if not session_key or self._session_manager is None:
            return
        from nanobot.prompts import PEManager

        manager = PEManager.instance()
        if manager.active_for_session(session_key) != manager.active:
            return  # already has an in-memory override
        session = self._session_manager.get_or_create(session_key)
        stored = session.metadata.get("pe_set")
        if isinstance(stored, str) and stored:
            manager.set_active_for_session(session_key, stored)

    def _handle_pe_list(self, request: WsRequest) -> Response:
        if not self._check_api_token(request):
            return _http_error(401, "Unauthorized")
        from nanobot.prompts import PEManager

        manager = PEManager.instance()
        return _http_json_response(
            {
                "ok": True,
                "sets": manager.list_sets(),
                "active": manager.active,
                "enabled": manager.enabled,
            }
        )

    def _handle_generation_settings_get(self, request: WsRequest, key: str) -> Response:
        if not self._check_api_token(request):
            return _http_error(401, "Unauthorized")
        if self._session_manager is None:
            return _http_error(503, "session manager unavailable")
        legacy_err = self._legacy_webui_session_key_error(key)
        if legacy_err is not None:
            return legacy_err
        decoded_key = self._resolve_webui_api_session_key(key, request)
        if decoded_key is None:
            if _decode_api_key(key) is None:
                return _http_error(400, "invalid session key")
            return _http_error(404, "session not found")

        from nanobot.session.generation_settings import (
            get_generation_settings,
            get_llm_sampling_for_api,
        )

        session = self._session_manager.get_or_create(decoded_key)
        settings = get_generation_settings(session.metadata)
        settings.update(get_llm_sampling_for_api(session.metadata))
        return _http_json_response({"ok": True, "session_key": decoded_key, **settings})

    def _parse_generation_settings_body(self, request: WsRequest) -> dict[str, Any]:
        query = _parse_query(request.path)
        body = self._parse_json_body_payload(request)
        payload: dict[str, Any] = {}
        if isinstance(body, dict):
            payload.update(body)
        for q_key, p_key in (
            ("duration_sec", "duration_sec"),
            ("n_shots", "n_shots"),
            ("width", "width"),
            ("height", "height"),
            ("language", "language"),
            ("temperature", "temperature"),
            ("top_p", "top_p"),
            ("top_k", "top_k"),
        ):
            raw = _query_first(query, q_key)
            if raw is not None:
                payload[p_key] = raw
        if payload.get("duration_sec") is None and "durationSec" in payload:
            payload["duration_sec"] = payload.get("durationSec")
        if payload.get("n_shots") is None and payload.get("nShot") is not None:
            payload["n_shots"] = payload.get("nShot")
        if payload.get("top_p") is None and payload.get("topP") is not None:
            payload["top_p"] = payload.get("topP")
        if payload.get("top_k") is None and payload.get("topK") is not None:
            payload["top_k"] = payload.get("topK")
        return payload

    def _handle_generation_settings_save(self, request: WsRequest, key: str) -> Response:
        if not self._check_api_token(request):
            return _http_error(401, "Unauthorized")
        if self._session_manager is None:
            return _http_error(503, "session manager unavailable")
        legacy_err = self._legacy_webui_session_key_error(key)
        if legacy_err is not None:
            return legacy_err
        decoded_key = self._resolve_webui_api_session_key(key, request)
        if decoded_key is None:
            if _decode_api_key(key) is None:
                return _http_error(400, "invalid session key")
            return _http_error(404, "session not found")

        from nanobot.session.generation_settings import (
            apply_generation_settings,
            apply_llm_sampling_settings,
            get_generation_settings,
            get_llm_sampling_for_api,
            normalize_duration_sec,
            normalize_language,
            normalize_n_shots,
        )

        payload = self._parse_generation_settings_body(request)
        duration_raw = payload.get("duration_sec")
        nshot_raw = payload.get("n_shots")
        width_raw = payload.get("width")
        height_raw = payload.get("height")
        language_raw = payload.get("language")
        has_duration = duration_raw is not None and duration_raw != ""
        has_nshots = nshot_raw is not None and nshot_raw != ""
        has_size = width_raw not in (None, "") or height_raw not in (None, "")
        has_language = language_raw not in (None, "")
        has_llm = any(
            key in payload
            for key in ("temperature", "top_p", "top_k", "topP", "topK")
        )
        if (
            not has_duration
            and not has_nshots
            and not has_size
            and not has_language
            and not has_llm
        ):
            return _http_error(400, "no settings to save")

        duration_sec = normalize_duration_sec(duration_raw) if has_duration else None
        n_shots = normalize_n_shots(nshot_raw) if has_nshots else None
        language = normalize_language(language_raw) if has_language else None
        if has_duration and duration_sec is None:
            return _http_error(400, f"invalid duration_sec: {duration_raw}")
        if has_nshots and n_shots is None:
            return _http_error(400, f"invalid n_shots: {nshot_raw}")
        if has_language and language is None:
            return _http_error(400, f"invalid language: {language_raw}")

        session = self._session_manager.get_or_create(decoded_key)
        try:
            if duration_sec is not None or n_shots is not None or has_size or language is not None:
                apply_generation_settings(
                    session.metadata,
                    n_shots=n_shots,
                    duration_sec=duration_sec,
                    width=width_raw if width_raw not in (None, "") else None,
                    height=height_raw if height_raw not in (None, "") else None,
                    language=language,
                )
            llm_kwargs: dict[str, Any] = {}
            if "temperature" in payload:
                llm_kwargs["temperature"] = payload.get("temperature")
            if "top_p" in payload or "topP" in payload:
                llm_kwargs["top_p"] = payload.get("top_p", payload.get("topP"))
            if "top_k" in payload or "topK" in payload:
                llm_kwargs["top_k"] = payload.get("top_k", payload.get("topK"))
            if llm_kwargs:
                apply_llm_sampling_settings(session.metadata, **llm_kwargs)
        except ValueError as exc:
            return _http_error(400, str(exc))
        self._session_manager.save(session)
        if language is not None:
            self._sync_story_profile_language(decoded_key, language)
        settings = get_generation_settings(session.metadata)
        settings.update(get_llm_sampling_for_api(session.metadata))
        return _http_json_response({"ok": True, "session_key": decoded_key, **settings})






    _I2V_IMAGE_DOWNLOAD_TIMEOUT_S = 15.0
    _I2V_IMAGE_MAX_BYTES = 20 * 1024 * 1024  # 20 MiB

    @staticmethod
    def _shot_recaption_prompt_path() -> Path:
        return (
            Path(__file__).resolve().parents[2]
            / "pe"
            / "v7_cinematic_full"
            / "references"
            / "shot-prompt-writer.md"
        )

    @staticmethod
    def _i2v_prompt_skill_path() -> Path:
        return (
            Path(__file__).resolve().parents[2]
            / "pe"
            / "v7_cinematic_full"
            / "skills"
            / "i2v-tail-frame-prompt-rewriter"
            / "SKILL.md"
        )

    @staticmethod
    async def _http_download_image_as_base64_uri(image_url: str) -> str:
        """Download an image and return a data URI for the configured VLM."""
        loop = asyncio.get_running_loop()

        def _download() -> tuple[str, bytes]:
            request = urllib.request.Request(image_url, method="GET")
            with urllib.request.urlopen(
                request,
                timeout=WebSocketChannel._I2V_IMAGE_DOWNLOAD_TIMEOUT_S,
            ) as response:
                return response.headers.get("Content-Type", ""), response.read()

        try:
            content_type, data = await loop.run_in_executor(None, _download)
        except urllib.error.HTTPError as exc:
            raise RuntimeError(
                f"Failed to download reference image (HTTP {exc.code}): {image_url}"
            ) from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise RuntimeError(f"Failed to download reference image: {exc}") from exc

        if len(data) > WebSocketChannel._I2V_IMAGE_MAX_BYTES:
            raise RuntimeError(
                f"Reference image too large ({len(data)} bytes, max "
                f"{WebSocketChannel._I2V_IMAGE_MAX_BYTES})"
            )

        mime_type = content_type.split(";")[0].strip()
        if not mime_type or mime_type == "application/octet-stream":
            extension = os.path.splitext(urlparse(image_url).path)[1].lower()
            mime_type = {
                ".jpg": "image/jpeg",
                ".jpeg": "image/jpeg",
                ".png": "image/png",
                ".webp": "image/webp",
                ".gif": "image/gif",
                ".bmp": "image/bmp",
            }.get(extension, "image/jpeg")

        encoded = base64.b64encode(data).decode("ascii")
        return f"data:{mime_type};base64,{encoded}"

    async def _rewrite_i2v_prompt_with_image(
        self,
        text: str,
        condition_image_url: str,
        story_profile: dict[str, Any] | None = None,
        *,
        is_first_frame: bool = False,
        enforce_first_frame_continuity: bool = False,
    ) -> str:
        """Ground an existing Director caption in an I2V condition image."""
        if self._provider is None or not self._model:
            raise RuntimeError("I2V caption model unavailable")
        if not isinstance(condition_image_url, str) or not condition_image_url.strip():
            raise RuntimeError("I2V condition image URL is missing")
        try:
            ordinary_prompt = self._shot_recaption_prompt_path().read_text(
                encoding="utf-8"
            )
            i2v_prompt = self._i2v_prompt_skill_path().read_text(encoding="utf-8")
        except OSError as exc:
            raise RuntimeError(f"I2V prompt resources unavailable: {exc}") from exc

        profile = story_profile if isinstance(story_profile, dict) else {}
        language = str(
            profile.get("caption_language") or profile.get("language") or ""
        )
        if is_first_frame and enforce_first_frame_continuity:
            mode_instruction = (
                "The supplied image is the authoritative first frame (frame 0). "
                "First understand every visible character, object, position, pose, "
                "environment, composition, camera position, lighting, and action state. "
                "Preserve the intended story beat, character IDs, and dialogue, while "
                "making the opening physically and temporally continuous from that image."
            )
            user_instruction = (
                "First understand the supplied first-frame image, then rewrite the "
                "caption so the action is physically and temporally continuous from it."
            )
        elif is_first_frame:
            mode_instruction = (
                "The supplied image is the user-provided first frame. Preserve the "
                "caption's story and use the image as its visual starting point."
            )
            user_instruction = (
                "Rewrite the caption for first-frame I2V while preserving its story."
            )
        else:
            mode_instruction = (
                "The supplied image is the authoritative previous-shot tail frame. "
                "Preserve the story beat while continuing naturally from that frame."
            )
            user_instruction = (
                "Rewrite the caption for I2V continuation from the supplied tail frame."
            )

        system_prompt = (
            f"{ordinary_prompt}\n\n# I2V REWRITE SKILL\n{i2v_prompt}\n\n"
            f"{mode_instruction} Return one complete generation caption only. "
            f"The caption language is locked to "
            f"{language or 'the original caption language'}."
        )
        user_text = f"{user_instruction}\n\nALREADY-GENERATED CAPTION:\n{text.strip()}"
        image_uri = await self._http_download_image_as_base64_uri(
            condition_image_url.strip()
        )
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_uri}},
                    {"type": "text", "text": user_text},
                ],
            },
        ]

        started = time.monotonic()
        response = await self._provider.chat_with_retry(
            model=self._model,
            messages=messages,
            tools=None,
            max_tokens=4096,
            temperature=0.4,
        )
        if response.finish_reason == "error":
            raise RuntimeError(
                f"I2V caption model failed: {response.content or 'unknown error'}"
            )
        rewritten = " ".join((response.content or "").split()).strip()
        if not rewritten:
            raise RuntimeError("I2V caption model returned empty output")
        language_error = _caption_language_validation_error(rewritten, profile)
        if language_error:
            retry_messages = [
                *messages,
                {"role": "assistant", "content": rewritten},
                {
                    "role": "user",
                    "content": (
                        "The previous result failed caption-language validation: "
                        f"{language_error.removeprefix('Error: ')}. Correct only the "
                        "language issue and return the complete caption without commentary."
                    ),
                },
            ]
            response = await self._provider.chat_with_retry(
                model=self._model,
                messages=retry_messages,
                tools=None,
                max_tokens=4096,
                temperature=0.4,
            )
            if response.finish_reason == "error":
                raise RuntimeError(
                    "I2V caption model failed during language correction: "
                    f"{response.content or 'unknown error'}"
                )
            rewritten = " ".join((response.content or "").split()).strip()
            language_error = _caption_language_validation_error(rewritten, profile)
            if language_error:
                raise RuntimeError(language_error.removeprefix("Error: "))
        logger.info(
            "I2V caption rewrite done model={} elapsed_s={:.2f} caption_chars={}",
            self._model,
            time.monotonic() - started,
            len(rewritten),
        )
        return rewritten

    @staticmethod
    def _user_facing_generation_error(error: str) -> str:
        text = (error or "").strip()
        lowered = text.lower()
        if (
            "connection refused" in lowered
            or "errno 111" in lowered
            or UNAVAILABLE_MESSAGE in text
        ):
            return UNAVAILABLE_MESSAGE
        return text[:500]

    def _mark_workplace_shot_failed(
        self,
        session_key: str,
        work_id: str,
        shot_id: int,
        error: str,
    ) -> None:
        """Persist a Director shot-generation failure for the WebUI."""
        try:
            chat_id = webui_wire_chat_id(session_key) or "direct"
            tool = self._director_generate_tool()
            tool.set_context("websocket", chat_id, effective_key=session_key)
            state = tool._load_state(work_id)
            shot = tool._load_shot(work_id, shot_id)
            shot["status"] = "error"
            shot["generation_error"] = self._user_facing_generation_error(error)
            state["stage"] = "failed"
            state["generation_error"] = shot["generation_error"]
            tool._save_shot(work_id, shot_id, shot)
            shots = state.setdefault("shots", {})
            if isinstance(shots, dict):
                shots[_shot_key(shot_id)] = tool._state_shot_entry(shot)
            tool._save_state(work_id, state)
        except Exception:
            logger.opt(exception=True).error(
                "failed to persist Director generation error work_id={}",
                work_id,
            )








    def _report_echo_unavailable(
        self,
        session_key: str,
        exc: BaseException,
        *,
        work_id: str | None = None,
        shot_id: int | None = None,
    ) -> str:
        message = UNAVAILABLE_MESSAGE
        resolved_work = work_id or self._resolve_work_id_for_session(session_key)
        logger.error(
            "Echo unavailable session_key={} work_id={} shot_id={} error={}",
            session_key,
            resolved_work,
            shot_id,
            exc,
        )
        if resolved_work:
            target_shot = shot_id
            if target_shot is None:
                for sid, shot in self._iter_workplace_shots(resolved_work):
                    status = str(shot.get("status") or "")
                    if status not in {"generated", "review_pass", "approved"}:
                        target_shot = sid
                        break
            if target_shot is not None:
                self._mark_workplace_shot_failed(
                    session_key, resolved_work, target_shot, message
                )
            else:
                try:
                    state = self._load_workplace_state(resolved_work)
                    state["stage"] = "failed"
                    state["generation_error"] = message
                    self._save_workplace_state(resolved_work, state)
                except Exception:
                    logger.opt(exception=True).error(
                        "failed to persist Echo unavailable state work_id={}",
                        resolved_work,
                    )
            self._schedule_publish_workplace_update(session_key)
        return message

    def _http_echo_gate_error(
        self,
        session_key: str,
        exc: BaseException,
        *,
        shot_id: int | None = None,
    ) -> Response:
        if isinstance(exc, EchoGeneratorUnavailableError) or is_connection_refused(exc):
            return _http_error(
                503,
                self._report_echo_unavailable(session_key, exc, shot_id=shot_id),
            )
        return _http_error(503, str(exc))





    def _envelope_source(self, envelope: dict[str, Any]) -> str | None:
        from nanobot.session.source import resolve_source_from_wire
        return resolve_source_from_wire(envelope)

    def _resolve_work_id_for_session(self, session_key: str) -> str | None:
        root = self._director_root()
        if root is None:
            return None
        session_map = self._read_json_file(root / "session_map.json", {})
        if not isinstance(session_map, dict):
            return None
        for key in self._session_map_lookup_keys(session_key):
            work_id = self._work_id_from_session_map_entry(session_map.get(key))
            if work_id:
                return work_id
        active = self._read_json_file(root / "active_work.json", {})
        if isinstance(active, dict):
            active_work_id = active.get("work_id")
            if not isinstance(active_work_id, str) or not active_work_id.strip():
                return None
            active_session = active.get("session_key")
            if isinstance(active_session, str) and active_session in self._session_map_lookup_keys(
                session_key
            ):
                return active_work_id.strip()
        return None

    def _webui_session_key_for_chat(self, chat_id: str) -> str | None:
        """Best-effort session key for workplace payloads keyed by wire chat_id."""
        for connection in self._subs.get(chat_id, ()):
            try:
                return self._webui_session_key_for_connection(connection, chat_id)
            except ValueError:
                continue
        root = self._director_root()
        if root is None:
            return None
        session_map = self._read_json_file(root / "session_map.json", {})
        if not isinstance(session_map, dict):
            return None
        suffix = f":{chat_id}"
        for map_key in session_map:
            if isinstance(map_key, str) and map_key.endswith(suffix):
                return map_key
        return None

    @staticmethod
    def _parse_iso_timestamp(raw: Any) -> datetime | None:
        if not isinstance(raw, str) or not raw.strip():
            return None
        try:
            return datetime.fromisoformat(raw.replace("Z", "+00:00"))
        except ValueError:
            return None

    @staticmethod
    def _format_timeline_time(seconds: float) -> str:
        total = max(0, int(round(seconds)))
        minutes, secs = divmod(total, 60)
        return f"{minutes:02d}:{secs:02d}"

    def _estimate_shot_duration_seconds(
        self,
        shot: dict[str, Any],
        goal_duration_s: float,
    ) -> float:
        goal = {"shot_duration_sec": goal_duration_s} if goal_duration_s > 0 else {}
        return resolve_echo_duration_seconds(shot, {"goal": goal})

    @staticmethod
    def _memory_display_name(memory_id: str) -> str:
        """Translate internal memory IDs to user-facing Chinese labels."""
        if memory_id.startswith("ID_"):
            return "角色_" + memory_id[3:]
        if memory_id == "PREVIOUS_SHOT":
            return "场景参考"
        return memory_id

    def _project_memory_selection(
        self, raw: Any
    ) -> dict[str, Any] | None:
        if not isinstance(raw, dict):
            return None
        memory_id = str(raw.get("memory_id") or raw.get("character_id") or "").strip()
        if not memory_id:
            return None
        projected: dict[str, Any] = {
            "memory_id": memory_id,
            "display_name": self._memory_display_name(memory_id),
            "kind": str(raw.get("kind") or "character"),
            "candidate_index": int(raw.get("candidate_index") or 0),
            "frame_index": int(raw.get("frame_index") or 0),
            "timestamp_sec": float(raw.get("timestamp_sec") or 0.0),
            "confidence": float(raw.get("confidence") or 0.0),
            "visual_status": str(raw.get("visual_status") or "provisional"),
            "reasoning": str(raw.get("reasoning") or ""),
            "source_shot_id": int(raw.get("source_shot_id") or 0),
            "audio_source_shot_id": int(
                raw.get("audio_source_shot_id") or raw.get("source_shot_id") or 0
            ),
        }
        for media_kind in ("image", "audio"):
            existing = raw.get(media_kind)
            if isinstance(existing, dict):
                url = existing.get("url")
                if isinstance(url, str):
                    public = self._public_media_entry(url)
                    if public is not None:
                        if existing.get("name"):
                            public["name"] = str(existing["name"])
                        projected[media_kind] = public
                        continue
            locator = None
            for key in (
                f"local_{media_kind}_path",
                f"{media_kind}_path",
                f"{media_kind}_url",
            ):
                value = raw.get(key)
                if isinstance(value, str) and value.strip():
                    locator = value.strip()
                    break
            if locator:
                public = self._public_media_entry(locator)
                if public is not None:
                    projected[media_kind] = public
        return projected

    def _project_memory_review(self, raw: Any) -> dict[str, Any] | None:
        if not isinstance(raw, dict):
            return None
        review_id = str(raw.get("review_id") or "").strip()
        if not review_id:
            return None
        selections = [
            projected
            for item in raw.get("selections", [])
            if (projected := self._project_memory_selection(item)) is not None
        ]
        history: list[dict[str, Any]] = []
        for attempt in raw.get("history", []):
            if not isinstance(attempt, dict):
                continue
            history.append(
                {
                    "attempt": int(attempt.get("attempt") or 0),
                    "selections": [
                        projected
                        for item in attempt.get("selections", [])
                        if (projected := self._project_memory_selection(item)) is not None
                    ],
                    "rejected_at": attempt.get("rejected_at"),
                }
            )
        projected_review = {
            "review_id": review_id,
            "status": str(raw.get("status") or "selecting"),
            "attempt": int(raw.get("attempt") or 1),
            "candidate_count": int(raw.get("candidate_count") or 0),
            "rejected_candidate_indices": [
                int(value) for value in raw.get("rejected_candidate_indices", [])
            ],
            "selections": selections,
            "history": history,
            "selection_mode": raw.get("selection_mode"),
            "required_memory_ids": list(raw.get("required_memory_ids") or []),
            "manual_selected_ids": list(raw.get("manual_selected_ids") or []),
            "error": raw.get("error"),
            "updated_at": raw.get("updated_at"),
        }
        if "retained_memory_ids" in raw:
            projected_review["retained_memory_ids"] = list(
                raw.get("retained_memory_ids") or []
            )
        return projected_review

    def _project_memory_bank(self, paths: dict[str, Path]) -> list[dict[str, Any]]:
        """Project the work's durable character bank plus current continuity slot."""
        bank = self._read_json_file(paths["memory_bank"], {})
        entries: list[dict[str, Any]] = []
        if isinstance(bank, dict):
            for memory_id in sorted(bank):
                raw = bank.get(memory_id)
                if not isinstance(raw, dict):
                    continue
                projected = self._project_memory_selection(
                    {
                        **raw,
                        "memory_id": str(raw.get("memory_id") or memory_id),
                        "kind": "character",
                    }
                )
                if projected is not None and projected.get("image"):
                    entries.append(projected)

        previous = self._read_json_file(paths["previous_shot_memory"], None)
        if isinstance(previous, dict):
            projected = self._project_memory_selection(
                {
                    **previous,
                    "memory_id": "PREVIOUS_SHOT",
                    "kind": "previous_shot",
                }
            )
            if projected is not None and projected.get("image"):
                entries.append(projected)
        return entries

    @staticmethod
    def _memory_workspace_asset_id(raw: dict[str, Any]) -> str:
        """Build a stable opaque id for an automatically selected Memory item."""
        fingerprint = json.dumps(
            [
                str(raw.get("memory_id") or raw.get("character_id") or ""),
                int(raw.get("source_shot_id") or 0),
                int(raw.get("frame_index") or 0),
                str(raw.get("kind") or "character"),
            ],
            ensure_ascii=False,
            separators=(",", ":"),
        )
        return "auto_" + hashlib.sha256(fingerprint.encode("utf-8")).hexdigest()[:20]

    def _automatic_memory_workspace_records(
        self,
        paths: dict[str, Path],
    ) -> list[tuple[str, dict[str, Any]]]:
        """Collect reusable VLM selections and canonical bank items in stable order."""
        records: list[tuple[str, dict[str, Any]]] = []
        seen: set[str] = set()

        def add(raw: Any) -> None:
            if not isinstance(raw, dict):
                return
            projected = self._project_memory_selection(raw)
            if projected is None or not projected.get("image"):
                return
            asset_id = self._memory_workspace_asset_id(raw)
            if asset_id in seen:
                return
            seen.add(asset_id)
            records.append((asset_id, raw))

        for shot_path in sorted(paths["shots"].glob("shot_*.json")):
            shot = self._read_json_file(shot_path, {})
            review = shot.get("memory_review") if isinstance(shot, dict) else None
            if not isinstance(review, dict):
                continue
            for selection in review.get("selections", []):
                add(selection)

        bank = self._read_json_file(paths["memory_bank"], {})
        if isinstance(bank, dict):
            for memory_id in sorted(bank):
                raw = bank.get(memory_id)
                if isinstance(raw, dict):
                    add(
                        {
                            **raw,
                            "memory_id": str(raw.get("memory_id") or memory_id),
                            "kind": "character",
                        }
                    )
        previous = self._read_json_file(paths["previous_shot_memory"], None)
        if isinstance(previous, dict):
            add({**previous, "memory_id": "PREVIOUS_SHOT", "kind": "previous_shot"})
        return records

    def _load_manual_memory_workspace(self, paths: dict[str, Path]) -> list[dict[str, Any]]:
        raw = self._read_json_file(paths["manual_memory_workspace"], {})
        assets = raw.get("assets") if isinstance(raw, dict) else None
        return [dict(item) for item in assets or [] if isinstance(item, dict)]

    def _save_manual_memory_workspace(
        self,
        paths: dict[str, Path],
        assets: list[dict[str, Any]],
    ) -> None:
        self._write_json_file(
            paths["manual_memory_workspace"],
            {
                "assets": assets,
                "updated_at": datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
            },
        )

    def _load_memory_asset_profiles(self, paths: dict[str, Path]) -> dict[str, dict[str, Any]]:
        raw = self._read_json_file(paths["memory_asset_profiles"], {})
        return {
            str(asset_id): dict(profile)
            for asset_id, profile in raw.items()
            if isinstance(asset_id, str) and isinstance(profile, dict)
        } if isinstance(raw, dict) else {}

    def _save_memory_asset_profiles(
        self, paths: dict[str, Path], profiles: dict[str, dict[str, Any]]
    ) -> None:
        self._write_json_file(paths["memory_asset_profiles"], profiles)

    def _project_memory_workspace_assets(
        self,
        paths: dict[str, Path],
    ) -> list[dict[str, Any]]:
        assets: list[dict[str, Any]] = []
        profile_overrides = self._load_memory_asset_profiles(paths)
        for asset_id, raw in self._automatic_memory_workspace_records(paths):
            projected = self._project_memory_selection(raw)
            if projected is None or not isinstance(projected.get("image"), dict):
                continue
            asset = {
                    "asset_id": asset_id,
                    "display_name": projected.get("display_name") or projected["memory_id"],
                    "source": "automatic",
                    "kind": projected.get("kind") or "character",
                    "memory_id": projected["memory_id"],
                    "source_shot_id": projected.get("source_shot_id"),
                    "frame_index": projected.get("frame_index"),
                    "image": projected["image"],
                    "audio": projected.get("audio"),
                    "media_type": "image_audio" if projected.get("audio") else "image",
                    "profile_text": str(
                        raw.get("profile_text") or raw.get("reasoning") or ""
                    ).strip(),
                    "profile_status": (
                        "ready"
                        if str(raw.get("profile_text") or raw.get("reasoning") or "").strip()
                        else "missing"
                    ),
                    "profile_source": str(
                        raw.get("profile_source")
                        or ("vlm" if raw.get("reasoning") else "none")
                    ),
                    "identity_ids": list(
                        raw.get("visible_character_ids")
                        or ([projected["memory_id"]] if str(projected["memory_id"]).startswith("ID_") else [])
                    ),
                    "provenance": {
                        "source": "generated_shot",
                        "shot_id": projected.get("source_shot_id"),
                        "timestamp_sec": projected.get("timestamp_sec"),
                    },
                }
            override = profile_overrides.get(asset_id)
            if isinstance(override, dict):
                asset.update({
                    key: override[key]
                    for key in (
                        "profile_text",
                        "profile_status",
                        "profile_source",
                        "identity_ids",
                        "reference_type",
                        "reference_label",
                    )
                    if key in override
                })
            assets.append(asset)

        for raw in self._load_manual_memory_workspace(paths):
            asset_id = str(raw.get("asset_id") or "").strip()
            if not asset_id:
                continue
            projected = self._project_memory_selection(
                {
                    **raw,
                    "memory_id": asset_id,
                    "kind": "manual",
                }
            )
            if projected is None or (
                not isinstance(projected.get("image"), dict)
                and not isinstance(projected.get("audio"), dict)
            ):
                continue
            image = dict(projected["image"]) if isinstance(projected.get("image"), dict) else None
            audio = dict(projected["audio"]) if isinstance(projected.get("audio"), dict) else None
            if image is not None and raw.get("image_name"):
                image["name"] = str(raw["image_name"])
            if audio is not None and raw.get("audio_name"):
                audio["name"] = str(raw["audio_name"])
            assets.append(
                {
                    "asset_id": asset_id,
                    "display_name": str(
                        raw.get("display_name")
                        or (image or {}).get("name")
                        or (audio or {}).get("name")
                        or "Local asset"
                    ),
                    "source": "local",
                    "kind": "manual",
                    **({"image": image} if image is not None else {}),
                    "audio": audio,
                    "media_type": (
                        "image_audio" if image is not None and audio is not None
                        else "image" if image is not None
                        else "audio"
                    ),
                    "profile_text": str(raw.get("profile_text") or "").strip(),
                    "profile_status": str(
                        raw.get("profile_status")
                        or ("ready" if str(raw.get("profile_text") or "").strip() else "missing")
                    ),
                    "profile_source": str(raw.get("profile_source") or "none"),
                    "identity_ids": [
                        str(value)
                        for value in raw.get("identity_ids", [])
                        if isinstance(value, str) and value.strip()
                    ],
                    **(
                        {"reference_type": str(raw["reference_type"]).strip()}
                        if str(raw.get("reference_type") or "").strip()
                        in _MEMORY_REFERENCE_TYPES
                        else {}
                    ),
                    **(
                        {"reference_label": str(raw["reference_label"]).strip()[:80]}
                        if str(raw.get("reference_label") or "").strip()
                        else {}
                    ),
                    "provenance": {
                        "source": (
                            "generated_shot"
                            if raw.get("source_shot_id")
                            else "local_upload"
                        ),
                        "shot_id": raw.get("source_shot_id"),
                        "timestamp_sec": raw.get("timestamp_sec"),
                    },
                }
            )
        return assets

    @staticmethod
    def _memory_slot_locator(
        raw: dict[str, Any],
        media_kind: str,
        *,
        display: bool,
    ) -> str | None:
        keys = (
            (f"local_{media_kind}_path", f"{media_kind}_path", f"{media_kind}_url")
            if display
            else (f"{media_kind}_path", f"{media_kind}_url", f"local_{media_kind}_path")
        )
        for key in keys:
            value = raw.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        nested = raw.get(media_kind)
        if isinstance(nested, dict):
            value = nested.get("url")
            if isinstance(value, str) and value.strip():
                return value.strip()
        return None

    def _memory_workspace_slot(
        self,
        raw: dict[str, Any],
        *,
        asset_id: str,
        source: str,
        display: bool,
    ) -> dict[str, Any]:
        image = self._memory_slot_locator(raw, "image", display=display)
        if not image:
            raise ValueError("memory asset image is unavailable")
        memory_id = str(raw.get("memory_id") or raw.get("character_id") or asset_id)
        display_name = str(raw.get("display_name") or self._memory_display_name(memory_id))
        reference_type = str(raw.get("reference_type") or "").strip()
        reference_label = str(raw.get("reference_label") or "").strip()[:80]
        identity_ids = list(dict.fromkeys(
            str(value).strip()[:80]
            for value in raw.get("identity_ids", [])
            if isinstance(value, str) and value.strip()
        ))
        local_memory_id = reference_label or (identity_ids[0] if identity_ids else asset_id)
        metadata: dict[str, Any] = {
            "id": memory_id if source == "automatic" else local_memory_id,
            "workspace_asset_id": asset_id,
            "display_name": display_name,
            "source": (
                str(raw.get("kind") or "automatic_memory")
                if source == "automatic"
                else "manual_workspace"
            ),
        }
        if reference_type in _MEMORY_REFERENCE_TYPES:
            metadata["reference_type"] = reference_type
        if reference_label:
            metadata["reference_label"] = reference_label
        if identity_ids:
            metadata["identity_ids"] = identity_ids
        profile_text = str(raw.get("profile_text") or "").strip()[:2000]
        if profile_text:
            metadata["profile_text"] = profile_text
        for key in (
            "visual_status",
            "source_shot_id",
            "frame_index",
            "timestamp_sec",
            "confidence",
            "audio_source_shot_id",
        ):
            if key in raw:
                metadata[key] = raw[key]
        slot: dict[str, Any] = {"image_url": image, "metadata": metadata}
        audio = self._memory_slot_locator(raw, "audio", display=display)
        if audio:
            slot["audio_url"] = audio
        else:
            slot["audio_mode"] = "empty"
        return slot

    def _persist_memory_asset_upload(
        self,
        *,
        work_id: str,
        asset_id: str,
        media_kind: str,
        payload: Any,
    ) -> dict[str, str]:
        if not isinstance(payload, dict):
            raise ValueError(f"{media_kind} upload is malformed")
        data_url = payload.get("data_url")
        if not isinstance(data_url, str) or not data_url:
            raise ValueError(f"{media_kind} data_url is required")
        mime = _extract_data_url_mime(data_url)
        allowed = _IMAGE_MIME_ALLOWED if media_kind == "image" else _AUDIO_MIME_ALLOWED
        if mime not in allowed:
            raise ValueError(f"unsupported {media_kind} type")
        limit = _MAX_IMAGE_BYTES if media_kind == "image" else _MAX_MEMORY_AUDIO_BYTES
        temp_path: Path | None = None
        try:
            saved = save_base64_data_url(data_url, get_media_dir("websocket"), max_bytes=limit)
            if saved is None:
                raise ValueError(f"invalid {media_kind} data")
            temp_path = Path(saved)
            publisher = configured_file_publisher(
                work_id,
                storage=self._tools_config.file_storage,
                workspace=self._session_manager.workspace,
            )
            public_url = publisher(
                str(temp_path),
                f"memory/manual/{asset_id}/{media_kind}{temp_path.suffix}",
            )
            local = resolve_local_asset_path(
                public_url,
                workspace=self._session_manager.workspace,
                config=self._tools_config.file_storage.local,
            )
            result = {f"{media_kind}_path": public_url}
            if local is not None:
                result[f"local_{media_kind}_path"] = str(local)
            name = payload.get("name")
            if isinstance(name, str) and name.strip():
                result[f"{media_kind}_name"] = Path(name.strip()).name[:160]
            return result
        except FileSizeExceeded as exc:
            raise ValueError(f"{media_kind} exceeds size limit") from exc
        finally:
            if temp_path is not None:
                temp_path.unlink(missing_ok=True)

    def _apply_workplace_memory_asset_save(
        self,
        session_key: str,
        payload: Any,
    ) -> tuple[str, dict[str, Any]]:
        if not isinstance(payload, dict):
            raise ValueError("asset is required")
        work_id = self._resolve_work_id_for_session(session_key)
        if not work_id:
            raise ValueError("workplace is not initialized")
        paths = self._workplace_paths(work_id)
        if paths is None:
            raise ValueError("workplace is unavailable")
        assets = self._load_manual_memory_workspace(paths)
        requested_id = str(payload.get("asset_id") or "").strip()
        existing_index = next(
            (index for index, item in enumerate(assets) if item.get("asset_id") == requested_id),
            None,
        ) if requested_id else None
        if requested_id and existing_index is None:
            automatic_ids = {
                asset_id for asset_id, _raw in self._automatic_memory_workspace_records(paths)
            }
            if requested_id not in automatic_ids:
                raise ValueError("memory asset not found")
            if any(payload.get(key) is not None for key in ("image", "audio")):
                raise ValueError("automatic asset media cannot be replaced")
            profiles = self._load_memory_asset_profiles(paths)
            profile_text = str(payload.get("profile_text") or "").strip()[:2000]
            identities = payload.get("identity_ids")
            existing_profile = profiles.get(requested_id)
            saved_profile = dict(existing_profile) if isinstance(existing_profile, dict) else {}
            saved_profile.update({
                "profile_text": profile_text,
                "profile_status": "ready" if profile_text else "missing",
                "profile_source": "human" if profile_text else "none",
                "updated_at": datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
            })
            self._apply_memory_reference_fields(saved_profile, payload)
            if isinstance(identities, list):
                saved_profile["identity_ids"] = list(dict.fromkeys(
                    str(value).strip()[:80]
                    for value in identities
                    if isinstance(value, str) and value.strip()
                ))
            elif "reference_type" in payload or "reference_label" in payload:
                if (
                    saved_profile.get("reference_type") == "character"
                    and saved_profile.get("reference_label")
                ):
                    saved_profile["identity_ids"] = [saved_profile["reference_label"]]
                elif saved_profile.get("reference_type") != "character":
                    saved_profile["identity_ids"] = []
            profiles[requested_id] = saved_profile
            self._save_memory_asset_profiles(paths, profiles)
            return work_id, self._build_workplace_payload(session_key)
        if existing_index is None and len(assets) >= _MAX_MEMORY_WORKSPACE_ASSETS:
            raise ValueError("memory workspace is full")
        asset_id = requested_id or uuid.uuid4().hex
        now = datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
        asset = dict(assets[existing_index]) if existing_index is not None else {
            "asset_id": asset_id,
            "created_at": now,
        }
        image_payload = payload.get("image")
        if image_payload is not None:
            asset.update(
                self._persist_memory_asset_upload(
                    work_id=work_id,
                    asset_id=asset_id,
                    media_kind="image",
                    payload=image_payload,
                )
            )
        audio_payload = payload.get("audio")
        if audio_payload is not None:
            asset.update(
                self._persist_memory_asset_upload(
                    work_id=work_id,
                    asset_id=asset_id,
                    media_kind="audio",
                    payload=audio_payload,
                )
            )
        if not asset.get("image_path") and not asset.get("audio_path"):
            raise ValueError("new memory asset requires image or audio")
        if payload.get("remove_audio") is True:
            for key in ("audio_path", "local_audio_path", "audio_name"):
                asset.pop(key, None)
            if not asset.get("image_path"):
                raise ValueError("delete an audio-only asset instead of removing its media")
        display_name = payload.get("display_name")
        if isinstance(display_name, str) and display_name.strip():
            asset["display_name"] = display_name.strip()[:120]
        elif not asset.get("display_name"):
            asset["display_name"] = str(asset.get("image_name") or "Local asset")
        if "profile_text" in payload:
            profile_text = str(payload.get("profile_text") or "").strip()[:2000]
            asset["profile_text"] = profile_text
            asset["profile_status"] = "ready" if profile_text else "missing"
            asset["profile_source"] = "human" if profile_text else "none"
        self._apply_memory_reference_fields(asset, payload)
        identity_ids = payload.get("identity_ids")
        if isinstance(identity_ids, list):
            asset["identity_ids"] = list(dict.fromkeys(
                str(value).strip()[:80]
                for value in identity_ids
                if isinstance(value, str) and value.strip()
            ))
        elif "reference_type" in payload or "reference_label" in payload:
            if asset.get("reference_type") == "character" and asset.get("reference_label"):
                asset["identity_ids"] = [asset["reference_label"]]
            elif asset.get("reference_type") != "character":
                asset["identity_ids"] = []
        elif image_payload is not None and asset.get("profile_source") != "human":
            # The configured Memory VLM may profile local image uploads. If no
            # VLM route exists, keep the asset usable by humans but invisible
            # to agent recommendation until a profile is entered manually.
            runner_options = getattr(self._memory_review_runner, "keywords", {})
            runner_options = runner_options if isinstance(runner_options, dict) else {}
            api_base = str(runner_options.get("api_base") or "").strip()
            api_key = str(runner_options.get("api_key") or "").strip()
            model = str(runner_options.get("vlm_model") or "").strip()
            local_image = str(asset.get("local_image_path") or "").strip()
            if api_base and api_key and model and local_image:
                try:
                    from nanobot.director.memory_selector import MemoryVlmSelector

                    generated = MemoryVlmSelector(
                        api_base=api_base,
                        api_key=api_key,
                        model=model,
                    ).profile_image(
                        image_path=Path(local_image),
                        display_name=str(asset.get("display_name") or ""),
                    )
                    asset["profile_text"] = generated["profile_text"]
                    asset["identity_ids"] = generated["identity_ids"]
                    asset["profile_status"] = "ready"
                    asset["profile_source"] = "vlm"
                except Exception as exc:
                    logger.warning(
                        "memory asset VLM profile failed work_id={} asset_id={} error={}",
                        work_id,
                        asset_id,
                        exc,
                    )
                    asset["profile_status"] = "error"
                    asset["profile_source"] = "none"
            else:
                asset["profile_status"] = "missing"
                asset["profile_source"] = "none"
        asset["updated_at"] = now
        if existing_index is None:
            assets.append(asset)
        else:
            assets[existing_index] = asset
        self._save_manual_memory_workspace(paths, assets)
        return work_id, self._build_workplace_payload(session_key)

    def _apply_workplace_shot_memory_asset_create(
        self,
        session_key: str,
        shot_id: int,
        payload: Any,
    ) -> tuple[str, dict[str, Any]]:
        """Save one user-chosen frame and optional audio clip from a shot."""
        if not isinstance(payload, dict):
            raise ValueError("asset is required")
        loaded = self._load_workplace_shot(session_key, shot_id)
        if loaded is None:
            raise ValueError(f"shot {shot_id} not found")
        work_id, _shot_path, _state, shot = loaded
        paths = self._workplace_paths(work_id)
        if paths is None:
            raise ValueError("workplace is unavailable")
        assets = self._load_manual_memory_workspace(paths)
        if len(assets) >= _MAX_MEMORY_WORKSPACE_ASSETS:
            raise ValueError("memory workspace is full")

        reference_type = str(payload.get("reference_type") or "").strip().lower()
        if reference_type not in _MEMORY_REFERENCE_TYPES:
            raise ValueError("reference_type is required")
        reference_label = str(payload.get("reference_label") or "").strip()[:80]
        profile_text = str(payload.get("profile_text") or "").strip()[:2000]
        try:
            timestamp_sec = float(payload.get("timestamp_sec"))
        except (TypeError, ValueError) as exc:
            raise ValueError("timestamp_sec is required") from exc
        if timestamp_sec < 0:
            raise ValueError("timestamp_sec must be non-negative")

        include_audio = payload.get("include_audio") is True
        audio_start_sec: float | None = None
        audio_end_sec: float | None = None
        if include_audio:
            try:
                audio_start_sec = float(payload.get("audio_start_sec"))
                audio_end_sec = float(payload.get("audio_end_sec"))
            except (TypeError, ValueError) as exc:
                raise ValueError("audio start and end are required") from exc
            if audio_start_sec < 0 or audio_end_sec <= audio_start_sec:
                raise ValueError("audio end must be after audio start")
            if audio_end_sec - audio_start_sec > 30:
                raise ValueError("memory audio clip cannot exceed 30 seconds")

        artifact = str(
            shot.get("artifact_url") or shot.get("artifact_path") or ""
        ).strip()
        if not artifact:
            raise ValueError(f"shot {shot_id} has no video artifact")
        video_path = (
            paths["work_dir"] / "memory" / "videos" / f"shot_{shot_id:03d}.mp4"
        )
        if not video_path.is_file():
            download_video(artifact, video_path)

        asset_id = uuid.uuid4().hex
        asset_dir = paths["work_dir"] / "memory" / "manual" / "assets" / asset_id
        image_path = asset_dir / "image.jpg"
        selected_time, frame_index = extract_video_frame(
            video_path, image_path, timestamp_sec
        )
        audio_path: Path | None = None
        if include_audio and audio_start_sec is not None and audio_end_sec is not None:
            audio_path = asset_dir / "audio.wav"
            audio_path.parent.mkdir(parents=True, exist_ok=True)
            result = subprocess.run(
                [
                    _resolve_media_binary("ffmpeg"),
                    "-loglevel", "error", "-y",
                    "-ss", f"{audio_start_sec:.6f}",
                    "-i", str(video_path),
                    "-t", f"{audio_end_sec - audio_start_sec:.6f}",
                    "-vn", "-acodec", "pcm_s16le", "-ar", "48000",
                    str(audio_path),
                ],
                capture_output=True,
                text=True,
            )
            if (
                result.returncode != 0
                or not audio_path.is_file()
                or audio_path.stat().st_size <= 44
            ):
                raise ValueError(
                    f"failed to extract memory audio: {result.stderr[:1000]}"
                )
            if audio_path.stat().st_size > _MAX_MEMORY_AUDIO_BYTES:
                raise ValueError("memory audio exceeds size limit")

        profile_status = "ready" if profile_text else "missing"
        profile_source = "human" if profile_text else "none"
        identities = (
            [reference_label]
            if reference_type == "character" and reference_label
            else []
        )
        if not profile_text:
            runner_options = getattr(self._memory_review_runner, "keywords", {})
            runner_options = runner_options if isinstance(runner_options, dict) else {}
            api_base = str(runner_options.get("api_base") or "").strip()
            api_key = str(runner_options.get("api_key") or "").strip()
            model = str(runner_options.get("vlm_model") or "").strip()
            if api_base and api_key and model:
                try:
                    from nanobot.director.memory_selector import MemoryVlmSelector

                    generated = MemoryVlmSelector(
                        api_base=api_base,
                        api_key=api_key,
                        model=model,
                    ).profile_image(
                        image_path=image_path,
                        display_name=reference_label or f"Shot {shot_id} reference",
                    )
                    profile_text = generated["profile_text"]
                    if reference_type == "character" and not identities:
                        identities = generated["identity_ids"]
                    profile_status = "ready"
                    profile_source = "vlm"
                except Exception as exc:
                    logger.warning(
                        "shot memory asset VLM profile failed work_id={} shot_id={} error={}",
                        work_id,
                        shot_id,
                        exc,
                    )
                    profile_status = "error"

        publisher = configured_file_publisher(
            work_id,
            storage=self._tools_config.file_storage,
            workspace=self._session_manager.workspace,
        )
        image_url = publisher(
            str(image_path), f"memory/manual/{asset_id}/image.jpg"
        )
        audio_url = (
            publisher(str(audio_path), f"memory/manual/{asset_id}/audio.wav")
            if audio_path is not None
            else None
        )
        local_image = resolve_local_asset_path(
            image_url,
            workspace=self._session_manager.workspace,
            config=self._tools_config.file_storage.local,
        )
        local_audio = (
            resolve_local_asset_path(
                audio_url,
                workspace=self._session_manager.workspace,
                config=self._tools_config.file_storage.local,
            )
            if audio_url is not None
            else None
        )
        now = datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
        asset: dict[str, Any] = {
            "asset_id": asset_id,
            "display_name": reference_label or f"Shot {shot_id} {reference_type}",
            "image_path": image_url,
            "image_name": f"shot_{shot_id:03d}_{selected_time:.2f}s.jpg",
            "source_shot_id": shot_id,
            "timestamp_sec": selected_time,
            "frame_index": frame_index,
            "reference_type": reference_type,
            "reference_label": reference_label,
            "profile_text": profile_text,
            "profile_status": profile_status,
            "profile_source": profile_source,
            "identity_ids": identities,
            "created_at": now,
            "updated_at": now,
        }
        if local_image is not None:
            asset["local_image_path"] = str(local_image)
        if audio_url is not None:
            asset["audio_path"] = audio_url
            asset["audio_name"] = (
                f"shot_{shot_id:03d}_{audio_start_sec:.2f}-{audio_end_sec:.2f}s.wav"
            )
            asset["audio_start_sec"] = audio_start_sec
            asset["audio_end_sec"] = audio_end_sec
            asset["audio_source_shot_id"] = shot_id
        if local_audio is not None:
            asset["local_audio_path"] = str(local_audio)
        assets.append(asset)
        self._save_manual_memory_workspace(paths, assets)
        return work_id, self._build_workplace_payload(session_key)

    @staticmethod
    def _apply_memory_reference_fields(
        target: dict[str, Any], payload: dict[str, Any]
    ) -> None:
        if "reference_type" in payload:
            reference_type = str(payload.get("reference_type") or "").strip().lower()
            if reference_type and reference_type not in _MEMORY_REFERENCE_TYPES:
                raise ValueError("invalid memory reference type")
            if reference_type:
                target["reference_type"] = reference_type
            else:
                target.pop("reference_type", None)
        if "reference_label" in payload:
            reference_label = str(payload.get("reference_label") or "").strip()[:80]
            if reference_label:
                target["reference_label"] = reference_label
            else:
                target.pop("reference_label", None)

    def _apply_workplace_memory_asset_delete(
        self,
        session_key: str,
        asset_id: str,
    ) -> tuple[str, dict[str, Any]]:
        work_id = self._resolve_work_id_for_session(session_key)
        if not work_id:
            raise ValueError("workplace is not initialized")
        paths = self._workplace_paths(work_id)
        if paths is None:
            raise ValueError("workplace is unavailable")
        assets = self._load_manual_memory_workspace(paths)
        kept = [item for item in assets if str(item.get("asset_id") or "") != asset_id]
        if len(kept) == len(assets):
            raise ValueError("memory asset not found")
        for shot_path in paths["shots"].glob("shot_*.json"):
            shot = self._read_json_file(shot_path, {})
            if not isinstance(shot, dict):
                continue
            for slot in shot.get("approved_memory_slots", []):
                metadata = slot.get("metadata") if isinstance(slot, dict) else None
                if (
                    isinstance(metadata, dict)
                    and asset_id in {
                        metadata.get("workspace_asset_id"),
                        metadata.get("audio_workspace_asset_id"),
                    }
                ):
                    raise ValueError("remove this asset from shot slots before deleting it")
        # Cached media stays on disk because an already-applied shot may still reference it.
        self._save_manual_memory_workspace(paths, kept)
        return work_id, self._build_workplace_payload(session_key)

    def _apply_workplace_shot_memory_slots_save(
        self,
        session_key: str,
        shot_id: int,
        refs: Any,
    ) -> tuple[str, dict[str, Any]]:
        if not isinstance(refs, list):
            raise ValueError("slots must be a list")
        if len(refs) > _MAX_MEMORY_SLOTS:
            raise ValueError("a shot supports at most 7 memory slots")
        loaded = self._load_workplace_shot(session_key, shot_id)
        if loaded is None:
            raise ValueError(f"shot {shot_id} not found")
        work_id, shot_path, state, shot = loaded
        if (
            shot_id > 1
            and str(state.get("stage") or "") == "awaiting_memory_build"
            and not shot.get("memory_slots_user_configured")
            and str(shot.get("memory_recommendation_source") or "") != "agent"
        ):
            raise ValueError("Memory recommendation is still in progress")
        paths = self._workplace_paths(work_id)
        if paths is None:
            raise ValueError("workplace is unavailable")
        profile_overrides = self._load_memory_asset_profiles(paths)
        automatic = {
            asset_id: {
                **raw,
                **(
                    profile_overrides.get(asset_id)
                    if isinstance(profile_overrides.get(asset_id), dict)
                    else {}
                ),
            }
            for asset_id, raw in self._automatic_memory_workspace_records(paths)
        }
        manual = {
            str(item.get("asset_id") or ""): item
            for item in self._load_manual_memory_workspace(paths)
            if item.get("asset_id")
        }
        slots: list[dict[str, Any]] = []
        display_slots: list[dict[str, Any]] = []
        saved_refs: list[dict[str, Any]] = []
        seen: set[str] = set()
        for ref in refs:
            if not isinstance(ref, dict):
                raise ValueError("memory slot is malformed")
            source = str(ref.get("source") or "")
            asset_id = str(
                ref.get("image_asset_id") or ref.get("asset_id") or ""
            ).strip()
            audio_asset_id = str(ref.get("audio_asset_id") or "").strip() or None
            if not source and asset_id:
                source = "automatic" if asset_id in automatic else "local"
            if source not in {"automatic", "local"} or not asset_id:
                raise ValueError("memory slot reference is invalid")
            if asset_id in seen:
                raise ValueError("memory slots cannot contain duplicates")
            seen.add(asset_id)
            raw = automatic.get(asset_id) if source == "automatic" else manual.get(asset_id)
            if raw is None:
                raise ValueError(f"memory asset {asset_id} not found")
            slot = self._memory_workspace_slot(
                raw, asset_id=asset_id, source=source, display=False
            )
            display_slot = self._memory_workspace_slot(
                raw, asset_id=asset_id, source=source, display=True
            )
            if audio_asset_id:
                audio_source = (
                    "automatic" if audio_asset_id in automatic
                    else "local" if audio_asset_id in manual
                    else ""
                )
                audio_raw = (
                    automatic.get(audio_asset_id)
                    if audio_source == "automatic"
                    else manual.get(audio_asset_id)
                )
                if audio_raw is None:
                    raise ValueError(f"memory audio asset {audio_asset_id} not found")
                audio = self._memory_slot_locator(audio_raw, "audio", display=False)
                display_audio = self._memory_slot_locator(audio_raw, "audio", display=True)
                if not audio:
                    raise ValueError(f"memory asset {audio_asset_id} has no audio")
                slot["audio_url"] = audio
                slot.pop("audio_mode", None)
                display_slot["audio_url"] = display_audio or audio
                display_slot.pop("audio_mode", None)
                slot["metadata"]["audio_workspace_asset_id"] = audio_asset_id
                display_slot["metadata"]["audio_workspace_asset_id"] = audio_asset_id
            slots.append(slot)
            display_slots.append(display_slot)
            saved_refs.append({
                "image_asset_id": asset_id,
                **({"audio_asset_id": audio_asset_id} if audio_asset_id else {}),
            })
        shot["approved_memory_slots"] = slots
        shot["approved_memory_display_slots"] = display_slots
        shot["approved_memory_slot_refs"] = saved_refs
        shot["memory_slots_user_configured"] = True
        shot["memory_slots_applied_at"] = (
            datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
        )
        self._save_workplace_shot(shot_path, shot)
        return work_id, self._build_workplace_payload(session_key)

    def _project_generation_memory(self, raw: Any) -> dict[str, Any] | None:
        """Project one approved R2V Memory slot without exposing local paths."""
        if not isinstance(raw, dict):
            return None
        metadata = raw.get("metadata")
        metadata = metadata if isinstance(metadata, dict) else {}
        memory_id = str(metadata.get("id") or raw.get("memory_id") or "").strip()
        if not memory_id:
            return None

        def public_ref(value: Any) -> dict[str, str] | None:
            if not isinstance(value, str) or not value.strip():
                return None
            locator = value.strip()
            if locator.startswith("/api/media/"):
                return {"url": locator}
            return self._public_media_entry(locator)

        image = public_ref(raw.get("image_url") or raw.get("image_path"))
        if image is None:
            return None
        projected: dict[str, Any] = {
            "id": memory_id,
            "display_name": str(
                metadata.get("display_name") or self._memory_display_name(memory_id)
            ),
            "image": image,
            "metadata": {
                key: metadata[key]
                for key in (
                    "source",
                    "visual_status",
                    "source_shot_id",
                    "frame_index",
                    "timestamp_sec",
                    "confidence",
                    "audio_source_shot_id",
                    "audio_workspace_asset_id",
                    "reference_type",
                    "reference_label",
                    "identity_ids",
                    "profile_text",
                )
                if key in metadata
            },
        }
        workspace_asset_id = metadata.get("workspace_asset_id")
        if isinstance(workspace_asset_id, str) and workspace_asset_id.strip():
            projected["workspace_asset_id"] = workspace_asset_id.strip()
        audio = public_ref(raw.get("audio_url") or raw.get("audio_path"))
        if audio is not None:
            projected["audio"] = audio
        return projected

    def _build_workplace_payload(self, session_key: str) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "session_key": session_key,
            "work_id": None,
            "story_md": "",
            "story_empty": True,
            "stage": None,
            "goal": {},
            "final_output_path": None,
            "final_output_url": None,
            "final_video": None,
            "memory_bank": [],
            "memory_workspace_assets": [],
            "shots": [],
            "updated_at": None,
        }
        work_id = self._resolve_work_id_for_session(session_key)
        if not work_id:
            return self._attach_session_reference_and_auto(session_key, payload)
        paths = self._workplace_paths(work_id)
        if paths is None:
            return payload
        state = self._read_json_file(paths["state"], {})
        if not isinstance(state, dict):
            state = {}
        self._reconcile_workplace_state_shots(work_id, state)
        story_profile = self._load_workplace_story_profile(work_id)
        beats = story_profile.get("beats") if isinstance(story_profile.get("beats"), list) else []
        beat_count = len(beats)
        if beat_count > 0:
            self._prune_workplace_shots_beyond_count(work_id, beat_count, state)
            self._reconcile_workplace_state_shots(work_id, state)
        stage = str(state.get("stage") or "")
        if stage == "shot_generating":
            self._ensure_workplace_shot_references(work_id)
        story_md = ""
        try:
            if paths["story"].exists():
                story_md = paths["story"].read_text(encoding="utf-8")
        except OSError:
            story_md = ""
        goal = state.get("goal") if isinstance(state.get("goal"), dict) else {}
        goal_duration_s = float(goal.get("shot_duration_sec") or 0)
        final_locator = None
        for candidate in (state.get("final_output_url"), state.get("final_output_path")):
            if isinstance(candidate, str) and candidate.strip():
                final_locator = candidate.strip()
                break
        final_video = self._public_media_entry(final_locator) if isinstance(final_locator, str) else None
        if (
            final_video is None
            and isinstance(final_locator, str)
            and final_locator
            and not urlparse(final_locator).scheme
        ):
            relative_candidate = Path(final_locator)
            if not relative_candidate.is_absolute():
                for base_dir in (paths["work_dir"], paths["work_dir"] / "outputs"):
                    candidate_entry = self._public_media_entry(str(base_dir / relative_candidate))
                    if candidate_entry is not None:
                        final_video = candidate_entry
                        break
        shot_rows: list[dict[str, Any]] = []
        cursor_s = 0.0
        for shot_path in sorted(paths["shots"].glob("shot_*.json")):
            shot = self._read_json_file(shot_path, {})
            if not isinstance(shot, dict):
                continue
            try:
                shot_id = int(shot.get("shot_id"))
            except (TypeError, ValueError):
                continue
            duration_s = self._estimate_shot_duration_seconds(shot, goal_duration_s)
            media_entry = None
            for candidate in (
                shot.get("artifact_url"),
                shot.get("artifact_path"),
                shot.get("remote_result", {}).get("video_path")
                if isinstance(shot.get("remote_result"), dict)
                else None,
            ):
                if not isinstance(candidate, str) or not candidate.strip():
                    continue
                media_entry = self._public_media_entry(candidate)
                if media_entry is not None:
                    break
            echo_data = shot.get("echo") if isinstance(shot.get("echo"), dict) else {}
            shot_updated = self._parse_iso_timestamp(shot.get("updated_at"))
            timeline_start = cursor_s
            timeline_end = cursor_s + duration_s
            cursor_s = timeline_end
            planned_reference_shot_ids = self._planned_reference_shot_ids(shot)
            projected_memory_review = self._project_memory_review(
                shot.get("memory_review")
            )
            if projected_memory_review is not None and media_entry is not None:
                projected_memory_review["source_video"] = media_entry
            projected_slots = [
                projected
                for item in (
                    shot.get("approved_memory_display_slots")
                    or shot.get("approved_memory_slots", [])
                )
                if (projected := self._project_generation_memory(item)) is not None
            ]
            projected_recommended_slots = [
                projected
                for item in (
                    shot.get("recommended_memory_display_slots")
                    or shot.get("recommended_memory_slots", [])
                )
                if (projected := self._project_generation_memory(item)) is not None
            ]
            recommendation_refs = shot.get("recommended_memory_slot_refs")
            if not isinstance(recommendation_refs, list):
                recommendation_refs = [
                    {
                        "image_asset_id": item["workspace_asset_id"],
                        "reason": "Matched from the accepted shot's visual continuity profile.",
                    }
                    for item in projected_recommended_slots
                    if isinstance(item.get("workspace_asset_id"), str)
                ]
            shot_rows.append(
                {
                    "shot_id": shot_id,
                    "shot_key": shot.get("shot_key") or f"shot_{shot_id:03d}",
                    "status": str(shot.get("status") or "planned"),
                    "summary": shot.get("summary") or "",
                    "caption": shot.get("caption") or "",
                    "num_frames": shot.get("num_frames"),
                    "cut": bool(shot.get("cut", True)),
                    "video": media_entry,
                    "has_video": media_entry is not None,
                    "last_review": shot.get("last_review"),
                    "review_notes": shot.get("review_notes") or "",
                    "generation_error": shot.get("generation_error") or "",
                    "memory_review": projected_memory_review,
                    "generation_memories": projected_slots,
                    "memory_slots": projected_slots,
                    "memory_slots_configured": bool(shot.get("memory_slots_user_configured")),
                    "approved_memory_slot_refs": list(
                        shot.get("approved_memory_slot_refs") or []
                    ),
                    "recommended_memory_slots": projected_recommended_slots,
                    "recommended_memory_slot_refs": recommendation_refs,
                    "memory_recommendation_source": shot.get("memory_recommendation_source"),
                    "version_id": echo_data.get("version_id") or "",
                    "echo_status": echo_data.get("status") or "",
                    "updated_at": shot.get("updated_at"),
                    "planned_reference_shot_ids": planned_reference_shot_ids,
                    "reference_shot_ids": planned_reference_shot_ids,
                    "reference_selection_note": str(shot.get("reference_selection_note") or ""),
                    "references_planned": "planned_reference_shot_ids" in shot,
                    "continuous_enabled": bool(shot.get("continuous_enabled", False)),
                    "tail_frame_url": shot.get("tail_frame_url") or "",
                    "timeline": {
                        "start_seconds": timeline_start,
                        "end_seconds": timeline_end,
                        "duration_seconds": duration_s,
                        "label": (
                            f"{self._format_timeline_time(timeline_start)} - "
                            f"{self._format_timeline_time(timeline_end)}"
                        ),
                    },
                    "has_actions": media_entry is not None,
                    "accepted": str(shot.get("status") or "") in {"review_pass", "approved"},
                }
            )
            if shot_updated is not None:
                current_updated = self._parse_iso_timestamp(payload["updated_at"])
                if current_updated is None or shot_updated > current_updated:
                    payload["updated_at"] = shot.get("updated_at")
        beats_editable = self._workplace_beats_editable(work_id, state)
        self._sync_story_confirmed_when_shot_specs_ready(
            work_id,
            state,
            beat_count=beat_count,
        )
        payload.update(
            {
                "work_id": work_id,
                "story_md": story_md,
                "story_empty": story_md.strip() == "",
                "story_profile": story_profile,
                "story_editable": stage == "story_discussion",
                "beats_editable": beats_editable,
                "story_confirmed": bool(state.get("story_confirmed")),
                "shot_prompts_ready": self._workplace_shot_prompts_ready(
                    work_id,
                    state,
                    beat_count=beat_count,
                ),
                "shot_prompts_progress": self._workplace_shot_prompts_progress(
                    work_id,
                    beat_count=beat_count,
                ),
                "references_ready": self._workplace_references_ready(work_id, beat_count=beat_count),
                "shot_generating_started_at": state.get("shot_generating_started_at"),
                "stage": state.get("stage"),
                "goal": goal,
                "final_output_path": state.get("final_output_path"),
                "final_output_url": state.get("final_output_url"),
                "final_video": final_video
                or (
                    shot_rows[0].get("video")
                    if len(shot_rows) == 1 and isinstance(shot_rows[0].get("video"), dict)
                    else None
                ),
                "reference_image": state.get("reference_image"),
                "reference_image_locked": is_reference_image_locked(state),
                "auto_generate": bool(state.get("auto_generate"))
                or self._session_auto_generate_flag(session_key),
                "generation_error": state.get("generation_error") or None,
                "memory_bank": self._project_memory_bank(paths),
                "memory_workspace_assets": self._project_memory_workspace_assets(paths),
                "shots": sorted(shot_rows, key=lambda item: int(item["shot_id"])),
                "updated_at": payload["updated_at"] or state.get("updated_at"),
                **self._echo_tracking_payload(state),
            }
        )
        payload["progress"] = self._workplace_progress(state, payload)
        shot_failed = any(str(row.get("status") or "") == "error" for row in shot_rows)
        if payload.get("auto_generate") and shot_failed:
            payload["stage"] = "failed"
        elif (
            payload.get("auto_generate")
            and payload.get("progress") == "04"
            and str(state.get("stage") or "") not in {"done", "failed"}
        ):
            # Frontend WorkplacePanel still keys content off `stage`, not `progress`.
            # Present merging so 04 shows a loading ComposePanel instead of 03 FramesPanel.
            payload["stage"] = "merging"
        return payload

    def _session_auto_generate_flag(self, session_key: str) -> bool:
        if self._session_manager is None:
            return False
        session = self._session_manager.get_or_create(session_key)
        metadata = session.metadata if isinstance(session.metadata, dict) else {}
        return get_auto_generate(metadata)

    def _attach_session_reference_and_auto(
        self,
        session_key: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        if self._session_manager is None:
            payload.setdefault("reference_image", None)
            payload.setdefault("reference_image_locked", False)
            payload.setdefault("auto_generate", False)
            payload.setdefault("progress", "01")
            return payload
        session = self._session_manager.get_or_create(session_key)
        metadata = session.metadata if isinstance(session.metadata, dict) else {}
        payload["reference_image"] = normalize_reference_image(metadata.get("reference_image"))
        payload["reference_image_locked"] = is_reference_image_locked(metadata)
        payload["auto_generate"] = get_auto_generate(metadata)
        payload["progress"] = "01"
        return payload

    @staticmethod
    def _workplace_progress(state: dict[str, Any], payload: dict[str, Any]) -> str:
        if payload.get("final_output_url") or (
            payload.get("final_video") and str(state.get("stage") or "") == "done"
        ):
            return "done"
        if state.get("final_output_url") or state.get("final_output_path"):
            return "done"
        auto = bool(state.get("auto_generate")) or bool(payload.get("auto_generate"))
        stage = str(state.get("stage") or "")
        if stage == "done":
            return "done"
        # 一键成片全程停在 04，避免 regenerate 落到 02/03 闪成逐镜打磨。
        if auto:
            return "04"
        if stage == "merging":
            return "04"
        if not state.get("story_confirmed"):
            return "01"
        if stage in {
            "shot_generating",
            "shot_reviewing",
            "shot_revising",
            "awaiting_memory_review",
            "failed",
        }:
            return "03"
        # 02 only after the user clicks 「下一步」(confirm_story → shot_planning).
        # Locking shot_count in chat must stay on 01.
        if stage == "shot_planning":
            return "02"
        return "01"

    def _workplace_first_frame_url(self, work_id: str) -> str | None:
        state = self._load_workplace_state(work_id)
        ref = normalize_reference_image(state.get("reference_image"))
        if not ref:
            return None
        url = ref.get("url")
        return url.strip() if isinstance(url, str) and url.strip() else None

    def _load_workplace_shot(self, session_key: str, shot_id: int) -> tuple[str, Path, dict[str, Any], dict[str, Any]] | None:
        work_id = self._resolve_work_id_for_session(session_key)
        if not work_id:
            return None
        paths = self._workplace_paths(work_id)
        if paths is None:
            return None
        state = self._read_json_file(paths["state"], {})
        if not isinstance(state, dict):
            state = {}
        shot_path = paths["shots"] / f"shot_{shot_id:03d}.json"
        shot = self._read_json_file(shot_path, {})
        if not isinstance(shot, dict) or not shot:
            return None
        return work_id, shot_path, state, shot

    @staticmethod
    def _planned_reference_shot_ids(shot: dict[str, Any]) -> list[int]:
        raw = shot.get("planned_reference_shot_ids")
        if isinstance(raw, list):
            out: list[int] = []
            for item in raw:
                try:
                    out.append(int(item))
                except (TypeError, ValueError):
                    continue
            return sorted(set(out))
        return []

    @staticmethod
    def _shot_has_generated_media(shot: dict[str, Any]) -> bool:
        status = str(shot.get("status") or "")
        if status in {"generated", "review_pass", "approved"}:
            return True
        for key in ("artifact_url", "artifact_path"):
            value = shot.get(key)
            if isinstance(value, str) and value.strip():
                return True
        return False

    @staticmethod
    def _format_reference_dependency_error(shot_id: int, missing: list[int]) -> str:
        refs = "、".join(str(item) for item in sorted(missing))
        return f"镜头{shot_id}依赖镜头{refs}，请先生成镜头{refs}，再来操作"

    def _previous_shot_approval_error(self, work_id: str, shot_id: int) -> str | None:
        """Require the immediately preceding Shot video and Memory before advancing."""
        if shot_id <= 1 or not self._memory_review_workflow_enabled():
            return None
        previous_rows = [
            (candidate_id, candidate)
            for candidate_id, candidate in self._iter_workplace_shots(work_id)
            if candidate_id < shot_id
        ]
        previous_id = shot_id - 1
        previous: dict[str, Any] | None = None
        if previous_rows:
            previous_id, previous = max(previous_rows, key=lambda item: item[0])
        video_accepted = bool(previous) and str(previous.get("status") or "") == "approved"
        review = previous.get("memory_review") if isinstance(previous, dict) else None
        memory_accepted = isinstance(review, dict) and review.get("status") == "approved"
        if video_accepted and memory_accepted:
            return None
        return f"请先接受镜头{previous_id}并确认其 Memory，再生成镜头{shot_id}"

    def _workplace_references_ready(self, work_id: str, *, beat_count: int) -> bool:
        paths = self._workplace_paths(work_id)
        if paths is None:
            return False
        shot_paths = sorted(paths["shots"].glob("shot_*.json"))
        if not shot_paths:
            return False
        if beat_count > 0 and len(shot_paths) != beat_count:
            return False
        for shot_path in shot_paths:
            shot = self._read_json_file(shot_path, {})
            if not isinstance(shot, dict) or "planned_reference_shot_ids" not in shot:
                return False
        return True

    def _missing_reference_generations(
        self,
        work_id: str,
        reference_shot_ids: list[int],
    ) -> list[int]:
        paths = self._workplace_paths(work_id)
        if paths is None:
            return list(reference_shot_ids)
        missing: list[int] = []
        for ref_id in reference_shot_ids:
            shot_path = paths["shots"] / f"shot_{ref_id:03d}.json"
            shot = self._read_json_file(shot_path, {})
            if not isinstance(shot, dict) or not self._shot_has_generated_media(shot):
                missing.append(ref_id)
        return missing

    def _director_tool_kwargs(self) -> dict[str, Any]:
        if self._session_manager is None:
            raise RuntimeError("session manager unavailable")
        return {
            "workspace": self._session_manager.workspace,
            "tools_config": self._tools_config,
        }

    def _ensure_echo_admission(self, *, operation: str = "generate_echo_shot") -> None:
        """Reject new generation work when the Echo backend reports overload."""
        EchoAdmissionController.from_tools_config(self._tools_config).ensure_allowed(
            operation=operation,
        )

    def _director_generate_tool(self) -> GenerateEchoShotTool:
        return GenerateEchoShotTool(**self._director_tool_kwargs())

    def _director_merge_tool(self) -> MergeShotTool:
        return MergeShotTool(**self._director_tool_kwargs())

    def _director_set_references_tool(self) -> SetShotReferencesTool:
        return SetShotReferencesTool(**self._director_tool_kwargs())

    # ── tail-frame extraction pipeline ──────────────────────────────────────

    @staticmethod
    def _download_shot_video(video_url: str, target: Path) -> None:
        """Download a shot video from *video_url* to a local *target* path."""
        target.parent.mkdir(parents=True, exist_ok=True)
        source = Path(video_url)
        if source.is_file():
            shutil.copyfile(source, target)
            return
        req = urllib.request.Request(
            video_url,
            headers={"Accept": "video/mp4,*/*", "User-Agent": "EchoDirector/1.0"},
        )
        with urllib.request.urlopen(req, timeout=120) as response:
            with target.open("wb") as output:
                shutil.copyfileobj(response, output)
        if not target.is_file() or target.stat().st_size <= 0:
            raise RuntimeError(f"downloaded empty video: {video_url}")

    @staticmethod
    def _extract_tail_frame(video_path: Path, output_path: Path) -> bool:
        """Extract the last frame of *video_path* as a PNG using ffmpeg.

        Returns ``True`` when the output file was created successfully.
        """
        ffmpeg = _resolve_media_binary("ffmpeg")
        cmd = [
            ffmpeg,
            "-sseof", "-1",
            "-i", str(video_path),
            "-update", "1",
            "-q:v", "1",
            str(output_path),
            "-y",
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, timeout=60)
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as exc:
            logger.error("ffmpeg tail-frame extraction failed: {}", exc)
            return False
        return result.returncode == 0 and output_path.is_file()

    def _publish_tail_frame(
        self, image_path: Path, work_id: str, shot_id: int
    ) -> str:
        """Persist a tail-frame PNG and return its local asset URL."""
        publisher = configured_file_publisher(
            work_id,
            storage=self._tools_config.file_storage,
            workspace=self._session_manager.workspace,
        )
        name = f"tail_frames/shot_{shot_id:03d}.png"
        return publisher(str(image_path), name)

    def _extract_and_publish_tail_frame(
        self, work_id: str, shot_id: int, video_url: str
    ) -> str | None:
        """Complete tail-frame pipeline: download → ffmpeg → file storage → URL.

        Returns the local asset URL of the extracted tail frame, or ``None`` on failure.
        """
        tmp_dir = Path(tempfile.mkdtemp(prefix="tail_frame_"))
        try:
            video_path = tmp_dir / "input.mp4"
            self._download_shot_video(video_url, video_path)

            frame_path = tmp_dir / "tail.png"
            if not self._extract_tail_frame(video_path, frame_path):
                logger.error(
                    "ffmpeg failed to extract tail frame for shot {} in work {}",
                    shot_id, work_id,
                )
                return None

            asset_url = self._publish_tail_frame(frame_path, work_id, shot_id)
            logger.info(
                "tail frame uploaded shot_id={} work_id={} url={}",
                shot_id, work_id, asset_url,
            )
            return asset_url
        except Exception:
            logger.opt(exception=True).error(
                "tail-frame pipeline failed shot_id={} work_id={}",
                shot_id, work_id,
            )
            return None
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    # ── reference helpers ───────────────────────────────────────────────────

    @staticmethod
    def _default_planned_reference_shot_ids(shot_id: int, *, cut: bool) -> list[int]:
        if shot_id <= 1:
            return []
        if not cut:
            return [shot_id - 1]
        return [shot_id - 1]

    def _ensure_workplace_shot_references(self, work_id: str) -> int:
        """Fill missing reference plans so the UI is not blocked waiting for the agent."""
        paths = self._workplace_paths(work_id)
        if paths is None:
            return 0
        tool = self._director_set_references_tool()
        updated = 0
        for shot_path in sorted(paths["shots"].glob("shot_*.json")):
            shot = self._read_json_file(shot_path, {})
            if not isinstance(shot, dict) or "planned_reference_shot_ids" in shot:
                continue
            if not self._shot_has_caption(shot):
                continue
            try:
                shot_id = int(shot.get("shot_id"))
            except (TypeError, ValueError):
                stem = shot_path.stem
                if not stem.startswith("shot_"):
                    continue
                try:
                    shot_id = int(stem.removeprefix("shot_"))
                except ValueError:
                    continue
            if shot_id <= 0:
                continue
            cut = bool(shot.get("cut", True))
            refs = self._default_planned_reference_shot_ids(shot_id, cut=cut)
            note = (
                "系统默认：首镜无参考"
                if shot_id <= 1
                else "系统默认：连续镜头参考上一镜"
            )
            tool.apply_set_references(work_id, shot_id, refs, selection_note=note)
            updated += 1
        return updated

    def _apply_workplace_shot_duration(
        self,
        session_key: str,
        shot_id: int,
        duration_sec: int,
    ) -> tuple[str, dict[str, Any]]:
        loaded = self._load_workplace_shot(session_key, shot_id)
        if loaded is None:
            raise ValueError("shot not found")
        work_id, shot_path, state, shot = loaded
        clamped = self._clamp_shot_duration_sec(duration_sec)
        sync_shot_echo_duration(shot, clamped)
        self._save_workplace_shot(shot_path, shot)
        self._sync_state_shot_row(state, shot)
        self._save_workplace_state(work_id, state)
        return work_id, self._build_workplace_payload(session_key)

    def _mark_workplace_shot_queued(
        self,
        work_id: str,
        shot_path: Path,
        shot: dict[str, Any],
    ) -> None:
        """Mark a workplace shot as queued before async Echo submission."""
        shot["status"] = "queued"
        shot.pop("generation_error", None)
        self._save_workplace_shot(shot_path, shot)
        state = self._load_workplace_state(work_id)
        state["stage"] = "shot_generating"
        self._save_workplace_state(work_id, state)

    def _validate_workplace_shot_generate(
        self,
        session_key: str,
        shot_id: int,
        *,
        reference_image_url: str | None = None,
        reference_image_name: str | None = None,
        reference_image_width: int | None = None,
        reference_image_height: int | None = None,
    ) -> tuple[str, Path, dict[str, Any], dict[str, Any], list[int], str | None, bool]:
        """Validate a workplace shot generate request and return submit context."""
        self._ensure_echo_admission(operation="generate_echo_shot")
        loaded = self._load_workplace_shot(session_key, shot_id)
        if loaded is None:
            raise ValueError("shot not found")
        work_id, shot_path, _state, shot = loaded
        use_continuous = (
            bool(shot.get("continuous_enabled"))
            and shot_id > 1
            and not self._workplace_auto_generate_active(session_key)
        )
        self._lock_workplace_video_size(session_key, work_id)
        if reference_image_url:
            logger.info(
                "workplace shot generate: ignoring request-body reference_image_* "
                "session_key={} work_id={} shot_id={} url={}",
                session_key,
                work_id,
                shot_id,
                reference_image_url,
            )
        status = str(shot.get("status") or "")
        if status == "queued":
            raise ValueError(f"shot {shot_id} generation already in progress")
        if status in {"generated", "review_pass", "approved"}:
            raise ValueError(f"shot {shot_id} is already generated")
        approval_error = self._previous_shot_approval_error(work_id, shot_id)
        if approval_error:
            raise ValueError(approval_error)
        self._ensure_workplace_shot_references(work_id)
        if "planned_reference_shot_ids" not in shot:
            shot = self._read_json_file(shot_path, {})
        if "planned_reference_shot_ids" not in shot:
            raise ValueError("reference plan is not ready; complete the previous workflow step first")
        reference_shot_ids = self._planned_reference_shot_ids(shot)
        missing = self._missing_reference_generations(work_id, reference_shot_ids)
        if missing:
            raise ValueError(self._format_reference_dependency_error(shot_id, missing))
        selection_note = shot.get("reference_selection_note")
        note = selection_note if isinstance(selection_note, str) else None
        return work_id, shot_path, shot, _state, reference_shot_ids, note, use_continuous

    def _submit_workplace_shot_generate(
        self,
        session_key: str,
        work_id: str,
        shot_id: int,
        shot: dict[str, Any],
        *,
        reference_shot_ids: list[int],
        selection_note: str | None,
        use_continuous: bool,
        reference_image_url: str | None = None,
        i2v_prompt: str | None = None,
    ) -> dict[str, Any]:
        """Submit a validated workplace shot to Echo."""
        generate_tool = self._director_generate_tool()
        generate_tool.set_context(
            "websocket",
            webui_wire_chat_id(session_key) or "direct",
            effective_key=session_key,
        )
        note = selection_note

        if use_continuous:
            previous_shot_id = shot_id - 1
            logger.info(
                "generate shot: shot_id={} continuous_enabled=True, "
                "using I2V (tail frame from shot {})",
                shot_id,
                previous_shot_id,
            )
            prev_loaded = self._load_workplace_shot(session_key, previous_shot_id)
            if prev_loaded is None:
                raise ValueError(f"previous shot {previous_shot_id} not found")
            _prev_work_id, prev_shot_path, _prev_state, prev_shot = prev_loaded
            prev_status = str(prev_shot.get("status") or "")
            if prev_status not in {"generated", "review_pass", "approved"}:
                raise ValueError(
                    f"previous shot {previous_shot_id} must be generated first "
                    f"(current: {prev_status})"
                )
            video_url = prev_shot.get("artifact_url") or (
                prev_shot.get("echo") or {}
            ).get("result_url")
            if not video_url:
                raise ValueError(
                    f"previous shot {previous_shot_id} has no artifact_url; "
                    f"cannot extract tail frame for continuous generation"
                )
            logger.info(
                "generate shot: shot_id={} extracting tail frame from "
                "previous shot {} video_url={}",
                shot_id,
                previous_shot_id,
                video_url,
            )
            condition_image_url = self._extract_and_publish_tail_frame(
                work_id,
                previous_shot_id,
                video_url,
            )
            if not condition_image_url:
                raise ValueError(
                    f"failed to extract tail frame from shot {previous_shot_id}"
                )
            logger.info(
                "generate shot: shot_id={} tail frame ready, "
                "condition_image_url={}",
                shot_id,
                condition_image_url,
            )
            prev_shot["tail_frame_url"] = condition_image_url
            self._save_workplace_shot(prev_shot_path, prev_shot)
            if previous_shot_id not in reference_shot_ids:
                reference_shot_ids = sorted(
                    set(reference_shot_ids) | {previous_shot_id}
                )
            job = generate_tool.apply_generate_continuous(
                work_id,
                shot_id,
                condition_image_url,
                reference_shot_ids,
                selection_note=note,
                i2v_prompt=i2v_prompt,
            )
        elif shot_id == 1 and reference_image_url:
            logger.info(
                "generate shot: shot_id=1 using user first-frame reference "
                "image (R2V with condition_img) url={}",
                reference_image_url,
            )
            job = generate_tool.apply_generate(
                work_id,
                shot_id,
                reference_shot_ids,
                selection_note=(
                    note or "Director R2V with user first-frame reference"
                ),
                condition_image_url=reference_image_url,
                i2v_prompt=i2v_prompt,
            )
        else:
            logger.info(
                "generate shot: shot_id={} continuous_enabled={}, "
                "using T2V (no tail frame)",
                shot_id,
                shot.get("continuous_enabled", False),
            )
            job = generate_tool.apply_generate(
                work_id,
                shot_id,
                reference_shot_ids,
                selection_note=note,
            )
        version_id = (
            job.get("remote", {}).get("version_id")
            if isinstance(job.get("remote"), dict)
            else None
        )
        logger.info(
            "workplace shot generate submitted work_id={} shot_id={} version_id={}",
            work_id,
            shot_id,
            version_id,
        )
        return job

    def _prepare_workplace_shot_generate_async(
        self,
        session_key: str,
        shot_id: int,
        *,
        reference_image_url: str | None = None,
        reference_image_name: str | None = None,
        reference_image_width: int | None = None,
        reference_image_height: int | None = None,
    ) -> tuple[str, dict[str, Any]]:
        """Validate shot 1 R2V generation and queue it for async re-caption + submit."""
        work_id, shot_path, shot, _state, _reference_shot_ids, _note, _use_continuous = (
            self._validate_workplace_shot_generate(
                session_key,
                shot_id,
                reference_image_url=reference_image_url,
                reference_image_name=reference_image_name,
                reference_image_width=reference_image_width,
                reference_image_height=reference_image_height,
            )
        )
        self._mark_workplace_shot_queued(work_id, shot_path, shot)
        workplace = self._build_workplace_payload(session_key)
        return work_id, workplace

    async def _complete_workplace_shot_generate_with_reference(
        self,
        session_key: str,
        *,
        work_id: str,
        shot_id: int,
        reference_image_url: str,
    ) -> None:
        """Background: PE I2V rewrite → Echo submit → workplace push."""
        try:
            if self._provider is None or not self._model:
                raise RuntimeError("re-caption model unavailable")
            generate_tool = self._director_generate_tool()
            shot = generate_tool._load_shot(work_id, shot_id)
            caption = str(shot.get("caption") or "").strip()
            if not caption:
                raise ValueError(f"Shot {shot_id} has no caption yet.")
            story_profile = self._load_workplace_story_profile(work_id)
            rewritten_prompt = await self._rewrite_i2v_prompt_with_image(
                caption,
                reference_image_url,
                story_profile,
                is_first_frame=True,
                enforce_first_frame_continuity=True,
            )
            reference_shot_ids = self._planned_reference_shot_ids(shot)
            selection_note = shot.get("reference_selection_note")
            note = selection_note if isinstance(selection_note, str) else None
            await asyncio.to_thread(
                self._submit_workplace_shot_generate,
                session_key,
                work_id,
                shot_id,
                shot,
                reference_shot_ids=reference_shot_ids,
                selection_note=note,
                use_continuous=False,
                reference_image_url=reference_image_url,
                i2v_prompt=rewritten_prompt,
            )
        except Exception as exc:
            logger.opt(exception=True).error(
                "workplace shot generate async failed work_id={} shot_id={}",
                work_id,
                shot_id,
            )
            self._mark_workplace_shot_failed(session_key, work_id, shot_id, str(exc))
        try:
            await self._publish_workplace_update(session_key)
        except Exception:
            logger.opt(exception=True).error(
                "workplace shot generate update failed after async submit work_id={}",
                work_id,
            )

    def _apply_workplace_shot_generate(
        self,
        session_key: str,
        shot_id: int,
        *,
        reference_image_url: str | None = None,
        reference_image_name: str | None = None,
        reference_image_width: int | None = None,
        reference_image_height: int | None = None,
    ) -> tuple[str, dict[str, Any]]:
        work_id, _shot_path, shot, _state, reference_shot_ids, note, use_continuous = (
            self._validate_workplace_shot_generate(
                session_key,
                shot_id,
            )
        )
        if shot_id == 1 and not reference_image_url:
            reference_image_url = self._workplace_first_frame_url(work_id)
        elif reference_image_url:
            logger.info(
                "workplace shot generate: ignoring request-body reference_image "
                "session_key={} shot_id={}",
                session_key,
                shot_id,
            )
            reference_image_url = (
                self._workplace_first_frame_url(work_id) if shot_id == 1 else None
            )
        self._submit_workplace_shot_generate(
            session_key,
            work_id,
            shot_id,
            shot,
            reference_shot_ids=reference_shot_ids,
            selection_note=note,
            use_continuous=use_continuous,
            reference_image_url=reference_image_url,
        )
        return work_id, self._build_workplace_payload(session_key)

    def _iter_workplace_shots(self, work_id: str) -> list[tuple[int, dict[str, Any]]]:
        paths = self._workplace_paths(work_id)
        if paths is None:
            return []
        rows: list[tuple[int, dict[str, Any]]] = []
        for shot_path in sorted(paths["shots"].glob("shot_*.json")):
            shot = self._read_json_file(shot_path, {})
            if not isinstance(shot, dict):
                continue
            try:
                shot_id = int(shot.get("shot_id"))
            except (TypeError, ValueError):
                continue
            rows.append((shot_id, shot))
        return rows

    def _lock_workplace_video_size(self, session_key: str, work_id: str) -> tuple[int, int]:
        """Copy the session size into the Director work once, before first generation."""
        state = self._load_workplace_state(work_id)
        goal = state.get("goal") if isinstance(state.get("goal"), dict) else {}
        try:
            locked_width = int(goal.get("width"))
            locked_height = int(goal.get("height"))
        except (TypeError, ValueError):
            locked_width = locked_height = 0
        if locked_width > 0 and locked_height > 0:
            return locked_width, locked_height

        from nanobot.session.generation_settings import get_generation_settings

        metadata: dict[str, Any] = {}
        if self._session_manager is not None:
            metadata = self._session_manager.get_or_create(session_key).metadata
        settings = get_generation_settings(metadata)
        width = int(settings["width"])
        height = int(settings["height"])
        goal["width"] = width
        goal["height"] = height
        state["goal"] = goal
        self._save_workplace_state(work_id, state)
        return width, height

    def _iter_workplace_shots_for_workflow(
        self,
        work_id: str,
        *,
        beat_count: int = 0,
    ) -> list[tuple[int, dict[str, Any]]]:
        rows = self._iter_workplace_shots(work_id)
        if beat_count > 0:
            rows = [(shot_id, shot) for shot_id, shot in rows if shot_id <= beat_count]
        return rows

    def _raise_generate_all_unavailable(
        self,
        work_id: str,
        *,
        beat_count: int = 0,
    ) -> None:
        rows = self._iter_workplace_shots_for_workflow(work_id, beat_count=beat_count)
        statuses = [str(shot.get("status") or "") for _shot_id, shot in rows]
        detail = ", ".join(f"{shot_id}:{status}" for shot_id, status in zip(
            (shot_id for shot_id, _ in rows),
            statuses,
            strict=True,
        ))
        suffix = f" (work_id={work_id}, shots=[{detail}])" if detail else f" (work_id={work_id})"
        if statuses and all(
            status in {"generated", "review_pass", "approved"} for status in statuses
        ):
            raise ValueError(f"all shots are already generated{suffix}")
        if statuses and all(
            status in {"queued", "generated", "review_pass", "approved"} for status in statuses
        ):
            raise ValueError(f"shots are already generating{suffix}")
        raise ValueError(f"no shots are ready to generate{suffix}")

    def _submit_workplace_echo_generation(
        self,
        session_key: str,
        work_id: str,
        shot_id: int,
        reference_shot_ids: list[int],
        selection_note: str | None,
    ) -> None:
        generate_tool = self._director_generate_tool()
        generate_tool.set_context(
            "websocket",
            webui_wire_chat_id(session_key) or "direct",
            effective_key=session_key,
        )
        self._clear_auto_generate_memory_wait(work_id)

        # Check if continuous (I2V) mode is enabled for this shot.
        loaded = self._load_workplace_shot(session_key, shot_id)
        shot = loaded[3] if loaded is not None else {}
        use_continuous = (
            bool(shot.get("continuous_enabled"))
            and shot_id > 1
            and not self._workplace_auto_generate_active(session_key)
        )

        if use_continuous:
            # ── I2V path: extract previous shot's tail frame ──
            previous_shot_id = shot_id - 1
            logger.info(
                "submit_echo_generation: shot_id={} continuous_enabled=True, "
                "using I2V (tail frame from shot {})",
                shot_id, previous_shot_id,
            )
            prev_loaded = self._load_workplace_shot(session_key, previous_shot_id)
            if prev_loaded is None:
                raise ValueError(f"previous shot {previous_shot_id} not found")
            _prev_work_id, prev_shot_path, _prev_state, prev_shot = prev_loaded
            prev_status = str(prev_shot.get("status") or "")
            if prev_status not in {"generated", "review_pass", "approved"}:
                raise ValueError(
                    f"previous shot {previous_shot_id} must be generated first "
                    f"(current: {prev_status})"
                )
            video_url = prev_shot.get("artifact_url") or (
                prev_shot.get("echo") or {}
            ).get("result_url")
            if not video_url:
                raise ValueError(
                    f"previous shot {previous_shot_id} has no artifact_url; "
                    f"cannot extract tail frame for continuous generation"
                )
            logger.info(
                "submit_echo_generation: shot_id={} extracting tail frame from "
                "previous shot {} video_url={}",
                shot_id, previous_shot_id, video_url,
            )
            condition_image_url = self._extract_and_publish_tail_frame(
                work_id, previous_shot_id, video_url,
            )
            if not condition_image_url:
                raise ValueError(
                    f"failed to extract tail frame from shot {previous_shot_id}"
                )
            logger.info(
                "submit_echo_generation: shot_id={} tail frame ready, "
                "condition_image_url={}",
                shot_id, condition_image_url,
            )
            # Persist tail_frame_url on previous shot for caching and UI display.
            prev_shot["tail_frame_url"] = condition_image_url
            self._save_workplace_shot(prev_shot_path, prev_shot)
            # Ensure the previous shot is included as a reference.
            if previous_shot_id not in reference_shot_ids:
                reference_shot_ids = sorted(
                    set(reference_shot_ids) | {previous_shot_id}
                )
            try:
                generate_tool.apply_generate_continuous(
                    work_id,
                    shot_id,
                    condition_image_url,
                    reference_shot_ids,
                    selection_note=selection_note,
                )
            except EchoGeneratorUnavailableError as exc:
                self._report_echo_unavailable(
                    session_key, exc, work_id=work_id, shot_id=shot_id
                )
                raise
        else:
            first_frame_url = self._workplace_first_frame_url(work_id) if shot_id == 1 else None
            if first_frame_url:
                logger.info(
                    "submit_echo_generation: shot_id=1 using state.reference_image "
                    "url={}",
                    first_frame_url,
                )
                try:
                    loop = asyncio.get_running_loop()
                    if loaded is not None:
                        _work, shot_path, _state, shot_obj = loaded
                        self._mark_workplace_shot_queued(work_id, shot_path, shot_obj)
                    loop.create_task(
                        self._complete_workplace_shot_generate_with_reference(
                            session_key,
                            work_id=work_id,
                            shot_id=shot_id,
                            reference_image_url=first_frame_url,
                        )
                    )
                    return
                except RuntimeError:
                    caption = str(shot.get("caption") or "").strip()
                    profile = self._load_workplace_story_profile(work_id)
                    language = ""
                    if isinstance(profile, dict):
                        language = str(
                            profile.get("caption_language") or profile.get("language") or ""
                        )
                    i2v_prompt = (
                        rewrite_prompt_for_i2v(caption, language) if caption else None
                    )
                    try:
                        generate_tool.apply_generate(
                            work_id,
                            shot_id,
                            reference_shot_ids,
                            selection_note=selection_note,
                            condition_image_url=first_frame_url,
                            i2v_prompt=i2v_prompt,
                        )
                    except EchoGeneratorUnavailableError as exc:
                        self._report_echo_unavailable(
                            session_key, exc, work_id=work_id, shot_id=shot_id
                        )
                        raise
                    return
            logger.info(
                "submit_echo_generation: shot_id={} continuous_enabled={}, "
                "using T2V (no tail frame)",
                shot_id, shot.get("continuous_enabled", False),
            )
            try:
                generate_tool.apply_generate(
                    work_id,
                    shot_id,
                    reference_shot_ids,
                    selection_note=selection_note,
                )
            except EchoGeneratorUnavailableError as exc:
                self._report_echo_unavailable(
                    session_key, exc, work_id=work_id, shot_id=shot_id
                )
                raise

    def _apply_workplace_generate_all(
        self,
        session_key: str,
    ) -> tuple[str, dict[str, Any], list[int]]:
        self._ensure_echo_admission(operation="generate_echo_shot")
        work_id = self._resolve_work_id_for_session(session_key)
        if not work_id:
            raise ValueError("director work not found")
        paths = self._workplace_paths(work_id)
        if paths is None:
            raise ValueError("director work not found")
        self._lock_workplace_video_size(session_key, work_id)
        profile = self._load_workplace_story_profile(work_id)
        beats = profile.get("beats") if isinstance(profile.get("beats"), list) else []
        beat_count = len(beats)
        state = self._load_workplace_state(work_id)
        if beat_count > 0:
            self._prune_workplace_shots_beyond_count(work_id, beat_count, state)
            self._reconcile_workplace_state_shots(work_id, state)
        self._ensure_workplace_shot_references(work_id)
        if not self._workplace_references_ready(work_id, beat_count=beat_count):
            raise ValueError("reference plan is not ready; complete the previous workflow step first")

        submitted: list[int] = []
        approval_errors: list[str] = []
        for shot_id, shot in self._iter_workplace_shots_for_workflow(work_id, beat_count=beat_count):
            status = str(shot.get("status") or "")
            if status in {"queued", "generated", "review_pass", "approved"}:
                continue
            if "planned_reference_shot_ids" not in shot:
                continue
            approval_error = self._previous_shot_approval_error(work_id, shot_id)
            if approval_error:
                approval_errors.append(approval_error)
                continue
            reference_shot_ids = self._planned_reference_shot_ids(shot)
            selection_note = shot.get("reference_selection_note")
            note = selection_note if isinstance(selection_note, str) else None
            self._submit_workplace_echo_generation(
                session_key,
                work_id,
                shot_id,
                reference_shot_ids,
                note,
            )
            submitted.append(shot_id)
            if self._memory_review_workflow_enabled():
                break

        if not submitted and approval_errors:
            raise ValueError(approval_errors[0])
        if not submitted:
            self._raise_generate_all_unavailable(work_id, beat_count=beat_count)
        logger.info(
            "workplace generate-all submitted work_id={} shot_ids={}",
            work_id,
            submitted,
        )
        return work_id, self._build_workplace_payload(session_key), submitted

    def _load_workplace_story_profile(self, work_id: str) -> dict[str, Any]:
        paths = self._workplace_paths(work_id)
        if paths is None:
            return {}
        data = self._read_json_file(paths["story_profile"], {})
        return data if isinstance(data, dict) else {}

    def _sync_story_profile_language(self, session_key: str, language: str) -> None:
        """Write UI language into story_profile when a director work already exists."""
        from nanobot.session.generation_settings import (
            language_to_caption_language,
            language_to_dialogue_language,
            normalize_language,
        )

        normalized = normalize_language(language)
        if normalized is None:
            return
        work_id = self._resolve_work_id_for_session(session_key)
        if not work_id:
            return
        paths = self._workplace_paths(work_id)
        if paths is None:
            return
        profile = self._load_workplace_story_profile(work_id)
        profile["language"] = normalized
        dialogue = language_to_dialogue_language(normalized)
        if dialogue:
            profile["dialogue_language"] = dialogue
        caption = language_to_caption_language(normalized)
        if caption:
            profile["caption_language"] = caption
        paths["story_profile"].parent.mkdir(parents=True, exist_ok=True)
        paths["story_profile"].write_text(
            json.dumps(profile, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        state = self._load_workplace_state(work_id)
        if isinstance(state, dict):
            state["story_profile"] = profile
            self._save_workplace_state(work_id, state)
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self._publish_workplace_update(session_key))
        except RuntimeError:
            pass

    @staticmethod
    def _normalize_story_profile_beats(beats: list[Any]) -> list[dict[str, Any]]:
        normalized: list[dict[str, Any]] = []
        for index, beat in enumerate(beats):
            if not isinstance(beat, dict):
                raise ValueError("each beat must be an object with shot_id and summary")
            summary = str(beat.get("summary") or "").strip()
            if not summary:
                raise ValueError("each beat.summary must be a non-empty string")
            normalized.append({"shot_id": index + 1, "summary": summary})
        return normalized

    @staticmethod
    def _validate_story_profile(profile: dict[str, Any]) -> None:
        summary = profile.get("summary")
        if not isinstance(summary, str) or not summary.strip():
            raise ValueError("story_profile.summary must be a non-empty string")
        beats = profile.get("beats")
        if not isinstance(beats, list) or len(beats) < 1:
            raise ValueError("story_profile.beats must contain at least one beat")
        profile["beats"] = WebSocketChannel._normalize_story_profile_beats(beats)

    @staticmethod
    def _sync_story_profile_derivatives(profile: dict[str, Any]) -> None:
        beats = profile.get("beats")
        if not isinstance(beats, list):
            return
        shot_to_content: dict[str, str] = {}
        content_to_shots: dict[str, list[str]] = {}
        for index, beat in enumerate(beats):
            if not isinstance(beat, dict):
                continue
            shot_id = beat.get("shot_id") or (index + 1)
            try:
                shot_id = int(shot_id)
            except (TypeError, ValueError):
                shot_id = index + 1
            shot_key = f"shot_{shot_id:03d}"
            summary = str(beat.get("summary") or "").strip()
            shot_to_content[shot_key] = summary
            content_to_shots[f"beat_{shot_id:03d}"] = [shot_key]
        profile["shot_to_content"] = shot_to_content
        profile["content_to_shots"] = content_to_shots

    def _save_workplace_story_profile(
        self,
        work_id: str,
        profile: dict[str, Any],
        state: dict[str, Any],
    ) -> None:
        from nanobot.session.generation_settings import (
            language_to_caption_language,
            language_to_dialogue_language,
            normalize_language,
        )

        paths = self._workplace_paths(work_id)
        if paths is None:
            return
        previous = self._load_workplace_story_profile(work_id)
        language = normalize_language(profile.get("language"))
        if language is None:
            language = normalize_language(previous.get("language"))
        if language is not None:
            profile["language"] = language
            dialogue = language_to_dialogue_language(language)
            if dialogue and not (
                isinstance(profile.get("dialogue_language"), str)
                and str(profile.get("dialogue_language") or "").strip()
            ):
                profile["dialogue_language"] = dialogue
            caption = language_to_caption_language(language)
            if caption and not str(profile.get("caption_language") or "").strip():
                profile["caption_language"] = caption
        elif (
            isinstance(previous.get("dialogue_language"), str)
            and previous["dialogue_language"].strip()
            and not (
                isinstance(profile.get("dialogue_language"), str)
                and str(profile.get("dialogue_language") or "").strip()
            )
        ):
            profile["dialogue_language"] = previous["dialogue_language"].strip()
        if not str(profile.get("caption_language") or "").strip():
            previous_caption = str(previous.get("caption_language") or "").strip()
            if previous_caption:
                profile["caption_language"] = previous_caption
            else:
                fallback_language = normalize_language(profile.get("dialogue_language"))
                caption = language_to_caption_language(fallback_language)
                if caption:
                    profile["caption_language"] = caption
        self._validate_story_profile(profile)
        self._sync_story_profile_derivatives(profile)
        paths["story_profile"].parent.mkdir(parents=True, exist_ok=True)
        paths["story_profile"].write_text(
            json.dumps(profile, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        state["story_profile"] = profile
        goal = state.setdefault("goal", {})
        if isinstance(goal, dict):
            beats = profile.get("beats")
            # Only sync shot_count from beats after the user has locked it.
            # Provisional beats during 01 must not jump the workplace to 02.
            if isinstance(beats, list) and locked_shot_count_from_goal(goal):
                goal["shot_count"] = len(beats)

    def _workplace_has_generated_video(self, work_id: str, state: dict[str, Any]) -> bool:
        if state.get("final_output_url") or state.get("final_output_path"):
            return True
        paths = self._workplace_paths(work_id)
        if paths is None:
            return False
        for shot_path in paths["shots"].glob("shot_*.json"):
            shot = self._read_json_file(shot_path, {})
            if isinstance(shot, dict) and self._shot_has_generated_media(shot):
                return True
        return False

    def _clear_workplace_shot_files(self, work_id: str, state: dict[str, Any]) -> None:
        paths = self._workplace_paths(work_id)
        if paths is None:
            return
        for shot_path in paths["shots"].glob("shot_*.json"):
            shot_path.unlink(missing_ok=True)
        state["shots"] = {}

    def _prune_workplace_shots_beyond_count(
        self,
        work_id: str,
        beat_count: int,
        state: dict[str, Any],
    ) -> None:
        """Remove orphan shot files/state rows when beats were merged or split."""
        if beat_count <= 0:
            return
        paths = self._workplace_paths(work_id)
        if paths is None:
            return
        changed = False
        for shot_path in list(paths["shots"].glob("shot_*.json")):
            shot = self._read_json_file(shot_path, {})
            shot_id = 0
            if isinstance(shot, dict):
                try:
                    shot_id = int(shot.get("shot_id") or 0)
                except (TypeError, ValueError):
                    shot_id = 0
            if shot_id <= 0:
                stem = shot_path.stem
                if stem.startswith("shot_"):
                    try:
                        shot_id = int(stem.removeprefix("shot_"))
                    except ValueError:
                        shot_id = 0
            if shot_id > beat_count:
                shot_path.unlink(missing_ok=True)
                changed = True
        shots = state.get("shots")
        if isinstance(shots, dict):
            for key in list(shots.keys()):
                entry = shots[key]
                if not isinstance(entry, dict):
                    continue
                try:
                    shot_id = int(entry.get("shot_id") or 0)
                except (TypeError, ValueError):
                    continue
                if shot_id > beat_count:
                    del shots[key]
                    changed = True
        if changed:
            self._save_workplace_state(work_id, state)

    def _load_workplace_state(self, work_id: str) -> dict[str, Any]:
        paths = self._workplace_paths(work_id)
        if paths is None:
            return {}
        state = self._read_json_file(paths["state"], {})
        return state if isinstance(state, dict) else {}

    _BEATS_EDITABLE_STAGES = frozenset(
        {"story_discussion", "story_confirmed", "shot_planning"},
    )
    _REGENERATION_SHOT_CLEAR_KEYS = (
        "artifact_url",
        "artifact_path",
        "generation_error",
        "last_job_id",
        "echo",
        "remote_result",
        "last_review",
        "review_notes",
        "reference_shot_ids",
        "reference_selection_note",
        "planned_reference_shot_ids",
        "memory_review",
    )

    def _workplace_beats_editable(self, work_id: str, state: dict[str, Any]) -> bool:
        stage = str(state.get("stage") or "")
        if stage not in self._BEATS_EDITABLE_STAGES:
            return False
        return not self._workplace_has_generated_video(work_id, state)

    def _assert_story_discussion_editable(self, state: dict[str, Any], *, field: str) -> None:
        if str(state.get("stage") or "") != "story_discussion":
            raise ValueError(f"{field} can only be edited during story_discussion")

    def _assert_beats_editable(self, work_id: str, state: dict[str, Any]) -> None:
        if not self._workplace_beats_editable(work_id, state):
            raise ValueError(
                "story_profile.beats can only be edited before video generation "
                "during story_discussion, story_confirmed, or shot_planning"
            )

    def _find_beat_index(self, beats: list[dict[str, Any]], shot_id: int) -> int:
        for index, beat in enumerate(beats):
            try:
                if int(beat.get("shot_id") or 0) == shot_id:
                    return index
            except (TypeError, ValueError):
                continue
        raise ValueError(f"beat {shot_id} not found")

    def _save_workplace_state(self, work_id: str, state: dict[str, Any]) -> None:
        paths = self._workplace_paths(work_id)
        if paths is None:
            return
        state["updated_at"] = datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
        self._write_json_file(paths["state"], state)

    def _try_sync_workplace_story_confirmed(
        self,
        work_id: str,
        paths: dict[str, Path],
        state: dict[str, Any],
    ) -> bool:
        """Persist story_confirmed when screenplay and profile already validate."""
        if state.get("story_confirmed"):
            return True
        if not self._workplace_story_text(paths).strip():
            return False
        profile = self._load_workplace_story_profile(work_id)
        if _story_profile_validation_error(profile):
            return False
        state["story_confirmed"] = True
        self._save_workplace_state(work_id, state)
        return True

    def _save_workplace_shot(self, shot_path: Path, shot: dict[str, Any]) -> None:
        shot["updated_at"] = datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
        self._write_json_file(shot_path, shot)

    @staticmethod
    def _shot_has_caption(shot: dict[str, Any]) -> bool:
        caption = shot.get("caption")
        return isinstance(caption, str) and bool(caption.strip())

    @classmethod
    def _reset_shot_for_regeneration(cls, shot: dict[str, Any]) -> None:
        for key in cls._REGENERATION_SHOT_CLEAR_KEYS:
            shot.pop(key, None)
        if cls._shot_has_caption(shot):
            shot["status"] = "prompt_ready"
        else:
            shot["status"] = "planned"

    def _apply_workplace_regenerate(
        self,
        session_key: str,
    ) -> tuple[str, dict[str, Any]]:
        """Return to storyboard editing after a final video so the user can revise and re-run."""
        work_id = self._resolve_work_id_for_session(session_key)
        if not work_id:
            raise ValueError("director work not found")
        paths = self._workplace_paths(work_id)
        if paths is None:
            raise ValueError("director work not found")
        state = self._read_json_file(paths["state"], {})
        if not isinstance(state, dict):
            state = {}
        stage = str(state.get("stage") or "")
        has_final_output = bool(state.get("final_output_url") or state.get("final_output_path"))
        if stage not in {"done", "merging"} and not has_final_output:
            raise ValueError("regenerate is only available after a final video was produced")

        state["pending_remote_jobs"] = {}
        state.pop("final_output_url", None)
        state.pop("final_output_path", None)
        state.pop("latest_merge_job_id", None)
        state.pop("review_completed_at", None)
        for key in (
            "echo_request_id",
            "like_status",
            "prompt_downloaded",
            "video_downloaded",
        ):
            state.pop(key, None)
        state["sequential_generate_all"] = False
        state["stage"] = "shot_planning"

        for shot_path in sorted(paths["shots"].glob("shot_*.json")):
            shot = self._read_json_file(shot_path, {})
            if not isinstance(shot, dict):
                continue
            self._reset_shot_for_regeneration(shot)
            self._save_workplace_shot(shot_path, shot)
            self._sync_state_shot_row(state, shot)

        self._save_workplace_state(work_id, state)
        # Shot specs survive regenerate but reference plans are cleared; restore
        # defaults immediately so step-3 generate-all is not blocked.
        self._ensure_workplace_shot_references(work_id)
        auto = bool(state.get("auto_generate")) or self._session_auto_generate_flag(
            session_key
        )
        if auto:
            state = self._load_workplace_state(work_id)
            state["auto_generate"] = True
            state["auto_generate_retry_count"] = 0
            state.pop("auto_generate_waited_memory", None)
            self._save_workplace_state(work_id, state)
            self._continue_auto_generate(session_key)
        return work_id, self._build_workplace_payload(session_key)

    @staticmethod
    def _merge_shot_text_parts(left: str, right: str) -> str:
        left = left.strip()
        right = right.strip()
        if not left:
            return right
        if not right:
            return left
        return f"{left}\n\n{right}"

    @classmethod
    def _merge_shot_payloads(
        cls,
        upper_shot: dict[str, Any],
        lower_shot: dict[str, Any],
        *,
        merged_summary: str | None = None,
    ) -> dict[str, Any]:
        merged = dict(upper_shot)
        if merged_summary is not None:
            merged["summary"] = merged_summary.strip()
        else:
            merged["summary"] = cls._merge_shot_text_parts(
                str(upper_shot.get("summary") or ""),
                str(lower_shot.get("summary") or ""),
            )
        upper_caption = upper_shot.get("caption")
        lower_caption = lower_shot.get("caption")
        if isinstance(upper_caption, str) or isinstance(lower_caption, str):
            merged["caption"] = cls._merge_shot_text_parts(
                upper_caption if isinstance(upper_caption, str) else "",
                lower_caption if isinstance(lower_caption, str) else "",
            )
        return merged

    @staticmethod
    def _shot_has_generated_media(shot: dict[str, Any]) -> bool:
        if shot.get("artifact_path") or shot.get("artifact_url"):
            return True
        status = str(shot.get("status") or "")
        return status in {"queued", "generated", "review_pass", "review_fail", "approved"}

    @staticmethod
    def _parse_json_body_payload(request: WsRequest) -> dict[str, Any] | None:
        raw_body = getattr(request, "body", None)
        if isinstance(raw_body, (bytes, bytearray)) and raw_body:
            try:
                payload = json.loads(raw_body)
            except (json.JSONDecodeError, UnicodeDecodeError):
                payload = None
            if isinstance(payload, dict):
                return payload
        encoded = request.headers.get("X-Nanobot-Body")
        if encoded:
            try:
                payload = json.loads(base64.b64decode(encoded, validate=True))
            except (json.JSONDecodeError, UnicodeDecodeError, binascii.Error, ValueError):
                payload = None
            if isinstance(payload, dict):
                return payload
        return None

    @staticmethod
    def _parse_split_shot_body(request: WsRequest) -> dict[str, Any] | None:
        payload = WebSocketChannel._parse_json_body_payload(request)
        result: dict[str, Any] = {}
        if isinstance(payload, dict):
            for key in ("before_text", "after_text", "cursor_pos"):
                if key in payload:
                    result[key] = payload[key]
        query = _parse_query(request.path)
        for key in ("before_text", "after_text", "cursor_pos"):
            value = _query_first(query, key)
            if value is not None:
                if key == "cursor_pos":
                    try:
                        result[key] = int(value)
                    except ValueError:
                        pass
                else:
                    result[key] = value
        return result or None

    @classmethod
    def _build_split_shot_payloads(
        cls,
        shot: dict[str, Any],
        *,
        before_text: str,
        after_text: str,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        upper = dict(shot)
        lower = dict(shot)
        upper["summary"] = before_text
        lower["summary"] = after_text
        caption = shot.get("caption")
        if isinstance(caption, str):
            upper["caption"] = before_text
            lower["caption"] = after_text
        for field in (
            "artifact_path",
            "artifact_url",
            "last_review",
            "review_notes",
            "generation_error",
            "last_job_id",
            "approved_at",
            "remote_result",
        ):
            upper.pop(field, None)
            lower.pop(field, None)
        upper["status"] = "prompt_ready"
        lower["status"] = "prompt_ready"
        lower["cut"] = True
        return upper, lower

    _SHOT_DURATION_MIN_SEC = 1
    _SHOT_DURATION_MAX_SEC = 10
    _SHOT_DURATION_DEFAULT_SEC = 5
    # Must stay aligned with webui AspectRatioPicker presets.
    _VALID_VIDEO_SIZES = frozenset(
        {
            (1280, 736),  # 16:9
            (736, 736),  # 1:1
            (736, 1280),  # 9:16
        }
    )

    @classmethod
    def _clamp_shot_duration_sec(cls, value: float) -> int:
        return max(
            cls._SHOT_DURATION_MIN_SEC,
            min(cls._SHOT_DURATION_MAX_SEC, int(round(value))),
        )

    @staticmethod
    def _parse_shot_duration_body(request: WsRequest) -> int | None:
        raw_body = getattr(request, "body", None)
        if isinstance(raw_body, (bytes, bytearray)) and raw_body:
            try:
                payload = json.loads(raw_body)
            except (json.JSONDecodeError, UnicodeDecodeError):
                payload = None
            if isinstance(payload, dict):
                value = payload.get("duration_sec")
                if value is None:
                    value = payload.get("duration_seconds")
                try:
                    return int(round(float(value)))
                except (TypeError, ValueError):
                    pass
        encoded = request.headers.get("X-Nanobot-Body")
        if encoded:
            try:
                payload = json.loads(base64.b64decode(encoded, validate=True))
            except (json.JSONDecodeError, UnicodeDecodeError, binascii.Error, ValueError):
                payload = None
            if isinstance(payload, dict):
                value = payload.get("duration_sec")
                if value is None:
                    value = payload.get("duration_seconds")
                try:
                    return int(round(float(value)))
                except (TypeError, ValueError):
                    pass
        query = _parse_query(request.path)
        for key in ("duration_sec", "duration_seconds"):
            value = _query_first(query, key)
            if value is None:
                continue
            try:
                return int(round(float(value)))
            except (TypeError, ValueError):
                continue
        return None

    @staticmethod
    def _parse_merge_shot_body(request: WsRequest) -> str | None:
        raw_body = getattr(request, "body", None)
        if isinstance(raw_body, (bytes, bytearray)) and raw_body:
            try:
                payload = json.loads(raw_body)
            except (json.JSONDecodeError, UnicodeDecodeError):
                payload = None
            if isinstance(payload, dict):
                value = payload.get("merged_text")
                if isinstance(value, str):
                    return value
        encoded = request.headers.get("X-Nanobot-Body")
        if encoded:
            try:
                payload = json.loads(base64.b64decode(encoded, validate=True))
            except (json.JSONDecodeError, UnicodeDecodeError, binascii.Error, ValueError):
                payload = None
            if isinstance(payload, dict):
                value = payload.get("merged_text")
                if isinstance(value, str):
                    return value
        value = _query_first(_parse_query(request.path), "merged_text")
        if value is not None:
            return value
        return None

    def _rebuild_workplace_state_shots(self, work_id: str, state: dict[str, Any]) -> None:
        paths = self._workplace_paths(work_id)
        if paths is None:
            return
        state["shots"] = {}
        for shot_path in sorted(paths["shots"].glob("shot_*.json")):
            shot = self._read_json_file(shot_path, {})
            if isinstance(shot, dict) and shot.get("shot_id"):
                self._sync_state_shot_row(state, shot)

    def _reconcile_workplace_state_shots(self, work_id: str, state: dict[str, Any]) -> bool:
        """Sync ``state.shots`` from on-disk shot JSON. Persists when changed."""
        before = json.dumps(state.get("shots"), sort_keys=True, default=str)
        self._rebuild_workplace_state_shots(work_id, state)
        after = json.dumps(state.get("shots"), sort_keys=True, default=str)
        if before == after:
            return False
        self._save_workplace_state(work_id, state)
        return True

    def _apply_workplace_shot_merge_up(
        self,
        session_key: str,
        shot_id: int,
        merged_text: str | None = None,
    ) -> tuple[str, dict[str, Any]] | None:
        if shot_id < 2:
            raise ValueError("cannot merge the first beat")
        work_id = self._resolve_work_id_for_session(session_key)
        if not work_id:
            return None
        state = self._load_workplace_state(work_id)
        self._assert_beats_editable(work_id, state)
        profile = self._load_workplace_story_profile(work_id)
        beats = profile.get("beats")
        if not isinstance(beats, list) or not beats:
            raise ValueError("story_profile.beats is empty")
        beat_items = [beat for beat in beats if isinstance(beat, dict)]
        index = self._find_beat_index(beat_items, shot_id)
        if index < 1:
            raise ValueError("cannot merge the first beat")
        upper = beat_items[index - 1]
        lower = beat_items[index]
        if merged_text is not None and merged_text.strip():
            merged_summary = merged_text.strip()
        else:
            merged_summary = self._merge_shot_text_parts(
                str(upper.get("summary") or ""),
                str(lower.get("summary") or ""),
            )
        upper["summary"] = merged_summary
        beat_items.pop(index)
        profile["beats"] = self._normalize_story_profile_beats(beat_items)
        self._clear_workplace_shot_files(work_id, state)
        self._save_workplace_story_profile(work_id, profile, state)
        self._save_workplace_state(work_id, state)
        self._schedule_workplace_beats_edit_instruction(session_key, work_id=work_id)
        workplace = self._build_workplace_payload(session_key)
        return work_id, workplace

    def _apply_workplace_shot_remove(
        self,
        session_key: str,
        shot_id: int,
    ) -> tuple[str, dict[str, Any]] | None:
        work_id = self._resolve_work_id_for_session(session_key)
        if not work_id:
            return None
        state = self._load_workplace_state(work_id)
        self._assert_beats_editable(work_id, state)
        profile = self._load_workplace_story_profile(work_id)
        beats = profile.get("beats")
        if not isinstance(beats, list) or not beats:
            raise ValueError("story_profile.beats is empty")
        beat_items = [beat for beat in beats if isinstance(beat, dict)]
        if len(beat_items) <= 1:
            raise ValueError("cannot remove the last beat")
        index = self._find_beat_index(beat_items, shot_id)
        beat_items.pop(index)
        profile["beats"] = self._normalize_story_profile_beats(beat_items)
        self._clear_workplace_shot_files(work_id, state)
        self._save_workplace_story_profile(work_id, profile, state)
        self._save_workplace_state(work_id, state)
        self._schedule_workplace_beats_edit_instruction(session_key, work_id=work_id)
        workplace = self._build_workplace_payload(session_key)
        return work_id, workplace

    def _apply_workplace_shot_split_shot(
        self,
        session_key: str,
        shot_id: int,
        split_payload: dict[str, Any] | None = None,
    ) -> tuple[str, dict[str, Any]] | None:
        work_id = self._resolve_work_id_for_session(session_key)
        if not work_id:
            return None
        state = self._load_workplace_state(work_id)
        self._assert_beats_editable(work_id, state)
        profile = self._load_workplace_story_profile(work_id)
        beats = profile.get("beats")
        if not isinstance(beats, list) or not beats:
            raise ValueError("story_profile.beats is empty")
        beat_items = [beat for beat in beats if isinstance(beat, dict)]
        index = self._find_beat_index(beat_items, shot_id)
        beat = beat_items[index]

        payload = split_payload or {}
        before_text = payload.get("before_text")
        after_text = payload.get("after_text")
        if before_text is None or after_text is None:
            if "cursor_pos" not in payload:
                raise ValueError("before_text and after_text are required")
            summary = str(beat.get("summary") or "")
            try:
                cursor_pos = int(payload["cursor_pos"])
            except (TypeError, ValueError) as exc:
                raise ValueError("cursor_pos must be an integer") from exc
            cursor_pos = max(0, min(cursor_pos, len(summary)))
            before_text = summary[:cursor_pos].rstrip()
            after_text = summary[cursor_pos:].lstrip()
        before_text = str(before_text).rstrip()
        after_text = str(after_text).lstrip()
        if not before_text or not after_text:
            raise ValueError("请把光标放在镜头文本中间再拆分（不能在开头或末尾）")

        beat_items[index] = {"shot_id": shot_id, "summary": before_text}
        beat_items.insert(index + 1, {"shot_id": shot_id + 1, "summary": after_text})
        profile["beats"] = self._normalize_story_profile_beats(beat_items)
        self._clear_workplace_shot_files(work_id, state)
        self._save_workplace_story_profile(work_id, profile, state)
        self._save_workplace_state(work_id, state)
        self._schedule_workplace_beats_edit_instruction(session_key, work_id=work_id)
        workplace = self._build_workplace_payload(session_key)
        return work_id, workplace

    def _sync_state_shot_row(self, state: dict[str, Any], shot: dict[str, Any]) -> None:
        shots = state.setdefault("shots", {})
        if not isinstance(shots, dict):
            shots = {}
            state["shots"] = shots
        shot_id = int(shot.get("shot_id") or 0)
        if shot_id <= 0:
            return
        shot_key = str(shot.get("shot_key") or f"shot_{shot_id:03d}")
        shots[shot_key] = {
            "shot_id": shot_id,
            "status": shot.get("status"),
            "summary": shot.get("summary") or "",
            "cut": bool(shot.get("cut", True)),
            "has_shot_spec": self._shot_has_caption(shot),
            "has_artifact": bool(shot.get("artifact_path") or shot.get("artifact_url")),
            "artifact_path": shot.get("artifact_path"),
            "artifact_url": shot.get("artifact_url"),
            "last_review": shot.get("last_review"),
            "review_notes": shot.get("review_notes") or "",
            "updated_at": shot.get("updated_at"),
        }

    @staticmethod
    def _shot_has_video_artifact(shot: dict[str, Any]) -> bool:
        for key in ("artifact_url", "artifact_path"):
            value = shot.get(key)
            if isinstance(value, str) and value.strip():
                return True
        remote_result = shot.get("remote_result")
        if isinstance(remote_result, dict):
            video_path = remote_result.get("video_path")
            if isinstance(video_path, str) and video_path.strip():
                return True
        return False

    def _shot_ready_to_accept(self, shot: dict[str, Any]) -> bool:
        status = str(shot.get("status") or "")
        if status not in {"generated", "review_pass"}:
            return False
        return self._shot_has_video_artifact(shot)

    def _raise_accept_all_unavailable(self, work_id: str) -> None:
        rows = self._iter_workplace_shots(work_id)
        video_rows = [
            (shot_id, shot)
            for shot_id, shot in rows
            if self._shot_has_video_artifact(shot)
            or str(shot.get("status") or "") in {"generated", "review_pass", "approved"}
        ]
        detail = ", ".join(
            f"{shot_id}:{status}"
            for shot_id, shot in rows
            for status in [str(shot.get("status") or "")]
        )
        suffix = f" (work_id={work_id}, shots=[{detail}])" if detail else f" (work_id={work_id})"
        if video_rows and all(
            str(shot.get("status") or "") == "approved" for _shot_id, shot in video_rows
        ):
            raise ValueError(f"all shots with video are already accepted{suffix}")
        raise ValueError(f"no shots with video are ready to accept{suffix}")

    def _apply_workplace_accept_all(
        self,
        session_key: str,
    ) -> tuple[str, dict[str, Any], list[int]]:
        work_id = self._resolve_work_id_for_session(session_key)
        if not work_id:
            raise ValueError("director work not found")
        accepted: list[int] = []
        for shot_id, shot in self._iter_workplace_shots(work_id):
            if not self._shot_ready_to_accept(shot):
                continue
            self._apply_workplace_review(session_key, shot_id, verdict="accept")
            accepted.append(shot_id)
        if not accepted:
            self._raise_accept_all_unavailable(work_id)
        return work_id, self._build_workplace_payload(session_key), accepted

    def _apply_workplace_review(
        self,
        session_key: str,
        shot_id: int,
        *,
        verdict: str,
        feedback: str | None = None,
        review_source: str = "human",
    ) -> tuple[str, dict[str, Any]] | None:
        if self._session_manager is None:
            return None
        loaded = self._load_workplace_shot(session_key, shot_id)
        if loaded is None:
            return None
        work_id, _shot_path, _state, _shot = loaded
        tool = ReviewShotTool(workspace=self._session_manager.workspace)
        tool._apply_shot_review(
            work_id,
            shot_id,
            verdict=verdict,
            review_source=review_source,
            feedback=feedback,
        )
        if verdict == "accept":
            self._maybe_advance_approved_shot_memory(session_key, work_id, shot_id)
        return work_id, self._build_workplace_payload(session_key)

    def _handle_workplace_status(self, request: WsRequest, key: str) -> Response:
        if not self._check_api_token(request):
            return _http_error(401, "Unauthorized")
        legacy_err = self._legacy_webui_session_key_error(key)
        if legacy_err is not None:
            return legacy_err
        decoded_key = self._resolve_webui_api_session_key(key, request)
        if decoded_key is None:
            return _http_error(404, "session not found")
        payload = self._build_workplace_payload(decoded_key)
        self._schedule_auto_generate_continue(decoded_key)
        return _http_json_response(payload)

    async def _publish_workplace_update(self, session_key: str) -> None:
        if self._session_manager is None:
            return
        if not self._is_webui_session_key(session_key):
            return
        chat_id = webui_wire_chat_id(session_key)
        if not chat_id:
            return
        await self.send(
            OutboundMessage(
                channel="websocket",
                chat_id=chat_id,
                content="",
                metadata={
                    "_workplace_event": "updated",
                    "workplace": self._build_workplace_payload(session_key),
                    "work_id": self._resolve_work_id_for_session(session_key),
                },
            )
        )

    def _schedule_coro(
        self,
        factory: Callable[[], Any],
        *,
        warning: str | None = None,
    ) -> bool:
        """Schedule ``factory()`` on the websocket loop, including from worker threads.

        ``send()`` continues one-click generate via ``asyncio.to_thread``. That
        worker has no running loop, so ``get_running_loop()`` fails and workplace
        injections (start_generation, revision, abort) would otherwise be dropped.
        """
        try:
            loop = asyncio.get_running_loop()
            in_loop_thread = True
        except RuntimeError:
            loop = self._loop
            in_loop_thread = False
        if loop is None or loop.is_closed():
            if warning:
                logger.warning(warning)
            return False
        if not in_loop_thread and not loop.is_running():
            if warning:
                logger.warning(warning)
            return False
        coro = factory()
        if in_loop_thread:
            loop.create_task(coro)
            return True
        asyncio.run_coroutine_threadsafe(coro, loop)
        return True

    def _schedule_publish_workplace_update(self, session_key: str) -> None:
        self._schedule_coro(lambda: self._publish_workplace_update(session_key))

    def _schedule_workplace_revision_instruction(
        self,
        session_key: str,
        *,
        work_id: str,
        shot_id: int,
        feedback: str,
    ) -> None:
        content = (
            "Internal workplace revision task. Execute silently.\n\n"
            f"Director work_id: `{work_id}`\n"
            f"Target shot_id: `{shot_id}`\n"
            f"User revision feedback:\n{feedback}\n\n"
            "Required actions, in order:\n"
            "1. Use director tools to inspect the current target shot.\n"
            "2. Rewrite/update only this shot's prompt/spec so it directly incorporates the feedback.\n"
            "3. Recalculate the correct `reference_shot_ids` for this revised shot.\n"
            "4. Immediately call `generate_echo_shot` for this same shot_id with the recalculated references.\n\n"
            "Strict constraints:\n"
            "- Do not ask the user any question.\n"
            "- Do not acknowledge, summarize, or explain the feedback to the user.\n"
            "- Do not send any user-facing message before or after the tool calls.\n"
            "- Do not regenerate other shots unless the feedback explicitly requires cross-shot changes.\n"
            "- Do not stop after recording state; the regeneration tool call is mandatory."
        )
        msg = InboundMessage(
            channel="system",
            sender_id="workplace",
            chat_id=session_key,
            content=content,
            session_key_override=session_key,
            metadata={
                "injected_event": "workplace_shot_revision",
                "injected_role": "user",
                "silent": True,
                "work_id": work_id,
                "shot_id": shot_id,
            },
        )
        self._schedule_coro(
            lambda: self.bus.publish_inbound(msg),
            warning="websocket: unable to schedule workplace revision instruction",
        )

    def _memory_review_workflow_enabled(self) -> bool:
        return self._tools_config.memory_review.enabled

    def _maybe_advance_approved_shot_memory(
        self,
        session_key: str,
        work_id: str,
        shot_id: int,
    ) -> bool:
        """Advance once, only after both the video and selected Memory are accepted."""
        if not self._memory_review_workflow_enabled():
            return False
        if self._workplace_auto_generate_active(session_key):
            # Auto-generation already submits the next shot via generate_all.
            return False
        loaded = self._load_workplace_shot(session_key, shot_id)
        if loaded is None:
            return False
        _work_id, shot_path, state, shot = loaded
        review = shot.get("memory_review")
        if str(shot.get("status") or "") != "approved":
            return False
        if not isinstance(review, dict) or review.get("status") != "approved":
            return False
        if review.get("advance_complete") is True:
            return False

        from nanobot.director.r2v_memory_workflow import (
            approve_review_and_prepare_next,
        )

        next_shot_id = approve_review_and_prepare_next(
            workspace=(
                self._session_manager.workspace
                if self._session_manager is not None
                else Path.cwd()
            ),
            work_id=work_id,
            shot_id=shot_id,
        )
        if next_shot_id is not None:
            # Interactive mode pauses here. The agent proposes a draft from
            # text profiles, then Build Memory is the sole approval gate.
            self._schedule_workplace_memory_recommendation_instruction(
                session_key,
                work_id=work_id,
                shot_id=int(next_shot_id),
            )

        current = self._read_json_file(shot_path, {})
        if not isinstance(current, dict):
            current = shot
        current_review = current.get("memory_review")
        if isinstance(current_review, dict):
            current_review["advance_complete"] = True
            current_review["advance_next_shot_id"] = next_shot_id
            current["memory_review"] = current_review
            self._write_json_file(shot_path, current)
            latest_state = self._load_workplace_state(work_id)
            self._sync_state_shot_row(latest_state, current)
            self._save_workplace_state(work_id, latest_state)
        return True

    def _apply_memory_review_action(
        self,
        session_key: str,
        shot_id: int,
        *,
        action: str,
        review_id: str,
        attempt: int,
        memory_id: str | None = None,
        retained_memory_ids: list[str] | None = None,
    ) -> tuple[str, dict[str, Any], bool]:
        loaded = self._load_workplace_shot(session_key, shot_id)
        if loaded is None:
            raise ValueError(f"shot {shot_id} not found")
        work_id, shot_path, state, shot = loaded
        review = shot.get("memory_review")
        if not isinstance(review, dict):
            raise ValueError(f"shot {shot_id} has no memory review")
        if action == "approve" and str(review.get("status") or "") == "awaiting_method":
            paths = self._workplace_paths(work_id)
            current_bank = self._read_json_file(paths["memory_bank"], {})
            review["status"] = "awaiting_review"
            review["selection_mode"] = "none"
            review["selections"] = []
            review["retained_memory_ids"] = []
            review["proposed_bank"] = (
                current_bank if isinstance(current_bank, dict) else {}
            )
            review["previous_shot"] = None
        updated_at = (
            datetime.now(timezone.utc)
            .isoformat(timespec="seconds")
            .replace("+00:00", "Z")
        )
        if action == "approve":
            changed = approve_memory_review(
                review,
                review_id=review_id,
                attempt=attempt,
                updated_at=updated_at,
                retained_memory_ids=retained_memory_ids,
            )
            if changed:
                from nanobot.director.r2v_memory_workflow import (
                    stage_after_memory_advance,
                )

                state["stage"] = stage_after_memory_advance(state, has_next=True)
        elif action == "reselect":
            changed = reselect_memory_review(
                review,
                review_id=review_id,
                attempt=attempt,
                updated_at=updated_at,
                memory_id=memory_id,
            )
            if changed:
                state["stage"] = "awaiting_memory_review"
        else:
            raise ValueError(f"unsupported memory review action: {action}")
        if changed:
            shot["memory_review"] = review
            self._write_json_file(shot_path, shot)
            self._sync_state_shot_row(state, shot)
            self._save_workplace_state(work_id, state)
            if action == "approve":
                self._maybe_advance_approved_shot_memory(
                    session_key, work_id, shot_id
                )
        return work_id, self._build_workplace_payload(session_key), changed

    async def _rerun_memory_selection(
        self,
        session_key: str,
        work_id: str,
        shot_id: int,
        memory_id: str | None = None,
        selection_mode: str = "vlm",
    ) -> None:
        try:
            runner_kwargs: dict[str, Any] = {
                "workspace": (
                    self._session_manager.workspace
                    if self._session_manager is not None
                    else Path.cwd()
                ),
                "work_id": work_id,
                "shot_id": shot_id,
                "target_memory_id": memory_id,
            }
            if selection_mode != "vlm":
                runner_kwargs["selection_mode"] = selection_mode
            await asyncio.to_thread(
                self._memory_review_runner,
                **runner_kwargs,
            )
        except Exception as exc:
            logger.exception(
                "websocket: memory reselection failed work_id={} shot_id={}",
                work_id,
                shot_id,
            )
            try:
                from nanobot.director.memory_coordinator import (
                    initialize_memory_review_method_prompt,
                )

                await asyncio.to_thread(
                    initialize_memory_review_method_prompt,
                    workspace=(
                        self._session_manager.workspace
                        if self._session_manager is not None
                        else Path.cwd()
                    ),
                    work_id=work_id,
                    shot_id=shot_id,
                    error=f"{selection_mode.upper()} memory selection failed: {exc}",
                )
            except Exception:
                logger.exception(
                    "websocket: failed to persist memory selection error "
                    "work_id={} shot_id={}",
                    work_id,
                    shot_id,
                )
        await self._publish_workplace_update(session_key)

    def _handle_memory_review_action(
        self,
        request: WsRequest,
        key: str,
        shot_id: int,
        action: str,
    ) -> Response:
        resolved = self._resolve_workplace_request(request, key)
        if not isinstance(resolved, tuple):
            return resolved
        decoded_key, _work_id = resolved
        payload = self._parse_json_body_payload(request)
        if not isinstance(payload, dict):
            return _http_error(400, "memory review action body is required")
        review_id = str(payload.get("review_id") or "").strip()
        try:
            attempt = int(payload.get("attempt"))
        except (TypeError, ValueError):
            attempt = 0
        if not review_id or attempt <= 0:
            return _http_error(400, "review_id and attempt are required")
        memory_id_value = payload.get("memory_id")
        memory_id = (
            str(memory_id_value).strip()
            if isinstance(memory_id_value, str) and memory_id_value.strip()
            else None
        )
        retained_memory_ids: list[str] | None = None
        if action == "approve" and "retained_memory_ids" in payload:
            raw_retained = payload.get("retained_memory_ids")
            if not isinstance(raw_retained, list) or not all(
                isinstance(value, str) and value.strip()
                for value in raw_retained
            ):
                return _http_error(
                    400, "retained_memory_ids must be a list of memory IDs"
                )
            retained_memory_ids = [value.strip() for value in raw_retained]
        if action == "select-mode":
            selection_mode = str(payload.get("selection_mode") or "").strip()
            if selection_mode not in {"manual", "vlm"}:
                return _http_error(400, "selection_mode must be manual or vlm")
            loaded = self._load_workplace_shot(decoded_key, shot_id)
            if loaded is None:
                return _http_error(404, f"shot {shot_id} not found")
            work_id, shot_path, state, shot = loaded
            review = shot.get("memory_review")
            if not isinstance(review, dict):
                return _http_error(400, f"shot {shot_id} has no memory review")
            if (
                str(review.get("review_id") or "") != review_id
                or int(review.get("attempt") or 0) != attempt
            ):
                return _http_error(409, "stale memory review attempt")
            if str(review.get("status") or "") != "awaiting_method":
                return _http_error(409, "memory selection method was already chosen")
            review["status"] = "selecting"
            review["selection_mode"] = selection_mode
            review["updated_at"] = (
                datetime.now(timezone.utc)
                .isoformat(timespec="seconds")
                .replace("+00:00", "Z")
            )
            shot["memory_review"] = review
            self._write_json_file(shot_path, shot)
            self._sync_state_shot_row(state, shot)
            self._save_workplace_state(work_id, state)
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(
                    self._rerun_memory_selection(
                        decoded_key,
                        work_id,
                        shot_id,
                        selection_mode=selection_mode,
                    )
                )
            except RuntimeError:
                return _http_error(500, "memory selection worker is unavailable")
            return _http_json_response(
                {
                    "ok": True,
                    "action": "memory_review_select_mode",
                    "work_id": work_id,
                    "shot_id": shot_id,
                    "changed": True,
                    "workplace": self._build_workplace_payload(decoded_key),
                }
            )
        if action == "manual-select":
            try:
                timestamp_sec = float(payload.get("timestamp_sec"))
            except (TypeError, ValueError):
                return _http_error(400, "timestamp_sec is required")
            if memory_id is None or timestamp_sec < 0:
                return _http_error(400, "memory_id and non-negative timestamp_sec are required")
            try:
                select_manual_memory_frame(
                    workspace=(
                        self._session_manager.workspace
                        if self._session_manager is not None
                        else Path.cwd()
                    ),
                    work_id=_work_id,
                    shot_id=shot_id,
                    review_id=review_id,
                    attempt=attempt,
                    memory_id=memory_id,
                    timestamp_sec=timestamp_sec,
                )
                loaded = self._load_workplace_shot(decoded_key, shot_id)
                if loaded is None:
                    raise ValueError(f"shot {shot_id} not found")
                work_id, _shot_path, state, shot = loaded
                self._sync_state_shot_row(state, shot)
                self._save_workplace_state(work_id, state)
                workplace = self._build_workplace_payload(decoded_key)
            except MemoryReviewConflict as exc:
                return _http_error(409, str(exc))
            except (OSError, RuntimeError, ValueError) as exc:
                return _http_error(400, str(exc))
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(self._publish_workplace_update(decoded_key))
            except RuntimeError:
                pass
            return _http_json_response(
                {
                    "ok": True,
                    "action": "memory_review_manual_select",
                    "work_id": work_id,
                    "shot_id": shot_id,
                    "changed": True,
                    "workplace": workplace,
                }
            )
        try:
            work_id, workplace, changed = self._apply_memory_review_action(
                decoded_key,
                shot_id,
                action=action,
                review_id=review_id,
                attempt=attempt,
                memory_id=memory_id,
                retained_memory_ids=retained_memory_ids,
            )
        except MemoryReviewConflict as exc:
            return _http_error(409, str(exc))
        except ValueError as exc:
            return _http_error(400, str(exc))
        if (
            changed
            and action == "reselect"
            and self._memory_review_workflow_enabled()
        ):
            asyncio.create_task(
                self._rerun_memory_selection(
                    decoded_key, work_id, shot_id, memory_id
                )
            )
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self._publish_workplace_update(decoded_key))
        except RuntimeError:
            pass
        return _http_json_response(
            {
                "ok": True,
                "action": f"memory_review_{action}",
                "work_id": work_id,
                "shot_id": shot_id,
                "changed": changed,
                "workplace": workplace,
            }
        )

    def _handle_workplace_shot_accept(self, request: WsRequest, key: str, shot_id: int) -> Response:
        if not self._check_api_token(request):
            return _http_error(401, "Unauthorized")
        legacy_err = self._legacy_webui_session_key_error(key)
        if legacy_err is not None:
            return legacy_err
        decoded_key = self._resolve_webui_api_session_key(key, request)
        if decoded_key is None:
            return _http_error(404, "session not found")
        try:
            reviewed = self._apply_workplace_review(decoded_key, shot_id, verdict="accept")
        except ValueError as exc:
            return _http_error(400, str(exc))
        if reviewed is None:
            return _http_error(404, "shot not found")
        work_id, workplace = reviewed
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self._publish_workplace_update(decoded_key))
        except RuntimeError:
            pass
        return _http_json_response(
            {
                "ok": True,
                "work_id": work_id,
                "shot_id": shot_id,
                "status": "approved",
                "workplace": workplace,
            }
        )

    def _handle_workplace_shot_accept_all(self, request: WsRequest, key: str) -> Response:
        logger.info("workplace HTTP shots/accept-all session_key={}", key)
        if not self._check_api_token(request):
            return _http_error(401, "Unauthorized")
        legacy_err = self._legacy_webui_session_key_error(key)
        if legacy_err is not None:
            return legacy_err
        decoded_key = self._resolve_webui_api_session_key(key, request)
        if decoded_key is None:
            return _http_error(404, "session not found")
        try:
            work_id, workplace, accepted_shot_ids = self._apply_workplace_accept_all(decoded_key)
        except ValueError as exc:
            return _http_error(400, str(exc))
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self._publish_workplace_update(decoded_key))
        except RuntimeError:
            pass
        return _http_json_response(
            {
                "ok": True,
                "work_id": work_id,
                "accepted_shot_ids": accepted_shot_ids,
                "workplace": workplace,
            }
        )

    def _handle_workplace_shot_merge_up(self, request: WsRequest, key: str, shot_id: int) -> Response:
        logger.info("workplace HTTP merge-up session_key={} shot_id={}", key, shot_id)
        if not self._check_api_token(request):
            return _http_error(401, "Unauthorized")
        legacy_err = self._legacy_webui_session_key_error(key)
        if legacy_err is not None:
            return legacy_err
        decoded_key = self._resolve_webui_api_session_key(key, request)
        if decoded_key is None:
            return _http_error(404, "session not found")
        merged_text = self._parse_merge_shot_body(request)
        try:
            merged = self._apply_workplace_shot_merge_up(
                decoded_key,
                shot_id,
                merged_text=merged_text,
            )
        except ValueError as exc:
            return _http_error(400, str(exc))
        if merged is None:
            return _http_error(404, "beat not found")
        work_id, workplace = merged
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self._publish_workplace_update(decoded_key))
        except RuntimeError:
            pass
        return _http_json_response(
            {
                "ok": True,
                "work_id": work_id,
                "merged_shot_id": shot_id,
                "into_shot_id": shot_id - 1,
                "workplace": workplace,
            }
        )

    def _handle_workplace_shot_remove_shot(self, request: WsRequest, key: str, shot_id: int) -> Response:
        logger.info("workplace HTTP remove-shot session_key={} shot_id={}", key, shot_id)
        if not self._check_api_token(request):
            return _http_error(401, "Unauthorized")
        legacy_err = self._legacy_webui_session_key_error(key)
        if legacy_err is not None:
            return legacy_err
        decoded_key = self._resolve_webui_api_session_key(key, request)
        if decoded_key is None:
            return _http_error(404, "session not found")
        try:
            removed = self._apply_workplace_shot_remove(decoded_key, shot_id)
        except ValueError as exc:
            return _http_error(400, str(exc))
        if removed is None:
            return _http_error(404, "beat not found")
        work_id, workplace = removed
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self._publish_workplace_update(decoded_key))
        except RuntimeError:
            pass
        return _http_json_response(
            {
                "ok": True,
                "work_id": work_id,
                "removed_shot_id": shot_id,
                "workplace": workplace,
            }
        )

    def _handle_workplace_shot_split_shot(self, request: WsRequest, key: str, shot_id: int) -> Response:
        logger.info("workplace HTTP split-shot session_key={} shot_id={}", key, shot_id)
        if not self._check_api_token(request):
            return _http_error(401, "Unauthorized")
        legacy_err = self._legacy_webui_session_key_error(key)
        if legacy_err is not None:
            return legacy_err
        decoded_key = self._resolve_webui_api_session_key(key, request)
        if decoded_key is None:
            return _http_error(404, "session not found")
        split_payload = self._parse_split_shot_body(request)
        try:
            split_result = self._apply_workplace_shot_split_shot(
                decoded_key,
                shot_id,
                split_payload=split_payload,
            )
        except ValueError as exc:
            return _http_error(400, str(exc))
        if split_result is None:
            return _http_error(404, "beat not found")
        work_id, workplace = split_result
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self._publish_workplace_update(decoded_key))
        except RuntimeError:
            pass
        return _http_json_response(
            {
                "ok": True,
                "work_id": work_id,
                "split_shot_id": shot_id,
                "new_shot_id": shot_id + 1,
                "workplace": workplace,
            }
        )

    def _handle_workplace_shot_revise(self, request: WsRequest, key: str, shot_id: int) -> Response:
        if not self._check_api_token(request):
            return _http_error(401, "Unauthorized")
        legacy_err = self._legacy_webui_session_key_error(key)
        if legacy_err is not None:
            return legacy_err
        decoded_key = self._resolve_webui_api_session_key(key, request)
        if decoded_key is None:
            return _http_error(404, "session not found")
        feedback = (_query_first(_parse_query(request.path), "feedback") or "").strip()
        if not feedback:
            return _http_error(400, "feedback required")
        try:
            reviewed = self._apply_workplace_review(decoded_key, shot_id, verdict="revise", feedback=feedback)
        except ValueError as exc:
            return _http_error(400, str(exc))
        if reviewed is None:
            return _http_error(404, "shot not found")
        work_id, workplace = reviewed
        self._schedule_workplace_revision_instruction(
            decoded_key,
            work_id=work_id,
            shot_id=shot_id,
            feedback=feedback,
        )
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self._publish_workplace_update(decoded_key))
        except RuntimeError:
            pass
        return _http_json_response(
            {
                "ok": True,
                "work_id": work_id,
                "shot_id": shot_id,
                "status": "review_fail",
                "feedback": feedback,
                "workplace": workplace,
            }
        )

    def _resolve_workplace_request(
        self,
        request: WsRequest,
        key: str,
    ) -> tuple[str, str] | Response:
        if not self._check_api_token(request):
            return _http_error(401, "Unauthorized")
        legacy_err = self._legacy_webui_session_key_error(key)
        if legacy_err is not None:
            return legacy_err
        decoded_key = self._resolve_webui_api_session_key(key, request)
        if decoded_key is None:
            return _http_error(404, "session not found")
        work_id = self._resolve_work_id_for_session(decoded_key)
        if not work_id:
            return _http_error(404, "director work not found")
        return decoded_key, work_id

    @staticmethod
    def _workplace_story_text(paths: dict[str, Path]) -> str:
        try:
            if paths["story"].exists():
                return paths["story"].read_text(encoding="utf-8")
        except OSError:
            return ""
        return ""

    @staticmethod
    def _write_workplace_story_text(paths: dict[str, Path], story_md: str) -> None:
        paths["story"].parent.mkdir(parents=True, exist_ok=True)
        paths["story"].write_text(story_md, encoding="utf-8")

    @staticmethod
    def _parse_confirm_story_body(request: WsRequest) -> str | None:
        """Return an optional ``story_md`` override from the confirm-story request.

        The websockets HTTP surface only accepts GET without a request body, so the
        WebUI ships JSON ``{"story_md": "..."}`` in ``X-Nanobot-Body`` (base64).
        Direct unit tests may attach a ``body`` attribute to the request object.
        """
        raw_body = getattr(request, "body", None)
        if isinstance(raw_body, (bytes, bytearray)) and raw_body:
            try:
                payload = json.loads(raw_body)
            except (json.JSONDecodeError, UnicodeDecodeError):
                payload = None
            if isinstance(payload, dict):
                value = payload.get("story_md")
                if isinstance(value, str):
                    return value

        encoded = request.headers.get("X-Nanobot-Body")
        if encoded:
            try:
                payload = json.loads(base64.b64decode(encoded, validate=True))
            except (json.JSONDecodeError, UnicodeDecodeError, binascii.Error, ValueError):
                payload = None
            if isinstance(payload, dict):
                value = payload.get("story_md")
                if isinstance(value, str):
                    return value

        value = _query_first(_parse_query(request.path), "story_md")
        if value is not None:
            return value
        return None

    @staticmethod
    def _workplace_shot_entries(state: dict[str, Any]) -> list[dict[str, Any]]:
        shots = state.get("shots")
        if not isinstance(shots, dict):
            return []
        entries = [item for item in shots.values() if isinstance(item, dict)]
        entries.sort(key=lambda item: int(item.get("shot_id") or 0))
        return entries

    def _workplace_disk_shots(self, work_id: str) -> list[dict[str, Any]]:
        paths = self._workplace_paths(work_id)
        if paths is None:
            return []
        shots: list[dict[str, Any]] = []
        for shot_path in sorted(paths["shots"].glob("shot_*.json")):
            shot = self._read_json_file(shot_path, {})
            if isinstance(shot, dict):
                shots.append(shot)
        return shots

    def _workplace_shot_specs_complete_on_disk(
        self,
        work_id: str,
        *,
        beat_count: int = 0,
    ) -> bool:
        shots = self._workplace_disk_shots(work_id)
        if not shots:
            return False
        if beat_count > 0 and len(shots) != beat_count:
            return False
        return all(self._shot_has_caption(shot) for shot in shots)

    def _workplace_shot_prompts_progress(
        self,
        work_id: str,
        *,
        beat_count: int,
    ) -> dict[str, int] | None:
        if beat_count <= 0:
            return None
        ready = sum(
            1
            for shot in self._workplace_disk_shots(work_id)
            if self._shot_has_caption(shot)
        )
        return {"ready": ready, "total": beat_count}

    @staticmethod
    def _workplace_shot_specs_complete(
        state: dict[str, Any],
        *,
        beat_count: int = 0,
    ) -> bool:
        shot_entries = WebSocketChannel._workplace_shot_entries(state)
        if not shot_entries:
            return False
        if beat_count > 0 and len(shot_entries) != beat_count:
            return False
        return all(item.get("has_shot_spec") for item in shot_entries)

    def _workplace_shot_prompts_ready(
        self,
        work_id: str,
        state: dict[str, Any],
        *,
        beat_count: int = 0,
    ) -> bool:
        if not state.get("story_confirmed"):
            return False
        return self._workplace_shot_specs_complete_on_disk(
            work_id,
            beat_count=beat_count,
        )

    def _sync_story_confirmed_when_shot_specs_ready(
        self,
        work_id: str,
        state: dict[str, Any],
        *,
        beat_count: int = 0,
    ) -> None:
        """Heal story_confirmed when shot specs exist but confirm was blocked."""
        if state.get("story_confirmed"):
            return
        if not self._workplace_shot_specs_complete(state, beat_count=beat_count):
            if not self._workplace_shot_specs_complete_on_disk(work_id, beat_count=beat_count):
                return
        state["story_confirmed"] = True
        self._save_workplace_state(work_id, state)

    @staticmethod
    def _workplace_pending_remote_jobs(state: dict[str, Any]) -> dict[str, Any]:
        pending = state.get("pending_remote_jobs")
        return pending if isinstance(pending, dict) else {}

    def _schedule_workplace_workflow_instruction(
        self,
        session_key: str,
        *,
        work_id: str,
        injected_event: str,
        content: str,
        silent: bool = True,
        extra_metadata: dict[str, Any] | None = None,
    ) -> None:
        metadata: dict[str, Any] = {
            "injected_event": injected_event,
            "injected_role": "user",
            "work_id": work_id,
        }
        if silent:
            metadata["silent"] = True
        if extra_metadata:
            metadata.update(extra_metadata)
        msg = InboundMessage(
            channel="system",
            sender_id="workplace",
            chat_id=session_key,
            content=content,
            session_key_override=session_key,
            metadata=metadata,
        )
        scheduled = self._schedule_coro(
            lambda: self.bus.publish_inbound(msg),
            warning=(
                "websocket: unable to schedule workplace workflow instruction "
                f"({injected_event})"
            ),
        )
        if scheduled:
            logger.info(
                "websocket: scheduled workplace workflow instruction event={} session_key={}",
                injected_event,
                session_key,
            )

    def _parse_story_save_body(self, request: WsRequest) -> str | None:
        payload = self._parse_json_body_payload(request)
        if isinstance(payload, dict):
            value = payload.get("story_md")
            if isinstance(value, str):
                return value
        value = _query_first(_parse_query(request.path), "story_md")
        if value is not None:
            return value
        return None

    def _parse_story_profile_save_body(self, request: WsRequest) -> dict[str, Any] | None:
        payload = self._parse_json_body_payload(request)
        if not isinstance(payload, dict):
            return None
        profile = payload.get("story_profile")
        if isinstance(profile, dict):
            return profile
        return payload

    def _schedule_workplace_story_edit_instruction(
        self,
        session_key: str,
        *,
        work_id: str,
    ) -> None:
        content = (
            "Internal workplace workflow task. Execute silently.\n\n"
            f"Director work_id: `{work_id}`\n\n"
            "The user edited `story.md` in the workplace UI.\n"
            "Required actions, in order:\n"
            "1. Call `get_story` to read the latest `story_md` and current `story_profile`.\n"
            "2. Reconcile `story_profile` with the updated screenplay:\n"
            "   - Update `summary` if the story focus changed.\n"
            "   - Rebuild `beats` so each planned shot in `story_md` has exactly one "
            "`{shot_id, summary}` entry (renumber from 1, match the new shot count).\n"
            "   - Remove beats for shots deleted from the screenplay; add beats for new shots.\n"
            "3. If the shot count changed, call `set_director_goal` with the new `shot_count`.\n"
            "4. Call `write_story` with the current `story_md` unchanged, the reconciled "
            "`story_profile`, and `confirmed=false`. Do not rewrite `story.md`.\n"
            "5. Do not call `create_shot_prompt` or `generate_echo_shot` in this turn.\n\n"
            "Strict constraints:\n"
            "- Do not ask the user any question.\n"
            "- Do not send any user-facing message before or after the tool calls.\n"
            "- Do not paste screenplay or profile JSON into chat."
        )
        self._schedule_workplace_workflow_instruction(
            session_key,
            work_id=work_id,
            injected_event="workplace_story_edit",
            content=content,
        )

    def _schedule_workplace_beats_edit_instruction(
        self,
        session_key: str,
        *,
        work_id: str,
    ) -> None:
        content = (
            "Internal workplace workflow task. Execute silently.\n\n"
            f"Director work_id: `{work_id}`\n\n"
            "The user edited `story_profile.beats` in the workplace UI (merge, split, remove, or save).\n"
            "Required actions, in order:\n"
            "1. Call `get_story` to read the latest `story_profile` and current `story_md`.\n"
            "2. Call `set_director_goal` with `shot_count` equal to the number of beats.\n"
            "3. Call `write_story` with the current `story_md` unchanged, the updated "
            "`story_profile` from disk (must keep non-empty `summary` and `beats`), "
            "and `confirmed=true`. Do not rewrite `story.md`.\n"
            "4. Create exactly one `create_shot_prompt` per beat (shot_id 1..N). "
            "Do not create or keep any shot beyond N.\n"
            "5. Do not call `generate_echo_shot` in this turn.\n\n"
            "Strict constraints:\n"
            "- Do not ask the user any question.\n"
            "- Do not send any user-facing message."
        )
        self._schedule_workplace_workflow_instruction(
            session_key,
            work_id=work_id,
            injected_event="workplace_beats_edit",
            content=content,
        )

    def _apply_workplace_story_save(
        self,
        session_key: str,
        story_md: str,
    ) -> tuple[str, dict[str, Any]]:
        work_id = self._resolve_work_id_for_session(session_key)
        if not work_id:
            raise ValueError("director work not found")
        paths = self._workplace_paths(work_id)
        if paths is None:
            raise ValueError("director work not found")
        state = self._load_workplace_state(work_id)
        self._assert_story_discussion_editable(state, field="story.md")
        cleaned = story_md.strip()
        if not cleaned:
            raise ValueError("story_md cannot be empty")
        existing = self._workplace_story_text(paths).strip()
        if cleaned == existing:
            # No changes — skip writing and agent notification.
            workplace = self._build_workplace_payload(session_key)
            return work_id, workplace
        try:
            self._write_workplace_story_text(paths, cleaned)
        except OSError as exc:
            raise ValueError(f"failed to write story: {exc}") from exc
        # Mark that the story was edited and agent has not yet reconciled it.
        state["story_pending_agent_review"] = True
        self._save_workplace_state(work_id, state)
        self._schedule_workplace_story_edit_instruction(session_key, work_id=work_id)
        workplace = self._build_workplace_payload(session_key)
        return work_id, workplace

    def _apply_workplace_story_profile_save(
        self,
        session_key: str,
        incoming: dict[str, Any],
    ) -> tuple[str, dict[str, Any]]:
        work_id = self._resolve_work_id_for_session(session_key)
        if not work_id:
            raise ValueError("director work not found")
        paths = self._workplace_paths(work_id)
        if paths is None:
            raise ValueError("director work not found")
        state = self._load_workplace_state(work_id)
        self._assert_beats_editable(work_id, state)
        profile = self._load_workplace_story_profile(work_id)
        _profile_fields = (
            "summary",
            "beats",
            "characters",
            "genre",
            "setting",
            "title",
            "tone",
            "anchors",
            "shot_to_content",
            "content_to_shots",
            "language",
            "dialogue_language",
        )
        before_snapshot = json.dumps(
            {f: profile.get(f) for f in _profile_fields}, sort_keys=True
        )
        for field in _profile_fields:
            if field in incoming:
                profile[field] = incoming[field]
        after_snapshot = json.dumps(
            {f: profile.get(f) for f in _profile_fields}, sort_keys=True
        )
        if before_snapshot == after_snapshot:
            # No changes — skip writing and agent notification.
            workplace = self._build_workplace_payload(session_key)
            return work_id, workplace
        self._clear_workplace_shot_files(work_id, state)
        self._save_workplace_story_profile(work_id, profile, state)
        self._save_workplace_state(work_id, state)
        self._schedule_workplace_beats_edit_instruction(session_key, work_id=work_id)
        workplace = self._build_workplace_payload(session_key)
        return work_id, workplace

    def _handle_workplace_story_save(self, request: WsRequest, key: str) -> Response:
        logger.info("workplace HTTP story/save session_key={}", key)
        resolved = self._resolve_workplace_request(request, key)
        if not isinstance(resolved, tuple):
            return resolved
        decoded_key, _work_id = resolved
        story_md = self._parse_story_save_body(request)
        if story_md is None:
            return _http_error(400, "story_md is required")
        try:
            work_id, workplace = self._apply_workplace_story_save(decoded_key, story_md)
        except ValueError as exc:
            return _http_error(400, str(exc))
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self._publish_workplace_update(decoded_key))
        except RuntimeError:
            pass
        return _http_json_response(
            {
                "ok": True,
                "action": "save_story",
                "work_id": work_id,
                "workplace": workplace,
            }
        )

    def _handle_workplace_story_profile_save(self, request: WsRequest, key: str) -> Response:
        logger.info("workplace HTTP story-profile/save session_key={}", key)
        resolved = self._resolve_workplace_request(request, key)
        if not isinstance(resolved, tuple):
            return resolved
        decoded_key, _work_id = resolved
        incoming = self._parse_story_profile_save_body(request)
        if not isinstance(incoming, dict):
            return _http_error(400, "story_profile is required")
        try:
            work_id, workplace = self._apply_workplace_story_profile_save(decoded_key, incoming)
        except ValueError as exc:
            return _http_error(400, str(exc))
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self._publish_workplace_update(decoded_key))
        except RuntimeError:
            pass
        return _http_json_response(
            {
                "ok": True,
                "action": "save_story_profile",
                "work_id": work_id,
                "workplace": workplace,
            }
        )

    def _resolve_workplace_session(self, request: WsRequest, key: str) -> str | Response:
        if not self._check_api_token(request):
            return _http_error(401, "Unauthorized")
        legacy_err = self._legacy_webui_session_key_error(key)
        if legacy_err is not None:
            return legacy_err
        decoded_key = self._resolve_webui_api_session_key(key, request)
        if decoded_key is None:
            return _http_error(404, "session not found")
        return decoded_key

    def _reference_image_locked_for_session(self, session_key: str) -> bool:
        if self._session_manager is None:
            return False
        session = self._session_manager.get_or_create(session_key)
        metadata = session.metadata if isinstance(session.metadata, dict) else {}
        if is_reference_image_locked(metadata):
            return True
        work_id = self._resolve_work_id_for_session(session_key)
        if not work_id:
            return False
        state = self._load_workplace_state(work_id)
        return is_reference_image_locked(state)

    def _parse_reference_image_save_body(self, request: WsRequest) -> dict[str, Any] | None:
        payload = self._parse_json_body_payload(request)
        query = _parse_query(request.path)
        merged: dict[str, Any] = dict(payload) if isinstance(payload, dict) else {}
        for key in ("url", "name", "width", "height", "referenceImageUrl", "reference_image_url"):
            value = _query_first(query, key)
            if value is not None and key not in merged:
                merged[key] = value
        return normalize_reference_image(merged)

    def _persist_reference_image(
        self,
        session_key: str,
        ref: dict[str, Any] | None,
    ) -> dict[str, Any]:
        if self._session_manager is None:
            raise ValueError("session manager unavailable")
        session = self._session_manager.get_or_create(session_key)
        if not isinstance(session.metadata, dict):
            session.metadata = {}
        if self._reference_image_locked_for_session(session_key):
            raise PermissionError("不可修改")
        previous_url = reference_image_url(session.metadata.get("reference_image"))
        next_url = reference_image_url(ref) if ref else ""
        if mark_reference_image_needs_story_rewrite(
            session.metadata,
            previous_url=previous_url,
            next_url=next_url,
        ):
            logger.info(
                "reference image changed, story rewrite required session_key={} "
                "previous_url={} next_url={} suppressed={}",
                session_key,
                previous_url or "-",
                next_url or "-",
                story_rewrite_suppressed(session.metadata),
            )
        if ref is None:
            session.metadata.pop("reference_image", None)
        else:
            session.metadata["reference_image"] = ref
        self._session_manager.save(session)
        work_id = self._resolve_work_id_for_session(session_key)
        if work_id:
            state = self._load_workplace_state(work_id)
            if not is_reference_image_locked(state):
                if ref is None:
                    state["reference_image"] = None
                else:
                    state["reference_image"] = ref
                self._save_workplace_state(work_id, state)
        return self._build_workplace_payload(session_key)

    def _reference_image_present_for_session(self, session_key: str) -> bool:
        if self._session_manager is None:
            return False
        session = self._session_manager.get_or_create(session_key)
        metadata = session.metadata if isinstance(session.metadata, dict) else {}
        if reference_image_present(metadata.get("reference_image")):
            return True
        work_id = self._resolve_work_id_for_session(session_key)
        if not work_id:
            return False
        state = self._load_workplace_state(work_id)
        return reference_image_present(state.get("reference_image"))

    def _commit_reference_image_gate(self, session_key: str, answer: str):
        from nanobot.session.reference_image_gate import (
            UploadGateResult,
            commit_upload_gate,
            decline_streak,
            evaluate_upload_gate,
            is_upload_gate_answer,
        )

        if self._session_manager is None or not is_upload_gate_answer(answer):
            return None
        session = self._session_manager.get_or_create(session_key)
        metadata = session.metadata if isinstance(session.metadata, dict) else {}
        if get_auto_generate(metadata) or self._workplace_auto_generate_active(session_key):
            return None
        present = self._reference_image_present_for_session(session_key)
        result: UploadGateResult = evaluate_upload_gate(
            present=present,
            answer=answer,
            streak=decline_streak(metadata),
        )
        commit_upload_gate(metadata, result)
        session.metadata = metadata
        self._session_manager.save(session)
        if result.delete_image:
            try:
                self._persist_reference_image(session_key, None)
            except PermissionError:
                logger.error(
                    "reference image gate could not delete locked image session_key={}",
                    session_key,
                )
            except Exception:
                logger.opt(exception=True).error(
                    "reference image gate delete failed session_key={}",
                    session_key,
                )
        return result

    async def _emit_upload_gate_mismatch_card(
        self,
        session_key: str,
        chat_id: str,
        question: str,
    ) -> None:
        from nanobot.agent.tools.ask_user import normalize_question_cards
        from nanobot.session.question_cards import build_ask_user_session_messages
        from nanobot.session.reference_image_gate import mismatch_card

        if self._session_manager is None:
            return
        cards = normalize_question_cards([mismatch_card(question)])
        if isinstance(cards, str):
            logger.error(
                "reference image gate card invalid session_key={} error={}",
                session_key,
                cards,
            )
            return
        batch_id = str(uuid.uuid4())
        tool_call_id = f"call_{batch_id}"
        session = self._session_manager.get_or_create(session_key)
        session.messages.extend(
            build_ask_user_session_messages(
                tool_call_id=tool_call_id,
                content=question,
                questions=cards,
                batch_id=batch_id,
                channel="websocket",
                chat_id=chat_id,
            )
        )
        session.updated_at = datetime.now()
        self._session_manager.save(session)
        await self.send(
            OutboundMessage(
                channel="websocket",
                chat_id=chat_id,
                content="",
                metadata={
                    "questions": cards,
                    "question_batch_id": batch_id,
                    "session_key": session_key,
                },
            )
        )
        logger.info(
            "reference image gate mismatch card session_key={} question={}",
            session_key,
            question,
        )

    def _persist_gate_user_reply(self, session_key: str, content: str) -> None:
        if self._session_manager is None:
            return
        session = self._session_manager.get_or_create(session_key)
        session.messages.append(
            {
                "role": "user",
                "content": content,
                "timestamp": datetime.now().isoformat(),
            }
        )
        session.updated_at = datetime.now()
        self._session_manager.save(session)

    def _handle_workplace_reference_image_save(self, request: WsRequest, key: str) -> Response:
        logger.info("workplace HTTP reference-image/save session_key={}", key)
        resolved = self._resolve_workplace_session(request, key)
        if not isinstance(resolved, str):
            return resolved
        ref = self._parse_reference_image_save_body(request)
        if ref is None:
            return _http_error(400, "url is required")
        url = str(ref.get("url") or "")
        if is_blocked_local_url(url):
            logger.error(
                "workplace reference-image/save rejected local url session_key={}",
                resolved,
            )
            return _http_error(400, "url must be a public HTTP(S) address")
        try:
            configure_download_policy(self._download_url_policy())
            validate_external_url(url)
        except UrlValidationError as exc:
            logger.error(
                "workplace reference-image/save invalid url session_key={} error={}",
                resolved,
                exc,
            )
            return _http_error(400, str(exc))
        try:
            workplace = self._persist_reference_image(resolved, ref)
        except PermissionError:
            return _http_error(409, "不可修改")
        except ValueError as exc:
            return _http_error(400, str(exc))
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self._publish_workplace_update(resolved))
        except RuntimeError:
            pass
        return _http_json_response(
            {
                "ok": True,
                "action": "save_reference_image",
                "workplace": workplace,
            }
        )

    def _handle_workplace_reference_image_delete(self, request: WsRequest, key: str) -> Response:
        logger.info("workplace HTTP reference-image/delete session_key={}", key)
        resolved = self._resolve_workplace_session(request, key)
        if not isinstance(resolved, str):
            return resolved
        try:
            workplace = self._persist_reference_image(resolved, None)
        except PermissionError:
            return _http_error(409, "不可修改")
        except ValueError as exc:
            return _http_error(400, str(exc))
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self._publish_workplace_update(resolved))
        except RuntimeError:
            pass
        return _http_json_response(
            {
                "ok": True,
                "action": "delete_reference_image",
                "workplace": workplace,
            }
        )

    def _schedule_workplace_confirm_story_instruction(
        self,
        session_key: str,
        *,
        work_id: str,
    ) -> None:
        content = (
            "Internal workplace workflow task. Execute silently.\n\n"
            f"Director work_id: `{work_id}`\n\n"
            "Required actions, in order:\n"
            "1. Call `get_story` and `get_workplace_status(include_shots=true)`.\n"
            "2. If `story.md` is missing or empty, stop after tool inspection only.\n"
            "3. Ensure required goal fields, especially `shot_count`, are written with `set_director_goal` when missing.\n"
            "4. If `story_profile` is missing, incomplete, or has empty `beats`, derive a valid "
            "`story_profile` (non-empty `summary` plus one beat per planned shot) from `story_md` "
            "before confirming.\n"
            "5. Call `write_story` with the current `story_md`, the validated `story_profile`, "
            "and `confirmed=true`.\n"
            "6. Do not call `create_shot_prompt` or `generate_echo_shot` in this turn.\n\n"
            "Strict constraints:\n"
            "- Do not ask the user any question.\n"
            "- Do not acknowledge, summarize, or explain progress to the user.\n"
            "- Do not send any user-facing message before or after the tool calls.\n"
            "- Do not paste screenplay or shot specs into chat."
        )
        self._schedule_workplace_workflow_instruction(
            session_key,
            work_id=work_id,
            injected_event="workplace_workflow_confirm_story",
            content=content,
        )

    def _schedule_workplace_start_merge_instruction(
        self,
        session_key: str,
        *,
        work_id: str,
    ) -> None:
        content = (
            "Internal workplace workflow task. Execute silently.\n\n"
            f"Director work_id: `{work_id}`\n\n"
            "Required actions, in order:\n"
            "1. Call `get_workplace_status(include_shots=true)`.\n"
            "2. Call `merge_shot` with every approved shot ID in timeline order.\n\n"
            "Strict constraints:\n"
            "- Do not ask the user any question.\n"
            "- Do not acknowledge, summarize, or explain progress to the user.\n"
            "- Do not send any user-facing message before or after the tool calls.\n"
            "- The merge tool call is mandatory once all shots are approved."
        )
        self._schedule_workplace_workflow_instruction(
            session_key,
            work_id=work_id,
            injected_event="workplace_workflow_start_merge",
            content=content,
        )

    def _schedule_workplace_start_generation_instruction(
        self,
        session_key: str,
        *,
        work_id: str,
    ) -> None:
        content = (
            "Internal workplace workflow task. Execute silently.\n\n"
            f"Director work_id: `{work_id}`\n\n"
            "Required actions, in order:\n"
            "1. Call `get_workplace_status(include_shots=true)` and `get_story`.\n"
            "2. If the screenplay is not confirmed yet, call `write_story(..., confirmed=true)` first.\n"
            "3. Call `set_director_goal(generation_mode=\"sequential\")` unless generation mode is already set.\n"
            "4. For each outer shot from 1 to the locked `goal.shot_count`, create or update shot prompts with "
            "`create_shot_prompt` when the caption is missing. Use `goal.shot_count` from status, not "
            "session duration defaults.\n"
            "5. Follow `get_guidance(topic=\"shot-prompt-writer\")` while writing shot captions.\n"
            "6. For every shot that has a caption and is not already queued, generated, or approved, call `get_shot` when needed.\n"
            "7. Decide `reference_shot_ids` for each shot.\n"
            "8. Call `set_shot_references` for each ready shot with the chosen references and a short selection note.\n\n"
            "Strict constraints:\n"
            "- Outer film length is the locked `goal.shot_count`. Create exactly that many `create_shot_prompt` calls "
            "(shot_id=1..shot_count). If the user locked 4 shots, write 4 captions.\n"
            "- Caption prefix `本视频包含N个镜头` / `This video has N shots` is INTERNAL segments of THIS 10s clip only. "
            "N must be 2 or 3 (prefer 3). Never set N to the outer shot_count. Never describe the whole film as 3 shots "
            "when outer shot_count is 4.\n"
            "- Do not call `generate_echo_shot` in this turn.\n"
            "- Do not ask the user any question.\n"
            "- Do not acknowledge, summarize, or explain progress to the user.\n"
            "- Do not send any user-facing message before or after the tool calls.\n"
            "- Every ready shot must receive a `set_shot_references` call before you stop."
        )
        self._schedule_workplace_workflow_instruction(
            session_key,
            work_id=work_id,
            injected_event="workplace_workflow_start_generation",
            content=content,
        )

    def _apply_workplace_start_generation(
        self,
        session_key: str,
    ) -> tuple[str, dict[str, Any]]:
        work_id = self._resolve_work_id_for_session(session_key)
        if not work_id:
            raise ValueError("director work not found")
        paths = self._workplace_paths(work_id)
        if paths is None:
            raise ValueError("director work not found")
        state = self._read_json_file(paths["state"], {})
        if not isinstance(state, dict):
            state = {}
        self._reconcile_workplace_state_shots(work_id, state)
        if state.get("final_output_url") or state.get("final_output_path"):
            raise ValueError("work already completed")
        if not state.get("story_confirmed") and not self._try_sync_workplace_story_confirmed(
            work_id,
            paths,
            state,
        ):
            raise ValueError("story is not confirmed; confirm the screenplay first")
        profile = self._load_workplace_story_profile(work_id)
        beats = profile.get("beats") if isinstance(profile.get("beats"), list) else []
        beat_count = len(beats)
        if beat_count > 0:
            self._prune_workplace_shots_beyond_count(work_id, beat_count, state)
            self._reconcile_workplace_state_shots(work_id, state)
        self._sync_story_confirmed_when_shot_specs_ready(
            work_id,
            state,
            beat_count=beat_count,
        )
        if beat_count <= 0:
            raise ValueError("story_profile.beats is empty; confirm the screenplay first")
        pending_kinds = {
            str(item.get("kind"))
            for item in self._workplace_pending_remote_jobs(state).values()
            if isinstance(item, dict)
        }
        if "generate_echo_shot" in pending_kinds:
            raise ValueError("shot generation already in progress")
        if pending_kinds.intersection({"merge_shot"}):
            raise ValueError("merge already in progress")
        stage = str(state.get("stage") or "")
        if stage not in {"merging", "done"}:
            state["stage"] = "shot_generating"
            state["shot_generating_started_at"] = (
                datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
            )
            self._save_workplace_state(work_id, state)
        specs_complete = self._workplace_shot_specs_complete_on_disk(work_id, beat_count=beat_count)
        self._ensure_workplace_shot_references(work_id)
        pending_review = bool(state.get("story_pending_agent_review"))
        has_prior_video = self._workplace_has_generated_video(work_id, state)
        # Fast-path only when specs exist and no replan / prior-generation residue remains.
        if not specs_complete or pending_review or has_prior_video:
            self._schedule_workplace_start_generation_instruction(session_key, work_id=work_id)
        workplace = self._build_workplace_payload(session_key)
        return work_id, workplace

    def _schedule_workplace_abort_generation_stop(self, session_key: str) -> None:
        """Cancel a hung start-generation agent turn without a user-facing chat reply."""
        msg = InboundMessage(
            channel="system",
            sender_id="workplace",
            chat_id=session_key,
            content="/stop",
            session_key_override=session_key,
            metadata={"silent": True},
        )
        scheduled = self._schedule_coro(
            lambda: self.bus.publish_inbound(msg),
            warning=(
                "websocket: unable to schedule abort-generation /stop "
                f"session_key={session_key}"
            ),
        )
        if scheduled:
            logger.info(
                "websocket: scheduled abort-generation /stop session_key={}",
                session_key,
            )

    def _apply_workplace_abort_generation(
        self,
        session_key: str,
    ) -> tuple[str, dict[str, Any]]:
        """Roll shot_generating back to shot_planning so the user can retry start-generation.

        Used when agent prep after start-generation hangs (e.g. model timeout). Not for
        mid-Echo generation — pending generate jobs block this path.
        """
        work_id = self._resolve_work_id_for_session(session_key)
        if not work_id:
            raise ValueError("director work not found")
        paths = self._workplace_paths(work_id)
        if paths is None:
            raise ValueError("director work not found")
        state = self._read_json_file(paths["state"], {})
        if not isinstance(state, dict):
            state = {}
        stage = str(state.get("stage") or "")
        if stage != "shot_generating":
            raise ValueError("abort-generation is only available during shot_generating")
        pending_kinds = {
            str(item.get("kind"))
            for item in self._workplace_pending_remote_jobs(state).values()
            if isinstance(item, dict)
        }
        if pending_kinds.intersection({"generate_echo_shot", "merge_shot"}):
            raise ValueError("shot generation or merge already in progress")
        state["stage"] = "shot_planning"
        state.pop("shot_generating_started_at", None)
        self._save_workplace_state(work_id, state)
        self._schedule_workplace_abort_generation_stop(session_key)
        workplace = self._build_workplace_payload(session_key)
        return work_id, workplace

    def _apply_workplace_confirm_story(
        self,
        session_key: str,
        *,
        story_md: str | None = None,
    ) -> tuple[str, dict[str, Any]]:
        work_id = self._resolve_work_id_for_session(session_key)
        if not work_id:
            raise ValueError("director work not found")
        paths = self._workplace_paths(work_id)
        if paths is None:
            raise ValueError("director work not found")
        existing_story = self._workplace_story_text(paths).strip()
        story_changed = False
        if story_md is not None:
            cleaned = story_md.strip()
            if not cleaned:
                raise ValueError("story_md cannot be empty")
            if cleaned != existing_story:
                try:
                    self._write_workplace_story_text(paths, cleaned)
                except OSError as exc:
                    raise ValueError(f"failed to write story: {exc}") from exc
                story_changed = True
        current_story = self._workplace_story_text(paths).strip()
        if not current_story:
            raise ValueError("story is empty; write a screenplay before confirming")
        state = self._read_json_file(paths["state"], {})
        if not isinstance(state, dict):
            state = {}
        if state.get("final_output_url") or state.get("final_output_path"):
            raise ValueError("work already completed")
        pending_kinds = {
            str(item.get("kind"))
            for item in self._workplace_pending_remote_jobs(state).values()
            if isinstance(item, dict)
        }
        if pending_kinds.intersection({"generate_echo_shot", "merge_shot"}):
            raise ValueError("generation or merge already in progress")
        # Fast-path: if story was not changed *in this request*, the profile is
        # already valid, and no prior save is waiting for agent reconciliation,
        # confirm directly without dispatching an agent turn.
        pending_review = bool(state.get("story_pending_agent_review"))
        if not story_changed and not pending_review:
            profile = self._load_workplace_story_profile(work_id)
            if not _story_profile_validation_error(profile):
                state["story_confirmed"] = True
                if str(state.get("stage") or "") not in {
                    "shot_planning",
                    "shot_generating",
                    "merging",
                    "done",
                }:
                    if locked_shot_count_from_state(state):
                        state["stage"] = "shot_planning"
                    else:
                        state["stage"] = "story_confirmed"
                self._save_workplace_state(work_id, state)
                workplace = self._build_workplace_payload(session_key)
                return work_id, workplace
        self._schedule_workplace_confirm_story_instruction(session_key, work_id=work_id)
        workplace = self._build_workplace_payload(session_key)
        return work_id, workplace

    def _handle_workplace_workflow_confirm_story(self, request: WsRequest, key: str) -> Response:
        resolved = self._resolve_workplace_request(request, key)
        if not isinstance(resolved, tuple):
            return resolved
        decoded_key, _work_id = resolved
        override_md = self._parse_confirm_story_body(request)
        try:
            work_id, workplace = self._apply_workplace_confirm_story(
                decoded_key,
                story_md=override_md,
            )
        except ValueError as exc:
            return _http_error(400, str(exc))
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self._publish_workplace_update(decoded_key))
        except RuntimeError:
            pass
        return _http_json_response(
            {
                "ok": True,
                "action": "confirm_story",
                "work_id": work_id,
                "scheduled": True,
                "workplace": workplace,
            }
        )

    def _handle_workplace_workflow_abort_generation(self, request: WsRequest, key: str) -> Response:
        logger.info("workplace HTTP workflow/abort-generation session_key={}", key)
        resolved = self._resolve_workplace_request(request, key)
        if not isinstance(resolved, tuple):
            return resolved
        decoded_key, _work_id = resolved
        try:
            work_id, workplace = self._apply_workplace_abort_generation(decoded_key)
        except ValueError as exc:
            status = 409 if "already" in str(exc).lower() or "only available" in str(exc).lower() else 400
            return _http_error(status, str(exc))
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self._publish_workplace_update(decoded_key))
        except RuntimeError:
            pass
        return _http_json_response(
            {
                "ok": True,
                "action": "abort_generation",
                "work_id": work_id,
                "workplace": workplace,
            }
        )

    def _handle_workplace_workflow_start_generation(self, request: WsRequest, key: str) -> Response:
        logger.info("workplace HTTP workflow/start-generation session_key={}", key)
        resolved = self._resolve_workplace_request(request, key)
        if not isinstance(resolved, tuple):
            return resolved
        decoded_key, _work_id = resolved
        try:
            work_id, workplace = self._apply_workplace_start_generation(decoded_key)
        except ValueError as exc:
            status = 409 if "already" in str(exc).lower() else 400
            return _http_error(status, str(exc))
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self._publish_workplace_update(decoded_key))
        except RuntimeError:
            pass
        return _http_json_response(
            {
                "ok": True,
                "action": "prepare_generation",
                "work_id": work_id,
                "scheduled": True,
                "workplace": workplace,
            }
        )

    def _handle_workplace_shot_duration(self, request: WsRequest, key: str, shot_id: int) -> Response:
        logger.info("workplace HTTP shot duration session_key={} shot_id={}", key, shot_id)
        if not self._check_api_token(request):
            return _http_error(401, "Unauthorized")
        legacy_err = self._legacy_webui_session_key_error(key)
        if legacy_err is not None:
            return legacy_err
        decoded_key = self._resolve_webui_api_session_key(key, request)
        if decoded_key is None:
            return _http_error(404, "session not found")
        duration_sec = self._parse_shot_duration_body(request)
        if duration_sec is None:
            return _http_error(400, "duration_sec is required")
        try:
            work_id, workplace = self._apply_workplace_shot_duration(
                decoded_key,
                shot_id,
                duration_sec,
            )
        except ValueError as exc:
            return _http_error(400, str(exc))
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self._publish_workplace_update(decoded_key))
        except RuntimeError:
            pass
        return _http_json_response(
            {
                "ok": True,
                "work_id": work_id,
                "shot_id": shot_id,
                "duration_sec": self._clamp_shot_duration_sec(duration_sec),
                "workplace": workplace,
            }
        )

    async def _handle_workplace_shot_generate(
        self, request: WsRequest, key: str, shot_id: int
    ) -> Response:
        logger.info("workplace HTTP shot generate session_key={} shot_id={}", key, shot_id)
        if not self._check_api_token(request):
            return _http_error(401, "Unauthorized")
        legacy_err = self._legacy_webui_session_key_error(key)
        if legacy_err is not None:
            return legacy_err
        decoded_key = self._resolve_webui_api_session_key(key, request)
        if decoded_key is None:
            return _http_error(404, "session not found")
        body = self._parse_json_body_payload(request)
        if isinstance(body, dict) and any(
            body.get(key)
            for key in (
                "reference_image_url",
                "referenceImageUrl",
                "reference_image_name",
                "referenceImageName",
            )
        ):
            logger.info(
                "workplace shot generate: ignoring request-body reference_image_* "
                "session_key={} shot_id={}",
                decoded_key,
                shot_id,
            )
        first_frame_url = None
        work_id = self._resolve_work_id_for_session(decoded_key)
        if shot_id == 1 and work_id:
            first_frame_url = self._workplace_first_frame_url(work_id)
        try:
            if first_frame_url and shot_id == 1:
                if self._provider is None or not self._model:
                    return _http_error(503, "re-caption model unavailable")
                work_id, workplace = self._prepare_workplace_shot_generate_async(
                    decoded_key,
                    shot_id,
                )
                try:
                    loop = asyncio.get_running_loop()
                    loop.create_task(
                        self._complete_workplace_shot_generate_with_reference(
                            decoded_key,
                            work_id=work_id,
                            shot_id=shot_id,
                            reference_image_url=first_frame_url,
                        )
                    )
                    loop.create_task(self._publish_workplace_update(decoded_key))
                except RuntimeError:
                    return _http_error(503, "event loop unavailable")
                return _http_json_response(
                    {
                        "ok": True,
                        "action": "generate_shot",
                        "work_id": work_id,
                        "shot_id": shot_id,
                        "status": "recaptioning",
                        "workplace": workplace,
                    }
                )
            work_id, workplace = await asyncio.to_thread(
                self._apply_workplace_shot_generate,
                decoded_key,
                shot_id,
            )
        except (EchoGeneratorBusyError, EchoGeneratorUnavailableError) as exc:
            return self._http_echo_gate_error(decoded_key, exc, shot_id=shot_id)
        except ValueError as exc:
            status = 409 if "already" in str(exc).lower() else 400
            return _http_error(status, str(exc))
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self._publish_workplace_update(decoded_key))
        except RuntimeError:
            pass
        return _http_json_response(
            {
                "ok": True,
                "action": "generate_shot",
                "work_id": work_id,
                "shot_id": shot_id,
                "workplace": workplace,
            }
        )

    def _apply_workplace_shot_continuous_mode(
        self,
        session_key: str,
        shot_id: int,
        enabled: bool,
    ) -> tuple[str, dict[str, Any]]:
        loaded = self._load_workplace_shot(session_key, shot_id)
        if loaded is None:
            raise ValueError("shot not found")
        work_id, shot_path, state, shot = loaded
        if shot_id <= 1 and enabled:
            raise ValueError("shot 1 cannot enable continuous mode")
        shot["continuous_enabled"] = bool(enabled)
        logger.info(
            "shot continuous_enabled set: work_id={} shot_id={} enabled={}",
            work_id,
            shot_id,
            enabled,
        )
        self._save_workplace_shot(shot_path, shot)
        self._sync_state_shot_row(state, shot)

        # Propagate the default to all subsequent shots so the user doesn't
        # have to toggle every shot manually.  Each shot's own toggle stays
        # independent and can still be changed afterwards.
        next_id = shot_id + 1
        while True:
            next_loaded = self._load_workplace_shot(session_key, next_id)
            if next_loaded is None:
                break
            _nw, next_path, _ns, next_shot = next_loaded
            next_shot["continuous_enabled"] = bool(enabled)
            logger.info(
                "shot continuous_enabled cascaded: work_id={} shot_id={} enabled={}",
                work_id,
                next_id,
                enabled,
            )
            self._save_workplace_shot(next_path, next_shot)
            self._sync_state_shot_row(state, next_shot)
            next_id += 1

        self._save_workplace_state(work_id, state)
        return work_id, self._build_workplace_payload(session_key)

    def _handle_workplace_shot_continuous_mode(
        self, request: WsRequest, key: str, shot_id: int
    ) -> Response:
        logger.info(
            "workplace HTTP shot continuous-mode session_key={} shot_id={}", key, shot_id
        )
        if not self._check_api_token(request):
            return _http_error(401, "Unauthorized")
        legacy_err = self._legacy_webui_session_key_error(key)
        if legacy_err is not None:
            return legacy_err
        decoded_key = self._resolve_webui_api_session_key(key, request)
        if decoded_key is None:
            return _http_error(404, "session not found")
        enabled_raw = str(
            _query_first(_parse_query(request.path), "enabled") or ""
        ).lower()
        if enabled_raw not in {"0", "1", "true", "false", "yes", "no", "on", "off"}:
            return _http_error(400, "enabled is required")
        enabled = enabled_raw in {"1", "true", "yes", "on"}
        try:
            work_id, workplace = self._apply_workplace_shot_continuous_mode(
                decoded_key,
                shot_id,
                enabled,
            )
        except ValueError as exc:
            return _http_error(400, str(exc))
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self._publish_workplace_update(decoded_key))
        except RuntimeError:
            pass
        return _http_json_response(
            {
                "ok": True,
                "action": "continuous_mode",
                "work_id": work_id,
                "shot_id": shot_id,
                "enabled": enabled,
                "workplace": workplace,
            }
        )

    def _handle_workplace_shot_continuous_generate(
        self, request: WsRequest, key: str, shot_id: int
    ) -> Response:
        logger.info(
            "workplace HTTP shot continuous-generate session_key={} shot_id={}", key, shot_id
        )
        if not self._check_api_token(request):
            return _http_error(401, "Unauthorized")
        legacy_err = self._legacy_webui_session_key_error(key)
        if legacy_err is not None:
            return legacy_err
        decoded_key = self._resolve_webui_api_session_key(key, request)
        if decoded_key is None:
            return _http_error(404, "session not found")
        try:
            (
                work_id,
                workplace,
                previous_shot_id,
                video_url,
                reference_shot_ids,
                selection_note,
            ) = self._prepare_workplace_shot_continuous_generate(
                decoded_key, shot_id
            )
        except (EchoGeneratorBusyError, EchoGeneratorUnavailableError) as exc:
            return self._http_echo_gate_error(decoded_key, exc, shot_id=shot_id)
        except ValueError as exc:
            status = 409 if "already" in str(exc).lower() else 400
            return _http_error(status, str(exc))
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(
                self._complete_workplace_shot_continuous_generate(
                    decoded_key,
                    work_id=work_id,
                    shot_id=shot_id,
                    previous_shot_id=previous_shot_id,
                    video_url=video_url,
                    reference_shot_ids=reference_shot_ids,
                    selection_note=selection_note,
                )
            )
            loop.create_task(self._publish_workplace_update(decoded_key))
        except RuntimeError:
            return _http_error(503, "event loop unavailable")
        return _http_json_response(
            {
                "ok": True,
                "action": "continuous_generate",
                "work_id": work_id,
                "shot_id": shot_id,
                "status": "i2v_preparing",
                "workplace": workplace,
            }
        )

    def _prepare_workplace_shot_continuous_generate(
        self,
        session_key: str,
        shot_id: int,
    ) -> tuple[str, dict[str, Any], int, str, list[int], str | None]:
        """Validate a continuation and mark it queued before async VLM work."""
        self._ensure_echo_admission(operation="generate_echo_shot")
        if shot_id <= 1:
            raise ValueError("continuous generation is only available for shot_id > 1")

        loaded = self._load_workplace_shot(session_key, shot_id)
        if loaded is None:
            raise ValueError("shot not found")
        work_id, shot_path, _state, shot = loaded
        if not shot.get("continuous_enabled"):
            raise ValueError(
                f"shot {shot_id} does not have continuous mode enabled; "
                "set continuous_enabled first via /continuous-mode"
            )
        self._lock_workplace_video_size(session_key, work_id)

        status = str(shot.get("status") or "")
        if status == "queued":
            raise ValueError(f"shot {shot_id} generation already in progress")
        if status in {"generated", "review_pass", "approved"}:
            raise ValueError(f"shot {shot_id} is already generated")

        previous_shot_id = shot_id - 1
        prev_loaded = self._load_workplace_shot(session_key, previous_shot_id)
        if prev_loaded is None:
            raise ValueError(f"previous shot {previous_shot_id} not found")
        _prev_work_id, _prev_path, _prev_state, prev_shot = prev_loaded
        prev_status = str(prev_shot.get("status") or "")
        if prev_status not in {"generated", "review_pass", "approved"}:
            raise ValueError(
                f"previous shot {previous_shot_id} must be generated first "
                f"(current: {prev_status})"
            )
        video_url = prev_shot.get("artifact_url") or (
            prev_shot.get("echo") if isinstance(prev_shot.get("echo"), dict) else {}
        ).get("result_url")
        if not isinstance(video_url, str) or not video_url.strip():
            raise ValueError(
                f"previous shot {previous_shot_id} has no artifact_url"
            )

        self._ensure_workplace_shot_references(work_id)
        if "planned_reference_shot_ids" not in shot:
            shot = self._read_json_file(shot_path, {})
        if "planned_reference_shot_ids" not in shot:
            raise ValueError(
                "reference plan is not ready; complete the previous workflow step first"
            )
        reference_shot_ids = self._planned_reference_shot_ids(shot)
        if previous_shot_id not in reference_shot_ids:
            reference_shot_ids = sorted(set(reference_shot_ids) | {previous_shot_id})
        missing = self._missing_reference_generations(work_id, reference_shot_ids)
        if missing:
            raise ValueError(self._format_reference_dependency_error(shot_id, missing))

        selection_note = shot.get("reference_selection_note")
        note = selection_note if isinstance(selection_note, str) else None
        shot["status"] = "queued"
        shot.pop("generation_error", None)
        self._save_workplace_shot(shot_path, shot)
        state = self._load_workplace_state(work_id)
        state["stage"] = "shot_generating"
        self._sync_state_shot_row(state, shot)
        self._save_workplace_state(work_id, state)
        return (
            work_id,
            self._build_workplace_payload(session_key),
            previous_shot_id,
            video_url.strip(),
            reference_shot_ids,
            note,
        )

    async def _complete_workplace_shot_continuous_generate(
        self,
        session_key: str,
        *,
        work_id: str,
        shot_id: int,
        previous_shot_id: int,
        video_url: str,
        reference_shot_ids: list[int],
        selection_note: str | None,
    ) -> None:
        """Extract the tail, rewrite the caption with the image, then submit I2V."""
        try:
            condition_image_url = await asyncio.to_thread(
                self._extract_and_publish_tail_frame,
                work_id,
                previous_shot_id,
                video_url,
            )
            previous_loaded = self._load_workplace_shot(session_key, previous_shot_id)
            if previous_loaded is not None:
                _prev_work_id, prev_path, _prev_state, previous_shot = previous_loaded
                previous_shot["tail_frame_url"] = condition_image_url
                self._save_workplace_shot(prev_path, previous_shot)

            if not condition_image_url:
                raise ValueError(
                    f"failed to extract tail frame for shot {previous_shot_id}"
                )

            loaded = self._load_workplace_shot(session_key, shot_id)
            if loaded is None:
                raise ValueError("shot not found")
            _work_id, _shot_path, _state, shot = loaded
            original_caption = str(shot.get("caption") or "").strip()
            if not original_caption:
                raise ValueError(f"shot {shot_id} has no caption")
            story_profile = self._load_workplace_story_profile(work_id)
            rewritten_prompt = await self._rewrite_i2v_prompt_with_image(
                original_caption,
                condition_image_url,
                story_profile,
            )

            generate_tool = self._director_generate_tool()
            generate_tool.set_context(
                "websocket",
                webui_wire_chat_id(session_key) or "direct",
                effective_key=session_key,
            )
            await asyncio.to_thread(
                generate_tool.apply_generate_continuous,
                work_id,
                shot_id,
                condition_image_url,
                reference_shot_ids,
                selection_note=selection_note,
                i2v_prompt=rewritten_prompt,
            )
            logger.info(
                "workplace I2V continuation submitted work_id={} shot_id={} previous_shot_id={}",
                work_id,
                shot_id,
                previous_shot_id,
            )
        except Exception as exc:
            logger.opt(exception=True).error(
                "workplace I2V continuation failed work_id={} shot_id={}",
                work_id,
                shot_id,
            )
            try:
                paths = self._workplace_paths(work_id)
                if paths is not None:
                    failed = self._read_json_file(
                        paths["shots"] / f"shot_{shot_id:03d}.json", {}
                    )
                    if isinstance(failed, dict):
                        failed["status"] = "error"
                        failed["generation_error"] = str(exc)[:1000]
                        self._save_workplace_shot(
                            paths["shots"] / f"shot_{shot_id:03d}.json",
                            failed,
                        )
                    state = self._load_workplace_state(work_id)
                    state["stage"] = "shot_revising"
                    state["generation_error"] = str(exc)[:1000]
                    self._save_workplace_state(work_id, state)
            except Exception:
                logger.opt(exception=True).error(
                    "failed to persist workplace I2V error work_id={} shot_id={}",
                    work_id,
                )
        finally:
            try:
                await self._publish_workplace_update(session_key)
            except Exception:
                logger.opt(exception=True).error(
                    "workplace I2V update failed work_id={} shot_id={}",
                    work_id,
                )

    def _apply_workplace_shot_continuous_generate(
        self,
        session_key: str,
        shot_id: int,
    ) -> tuple[str, dict[str, Any]]:
        """Submit an I2V continuous generation using the previous shot's tail frame."""
        self._ensure_echo_admission(operation="generate_echo_shot")

        if shot_id <= 1:
            raise ValueError("continuous generation is only available for shot_id > 1")

        loaded = self._load_workplace_shot(session_key, shot_id)
        if loaded is None:
            raise ValueError("shot not found")
        work_id, _shot_path, _state, shot = loaded

        if not shot.get("continuous_enabled"):
            raise ValueError(
                f"shot {shot_id} does not have continuous mode enabled; "
                "set continuous_enabled first via /continuous-mode"
            )

        self._lock_workplace_video_size(session_key, work_id)

        status = str(shot.get("status") or "")
        if status == "queued":
            raise ValueError(f"shot {shot_id} generation already in progress")
        if status in {"generated", "review_pass", "approved"}:
            raise ValueError(f"shot {shot_id} is already generated")

        # Load the previous shot and verify it has a usable artifact_url.
        previous_shot_id = shot_id - 1
        prev_loaded = self._load_workplace_shot(session_key, previous_shot_id)
        if prev_loaded is None:
            raise ValueError(f"previous shot {previous_shot_id} not found")
        _prev_work_id, prev_shot_path, _prev_state, prev_shot = prev_loaded

        prev_status = str(prev_shot.get("status") or "")
        if prev_status not in {"generated", "review_pass", "approved"}:
            raise ValueError(
                f"previous shot {previous_shot_id} must be generated first (current: {prev_status})"
            )

        video_url = prev_shot.get("artifact_url") or (
            prev_shot.get("echo") or {}
        ).get("result_url")
        if not video_url:
            raise ValueError(f"previous shot {previous_shot_id} has no artifact_url")

        # Extract tail frame from the previous shot's video and publish locally.
        logger.info(
            "continuous-generate: shot_id={} using previous shot {} tail frame, "
            "extracting from video_url={}",
            shot_id, previous_shot_id, video_url,
        )
        condition_image_url = self._extract_and_publish_tail_frame(
            work_id, previous_shot_id, video_url
        )
        if not condition_image_url:
            raise ValueError(
                f"failed to extract tail frame for shot {previous_shot_id}"
            )
        logger.info(
            "continuous-generate: tail frame ready, shot_id={} condition_image_url={}",
            shot_id, condition_image_url,
        )

        # Persist tail_frame_url on previous shot for caching and UI display.
        prev_shot["tail_frame_url"] = condition_image_url
        self._save_workplace_shot(prev_shot_path, prev_shot)

        self._ensure_workplace_shot_references(work_id)
        if "planned_reference_shot_ids" not in shot:
            shot = self._read_json_file(_shot_path, {})
        if "planned_reference_shot_ids" not in shot:
            raise ValueError(
                "reference plan is not ready; complete the previous workflow step first"
            )
        reference_shot_ids = self._planned_reference_shot_ids(shot)
        # Ensure the previous shot is included as a reference.
        if previous_shot_id not in reference_shot_ids:
            reference_shot_ids = sorted(set(reference_shot_ids) | {previous_shot_id})
        missing = self._missing_reference_generations(work_id, reference_shot_ids)
        if missing:
            raise ValueError(self._format_reference_dependency_error(shot_id, missing))

        selection_note = shot.get("reference_selection_note")
        note = selection_note if isinstance(selection_note, str) else None
        generate_tool = self._director_generate_tool()
        generate_tool.set_context(
            "websocket",
            webui_wire_chat_id(session_key) or "direct",
            effective_key=session_key,
        )
        generate_tool.apply_generate_continuous(
            work_id,
            shot_id,
            condition_image_url,
            reference_shot_ids,
            selection_note=note,
        )
        return work_id, self._build_workplace_payload(session_key)

    async def _handle_workplace_workflow_generate_all(
        self, request: WsRequest, key: str
    ) -> Response:
        logger.info("workplace HTTP workflow/generate-all session_key={}", key)
        if not self._check_api_token(request):
            return _http_error(401, "Unauthorized")
        legacy_err = self._legacy_webui_session_key_error(key)
        if legacy_err is not None:
            return legacy_err
        decoded_key = self._resolve_webui_api_session_key(key, request)
        if decoded_key is None:
            return _http_error(404, "session not found")
        try:
            work_id, workplace, submitted = await asyncio.to_thread(
                self._apply_workplace_generate_all,
                decoded_key,
            )
        except (EchoGeneratorBusyError, EchoGeneratorUnavailableError) as exc:
            return self._http_echo_gate_error(decoded_key, exc)
        except ValueError as exc:
            status = 409 if "already" in str(exc).lower() else 400
            return _http_error(status, str(exc))
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self._publish_workplace_update(decoded_key))
        except RuntimeError:
            pass
        return _http_json_response(
            {
                "ok": True,
                "action": "generate_all",
                "work_id": work_id,
                "submitted_shot_ids": submitted,
                "workplace": workplace,
            }
        )

    def _handle_workplace_workflow_auto_generate(self, request: WsRequest, key: str) -> Response:
        logger.info("workplace HTTP workflow/auto-generate session_key={}", key)
        resolved = self._resolve_workplace_request(request, key)
        if not isinstance(resolved, tuple):
            return resolved
        decoded_key, _work_id = resolved
        duration_raw = _query_first(_parse_query(request.path), "duration_sec")
        body = self._parse_json_body_payload(request)
        if duration_raw is None and isinstance(body, dict):
            duration_raw = body.get("duration_sec") or body.get("durationSec")
        try:
            work_id, workplace = self._apply_workplace_auto_generate(
                decoded_key,
                duration_sec=duration_raw,
            )
        except (EchoGeneratorBusyError, EchoGeneratorUnavailableError) as exc:
            return self._http_echo_gate_error(decoded_key, exc)
        except ValueError as exc:
            logger.error(
                "workplace auto-generate failed session_key={} error={}",
                decoded_key,
                exc,
            )
            status = 409 if "already" in str(exc).lower() else 400
            return _http_error(status, str(exc))
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self._publish_workplace_update(decoded_key))
        except RuntimeError:
            pass
        return _http_json_response(
            {
                "ok": True,
                "action": "auto_generate",
                "work_id": work_id,
                "workplace": workplace,
            }
        )

    def _sync_auto_generate_n_shots(
        self,
        metadata: dict[str, Any],
        *,
        duration_sec: Any = None,
        locked_shot_count: int | None = None,
    ) -> None:
        """Persist generation shot count only while the workplace is unlocked."""
        from nanobot.session.generation_settings import SESSION_NSHOT_KEY, apply_generation_settings

        if locked_shot_count:
            return
        if duration_sec not in (None, ""):
            n_shots = shot_count_for_auto_generate(duration_sec)
            apply_generation_settings(
                metadata,
                n_shots=n_shots,
                duration_sec=int(duration_sec),
            )
            return
        if SESSION_NSHOT_KEY not in metadata:
            apply_generation_settings(
                metadata,
                n_shots=shot_count_for_auto_generate(DEFAULT_AUTO_GENERATE_DURATION_SEC),
                duration_sec=DEFAULT_AUTO_GENERATE_DURATION_SEC,
            )

    def _apply_workplace_auto_generate(
        self,
        session_key: str,
        *,
        duration_sec: Any = None,
    ) -> tuple[str, dict[str, Any]]:
        if self._session_manager is None:
            raise ValueError("session manager unavailable")

        work_id = self._resolve_work_id_for_session(session_key)
        if not work_id:
            raise ValueError("director work not found")
        state = self._load_workplace_state(work_id)
        locked_n = locked_shot_count_from_state(state)

        session = self._session_manager.get_or_create(session_key)
        apply_auto_generate(session.metadata, True)
        try:
            self._sync_auto_generate_n_shots(
                session.metadata,
                duration_sec=duration_sec,
                locked_shot_count=locked_n,
            )
        except ValueError as exc:
            logger.error(
                "workplace auto-generate invalid duration_sec={} session_key={} error={}",
                duration_sec,
                session_key,
                exc,
            )
            raise
        self._session_manager.save(session)
        state["auto_generate"] = True
        self._save_workplace_state(work_id, state)
        self._disable_auto_generate_continuous(session_key, work_id)
        profile = self._load_workplace_story_profile(work_id)
        beats = profile.get("beats") if isinstance(profile.get("beats"), list) else []
        beat_count = len(beats)
        specs_ready = self._workplace_shot_specs_complete_on_disk(
            work_id, beat_count=beat_count
        )
        refs_ready = self._workplace_references_ready(work_id, beat_count=beat_count)
        if specs_ready and refs_ready:
            disk_shots = self._workplace_disk_shots(work_id)
            if any(str(shot.get("status") or "") == "queued" for shot in disk_shots):
                self._continue_auto_generate(session_key)
                return work_id, self._build_workplace_payload(session_key)
            try:
                work_id, workplace, _submitted = self._apply_workplace_generate_all(session_key)
                return work_id, workplace
            except EchoGeneratorUnavailableError:
                raise
            except ValueError as exc:
                logger.info(
                    "auto_generate start generate_all skipped session_key={} work_id={} error={}",
                    session_key,
                    work_id,
                    exc,
                )
                self._continue_auto_generate(session_key)
                return work_id, self._build_workplace_payload(session_key)
        return self._apply_workplace_start_generation(session_key)

    def _workplace_auto_generate_active(
        self, session_key: str, state: dict[str, Any] | None = None
    ) -> bool:
        if isinstance(state, dict) and bool(state.get("auto_generate")):
            return True
        work_id = self._resolve_work_id_for_session(session_key)
        if work_id:
            loaded = state if isinstance(state, dict) else self._load_workplace_state(work_id)
            if bool(loaded.get("auto_generate")):
                return True
        return self._session_auto_generate_flag(session_key)

    def _disable_auto_generate_continuous(self, session_key: str, work_id: str) -> None:
        """Disable tail-frame continuation while auto-generation is active."""
        state = self._load_workplace_state(work_id)
        changed = False
        for shot_id, shot in self._iter_workplace_shots(work_id):
            if not shot.get("continuous_enabled"):
                continue
            loaded = self._load_workplace_shot(session_key, shot_id)
            if loaded is None:
                continue
            _work, shot_path, _state, current = loaded
            current["continuous_enabled"] = False
            self._save_workplace_shot(shot_path, current)
            self._sync_state_shot_row(state, current)
            changed = True
            logger.info(
                "auto_generate disabled continuous session_key={} work_id={} shot_id={}",
                session_key,
                work_id,
                shot_id,
            )
        if changed:
            self._save_workplace_state(work_id, state)

    def _clear_auto_generate_memory_wait(self, work_id: str) -> None:
        state = self._load_workplace_state(work_id)
        if not state.get("auto_generate_waited_memory"):
            return
        state["auto_generate_waited_memory"] = False
        self._save_workplace_state(work_id, state)

    def _auto_generate_memory_status(self, shot: dict[str, Any]) -> str:
        review = shot.get("memory_review")
        if not isinstance(review, dict):
            return ""
        return str(review.get("status") or "")

    def _auto_generate_shot_ready(self, shot: dict[str, Any]) -> bool:
        if str(shot.get("status") or "") != "approved":
            return False
        if not self._memory_review_workflow_enabled():
            return True
        status = self._auto_generate_memory_status(shot)
        if status == "approved":
            return True
        return False

    def _schedule_auto_generate_continue(self, session_key: str) -> None:
        """Continue auto-generation off the outbound send path.

        ``_continue_auto_generate`` may block on urllib to Echo. Running it
        inside ``send()`` freezes the gateway event loop when the video
        service is down.
        """
        if not session_key or self._session_manager is None:
            return
        if session_key in self._auto_generate_inflight:
            return
        if not self._workplace_auto_generate_active(session_key):
            return
        self._auto_generate_inflight.add(session_key)

        def _run() -> None:
            try:
                self._continue_auto_generate(session_key)
            except Exception:
                logger.opt(exception=True).error(
                    "auto_generate continue failed session_key={}",
                    session_key,
                )

        async def _wrapped() -> None:
            try:
                await asyncio.to_thread(_run)
            finally:
                self._auto_generate_inflight.discard(session_key)

        if not self._schedule_coro(
            lambda: _wrapped(),
            warning=(
                "websocket: unable to schedule auto_generate continue "
                f"session_key={session_key}"
            ),
        ):
            try:
                _run()
            finally:
                self._auto_generate_inflight.discard(session_key)

    def _continue_auto_generate(self, session_key: str) -> None:
        session = self._session_manager.get_or_create(session_key)
        metadata = session.metadata if isinstance(session.metadata, dict) else {}
        work_id = self._resolve_work_id_for_session(session_key)
        if not work_id:
            return
        state = self._load_workplace_state(work_id)
        auto = bool(state.get("auto_generate")) or get_auto_generate(metadata)
        if not auto:
            return
        if not bool(state.get("auto_generate")):
            state["auto_generate"] = True
            self._save_workplace_state(work_id, state)
        self._disable_auto_generate_continuous(session_key, work_id)
        if state.get("final_output_url") or state.get("final_output_path"):
            return
        if str(state.get("stage") or "") == "merging":
            logger.info(
                "auto_generate waiting for merge session_key={} work_id={}",
                session_key,
                work_id,
            )
            return
        goal = state.get("goal") if isinstance(state.get("goal"), dict) else {}
        try:
            locked_shot_count = int(goal.get("shot_count") or 0)
        except (TypeError, ValueError):
            locked_shot_count = 0
        self._reconcile_workplace_state_shots(work_id, state)
        pending_kinds = {
            str(item.get("kind"))
            for item in self._workplace_pending_remote_jobs(state).values()
            if isinstance(item, dict)
        }
        if pending_kinds.intersection({"generate_echo_shot", "merge_shot"}):
            logger.info(
                "auto_generate waiting for pending jobs session_key={} work_id={} kinds={}",
                session_key,
                work_id,
                sorted(pending_kinds),
            )
            return
        disk_shots = self._workplace_disk_shots(work_id)
        if any(str(shot.get("status") or "") == "queued" for shot in disk_shots):
            logger.info(
                "auto_generate waiting for queued shot session_key={} work_id={}",
                session_key,
                work_id,
            )
            return
        retry_count = int(state.get("auto_generate_retry_count") or 0)
        for shot in disk_shots:
            status = str(shot.get("status") or "")
            shot_id = int(shot.get("shot_id") or 0)
            if shot_id <= 0:
                continue
            if status == "error":
                generation_error = str(shot.get("generation_error") or "")
                logger.error(
                    "auto_generate shot error session_key={} work_id={} shot_id={} error={}",
                    session_key,
                    work_id,
                    shot_id,
                    generation_error or "unknown",
                )
                if UNAVAILABLE_MESSAGE in generation_error:
                    return
                if retry_count < 1:
                    state["auto_generate_retry_count"] = retry_count + 1
                    self._save_workplace_state(work_id, state)
                    try:
                        self._apply_workplace_shot_generate(session_key, shot_id)
                    except EchoGeneratorUnavailableError as exc:
                        self._report_echo_unavailable(
                            session_key, exc, work_id=work_id, shot_id=shot_id
                        )
                    except Exception:
                        logger.opt(exception=True).error(
                            "auto_generate retry failed session_key={} work_id={} shot_id={}",
                            session_key,
                            work_id,
                            shot_id,
                        )
                    return
                state["generation_error"] = generation_error or "镜头生成失败"
                self._save_workplace_state(work_id, state)
                return
            if status == "generated":
                try:
                    self._apply_workplace_review(
                        session_key,
                        shot_id,
                        verdict="accept",
                        review_source="auto",
                    )
                except ValueError as exc:
                    logger.error(
                        "auto_generate accept failed session_key={} work_id={} "
                        "shot_id={} error={}",
                        session_key,
                        work_id,
                        shot_id,
                        exc,
                    )
                    return
                self._auto_approve_memory_review_if_needed(session_key, shot_id)

        state = self._load_workplace_state(work_id)
        self._reconcile_workplace_state_shots(work_id, state)
        disk_shots = self._workplace_disk_shots(work_id)
        memory_busy = False
        approved_memory = False
        for shot in disk_shots:
            shot_id = int(shot.get("shot_id") or 0)
            if shot_id <= 0:
                continue
            memory_status = self._auto_generate_memory_status(shot)
            if memory_status == "awaiting_review":
                self._auto_approve_memory_review_if_needed(session_key, shot_id)
                approved_memory = True
                continue
            if memory_status in {"selecting", "reselecting"}:
                memory_busy = True
                continue
            video_status = str(shot.get("status") or "")
            if (
                self._memory_review_workflow_enabled()
                and video_status in {"generated", "review_pass", "approved"}
                and not memory_status
            ):
                memory_busy = True

        if approved_memory:
            state = self._load_workplace_state(work_id)
            state["auto_generate_waited_memory"] = False
            self._save_workplace_state(work_id, state)
        if memory_busy:
            state = self._load_workplace_state(work_id)
            state["auto_generate_waited_memory"] = True
            self._save_workplace_state(work_id, state)
            logger.info(
                "auto_generate waiting for memory session_key={} work_id={}",
                session_key,
                work_id,
            )
            return

        state = self._load_workplace_state(work_id)
        self._reconcile_workplace_state_shots(work_id, state)
        pending_kinds = {
            str(item.get("kind"))
            for item in self._workplace_pending_remote_jobs(state).values()
            if isinstance(item, dict)
        }
        if pending_kinds.intersection({"generate_echo_shot", "merge_shot"}):
            logger.info(
                "auto_generate waiting for pending jobs session_key={} work_id={} kinds={}",
                session_key,
                work_id,
                sorted(pending_kinds),
            )
            return
        disk_shots = self._workplace_disk_shots(work_id)
        if any(str(shot.get("status") or "") == "queued" for shot in disk_shots):
            logger.info(
                "auto_generate waiting for queued shot session_key={} work_id={}",
                session_key,
                work_id,
            )
            return
        if disk_shots and all(
            self._auto_generate_shot_ready(shot)
            for shot in disk_shots
        ):
            self._schedule_auto_generate_merge(session_key, work_id, state)
            return
        if locked_shot_count <= 0:
            return
        predecessors_ready = True
        generated_incomplete = False
        for shot in disk_shots:
            status = str(shot.get("status") or "")
            if status in {"queued", "generated", "review_pass"}:
                generated_incomplete = True
            if status in {"queued", "generated", "review_pass", "approved"} and not self._auto_generate_shot_ready(shot):
                predecessors_ready = False
        if generated_incomplete or not predecessors_ready:
            logger.info(
                "auto_generate holding generate_all session_key={} work_id={} "
                "generated_incomplete={} predecessors_ready={}",
                session_key,
                work_id,
                generated_incomplete,
                predecessors_ready,
            )
            return
        profile = self._load_workplace_story_profile(work_id)
        beats = profile.get("beats") if isinstance(profile.get("beats"), list) else []
        beat_count = len(beats)
        if self._workplace_shot_specs_complete_on_disk(
            work_id, beat_count=beat_count
        ) and self._workplace_references_ready(work_id, beat_count=beat_count):
            try:
                self._apply_workplace_generate_all(session_key)
            except EchoGeneratorUnavailableError as exc:
                self._report_echo_unavailable(session_key, exc, work_id=work_id)
                return
            except EchoGeneratorBusyError as exc:
                logger.error(
                    "auto_generate generate_all busy session_key={} work_id={} error={}",
                    session_key,
                    work_id,
                    exc,
                )
                return
            except ValueError as exc:
                logger.info(
                    "auto_generate generate_all skipped session_key={} work_id={} error={}",
                    session_key,
                    work_id,
                    exc,
                )
            return
        if state.get("story_confirmed") and str(state.get("stage") or "") not in {
            "shot_generating",
            "merging",
            "done",
        }:
            try:
                self._apply_workplace_start_generation(session_key)
            except ValueError as exc:
                logger.info(
                    "auto_generate start_generation skipped session_key={} error={}",
                    session_key,
                    exc,
                )

    def _auto_approve_memory_review_if_needed(self, session_key: str, shot_id: int) -> None:
        if not self._memory_review_workflow_enabled():
            return
        loaded = self._load_workplace_shot(session_key, shot_id)
        if loaded is None:
            return
        _work_id, _path, _state, shot = loaded
        review = shot.get("memory_review")
        if not isinstance(review, dict) or review.get("status") != "awaiting_review":
            return
        try:
            self._apply_memory_review_action(
                session_key,
                shot_id,
                action="approve",
                review_id=str(review.get("review_id") or ""),
                attempt=int(review.get("attempt") or 1),
            )
        except Exception:
            logger.opt(exception=True).error(
                "auto_generate memory approve failed session_key={} shot_id={}",
                session_key,
                shot_id,
            )

    def _schedule_auto_generate_merge(
        self,
        session_key: str,
        work_id: str,
        state: dict[str, Any],
    ) -> None:
        if state.get("final_output_url") or state.get("final_output_path"):
            return
        pending_kinds = {
            str(item.get("kind"))
            for item in self._workplace_pending_remote_jobs(state).values()
            if isinstance(item, dict)
        }
        if pending_kinds.intersection({"merge_shot"}):
            return
        shot_entries = self._workplace_shot_entries(state)
        approval_error = self._auto_approve_workplace_shots(session_key, shot_entries)
        if approval_error:
            logger.error(
                "auto_generate merge blocked session_key={} work_id={} error={}",
                session_key,
                work_id,
                approval_error,
            )
            return
        state = self._load_workplace_state(work_id)
        if not state.get("review_completed_at"):
            state["review_completed_at"] = (
                datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
            )
        state["stage"] = "merging"
        self._save_workplace_state(work_id, state)
        logger.info(
            "auto_generate merge starting session_key={} work_id={}",
            session_key,
            work_id,
        )
        if not self._schedule_coro(
            lambda: self._complete_auto_generate_merge(session_key, work_id),
        ):
            self._schedule_workplace_start_merge_instruction(session_key, work_id=work_id)

    async def _complete_auto_generate_merge(self, session_key: str, work_id: str) -> None:
        try:
            await asyncio.to_thread(self._submit_workplace_merge, session_key, work_id)
        except Exception:
            logger.opt(exception=True).error(
                "auto_generate merge submit failed session_key={} work_id={}",
                session_key,
                work_id,
            )
            self._schedule_workplace_start_merge_instruction(session_key, work_id=work_id)
        try:
            await self._publish_workplace_update(session_key)
        except Exception:
            logger.opt(exception=True).error(
                "auto_generate merge update failed session_key={} work_id={}",
                session_key,
                work_id,
            )

    def _submit_workplace_merge(self, session_key: str, work_id: str) -> dict[str, Any]:
        merge_tool = self._director_merge_tool()
        merge_tool.set_context(
            "websocket",
            webui_wire_chat_id(session_key) or "direct",
            effective_key=session_key,
        )
        token = _WORKFLOW_INJECTED_EVENT.set(WORKFLOW_GATE_BYPASS)
        try:
            job = merge_tool.apply_merge(work_id=work_id)
        finally:
            _WORKFLOW_INJECTED_EVENT.reset(token)
        logger.info(
            "auto_generate merge submitted session_key={} work_id={} job_id={}",
            session_key,
            work_id,
            job.get("job_id"),
        )
        return job

    def _auto_approve_workplace_shots(self, session_key: str, shot_entries: list[dict[str, Any]]) -> str | None:
        """Accept every generated shot. Returns an error message or None on success."""
        for item in shot_entries:
            status = str(item.get("status") or "")
            if status in {"review_fail", "error"}:
                shot_id = item.get("shot_id")
                return f"resolve failed shot {shot_id} before merging"
            if status in {"queued", "prompt_ready", "revised_prompt_ready", "planned"}:
                shot_id = item.get("shot_id")
                return f"finish generation for shot {shot_id} before merging"
        for item in shot_entries:
            status = str(item.get("status") or "")
            shot_id = int(item.get("shot_id") or 0)
            if shot_id <= 0:
                continue
            if status in {"generated", "review_pass"}:
                try:
                    self._apply_workplace_review(
                        session_key, shot_id, verdict="accept", review_source="auto"
                    )
                except ValueError as exc:
                    return str(exc)
            self._auto_approve_memory_review_if_needed(session_key, shot_id)
            loaded = self._load_workplace_shot(session_key, shot_id)
            if loaded is not None:
                _work, _path, _state, latest = loaded
                if str(latest.get("status") or "") != "approved":
                    return f"shot {shot_id} is not ready to merge"
            elif status != "approved":
                return f"shot {shot_id} is not ready to merge"
        return None

    def _handle_workplace_workflow_regenerate(self, request: WsRequest, key: str) -> Response:
        logger.info("workplace HTTP workflow/regenerate session_key={}", key)
        if not self._check_api_token(request):
            return _http_error(401, "Unauthorized")
        legacy_err = self._legacy_webui_session_key_error(key)
        if legacy_err is not None:
            return legacy_err
        decoded_key = self._resolve_webui_api_session_key(key, request)
        if decoded_key is None:
            return _http_error(404, "session not found")
        try:
            work_id, workplace = self._apply_workplace_regenerate(decoded_key)
        except ValueError as exc:
            return _http_error(400, str(exc))
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self._publish_workplace_update(decoded_key))
        except RuntimeError:
            pass
        return _http_json_response(
            {
                "ok": True,
                "action": "regenerate",
                "work_id": work_id,
                "workplace": workplace,
            }
        )

    def _download_url_policy(self) -> DownloadUrlPolicy:
        suffixes = tuple(
            suffix.strip().lower()
            for suffix in self.config.download_allowed_domain_suffixes
            if isinstance(suffix, str) and suffix.strip()
        )
        trusted = frozenset(
            domain.strip().lower()
            for domain in self.config.download_trusted_internal_domains
            if isinstance(domain, str) and domain.strip()
        )
        return DownloadUrlPolicy(
            allowed_domain_suffixes=suffixes,
            trusted_internal_domains=trusted,
        )

    def _single_shot_video_locator(self, work_id: str) -> str | None:
        """When there is exactly one generated shot, use it as the downloadable final.

        Matches ``_build_workplace_payload`` which exposes that shot as ``final_video``
        for short-video / one-shot flows before an explicit merge writes final_output_*.
        """
        paths = self._workplace_paths(work_id)
        if paths is None:
            return None
        shot_paths = sorted(paths["shots"].glob("shot_*.json"))
        if len(shot_paths) != 1:
            return None
        shot = self._read_json_file(shot_paths[0], {})
        if not isinstance(shot, dict):
            return None
        remote = shot.get("remote_result") if isinstance(shot.get("remote_result"), dict) else {}
        for candidate in (
            shot.get("artifact_url"),
            shot.get("artifact_path"),
            remote.get("video_path"),
        ):
            if isinstance(candidate, str) and candidate.strip():
                return candidate.strip()
        return None

    def _resolve_workplace_final_video_locator(self, session_key: str) -> tuple[str, str, str]:
        """Return ``(work_id, locator, filename)`` for the session's final video."""
        work_id = self._resolve_work_id_for_session(session_key)
        if work_id:
            paths = self._workplace_paths(work_id)
            if paths is None:
                raise ValueError("director work not found")
            state = self._read_json_file(paths["state"], {})
            if not isinstance(state, dict):
                state = {}
            locator = None
            for candidate in (state.get("final_output_url"), state.get("final_output_path")):
                if isinstance(candidate, str) and candidate.strip():
                    locator = candidate.strip()
                    break
            if not locator:
                # Short video: single generated shot is treated as the final for download.
                locator = self._single_shot_video_locator(work_id)
            if not locator:
                raise ValueError("final video is not ready")
            if not urlparse(locator).scheme and not Path(locator).is_absolute():
                for base_dir in (paths["work_dir"], paths["work_dir"] / "outputs", paths["shots"]):
                    candidate = base_dir / locator
                    if candidate.is_file():
                        locator = str(candidate)
                        break
            filename = _media_display_name(locator) or f"{work_id}-final.mp4"
            return work_id, locator, filename

        raise ValueError("director work not found")

    def _fetch_workplace_final_video(
        self,
        locator: str,
        filename: str,
    ) -> tuple[bytes, str, str]:
        """Download final video bytes and return ``(data, content_type, filename)``."""
        parsed = urlparse(locator)
        if parsed.scheme in _REMOTE_MEDIA_SCHEMES:
            configure_download_policy(self._download_url_policy())
            try:
                validate_external_url(locator)
            except UrlValidationError as exc:
                logger.warning("final video URL blocked by SSRF policy: {}", exc)
                raise ValueError("URL安全校验失败") from exc
            try:
                result = download_http_bytes(
                    locator,
                    max_bytes=int(self.config.download_max_bytes),
                    timeout_s=float(self.config.download_timeout_s),
                )
            except HttpDownloadError as exc:
                logger.warning("final video HTTP download failed: {}", exc)
                raise ValueError("下载失败") from exc
            return result.data, result.content_type, filename

        local = Path(unquote(parsed.path)) if parsed.scheme == "file" else Path(locator).expanduser()
        try:
            candidate = local.resolve()
        except OSError as exc:
            raise ValueError("final video is not ready") from exc
        if not candidate.is_file():
            raise ValueError("final video is not ready")
        max_bytes = int(self.config.download_max_bytes)
        try:
            size = candidate.stat().st_size
        except OSError as exc:
            raise ValueError("final video is not ready") from exc
        if size > max_bytes:
            raise ValueError("文件大小超过限制")
        try:
            data = candidate.read_bytes()
        except OSError as exc:
            raise ValueError("下载失败") from exc
        mime, _ = mimetypes.guess_type(candidate.name)
        content_type = mime if mime in _MEDIA_ALLOWED_MIMES else "application/octet-stream"
        display_name = _media_display_name(str(candidate)) or filename
        return data, content_type, display_name

    def _record_echo_video_download(self, session_key: str) -> None:
        tracking_ctx = self._resolve_echo_tracking_context(session_key)
        if tracking_ctx is None:
            return
        state = tracking_ctx["state"]
        state["video_downloaded"] = True
        state["updated_at"] = datetime.now(timezone.utc).isoformat()
        self._save_echo_tracking_context(tracking_ctx)

    def _handle_workplace_download_final(self, request: WsRequest, key: str) -> Response:
        """Authenticated proxy download for the workplace final merged video."""
        logger.info("workplace HTTP download/final session_key={}", key)
        if not self._check_api_token(request):
            return _http_error(401, "Unauthorized")
        legacy_err = self._legacy_webui_session_key_error(key)
        if legacy_err is not None:
            return legacy_err
        decoded_key = self._resolve_webui_api_session_key(key, request)
        if decoded_key is None:
            return _http_error(404, "session not found")
        try:
            work_id, locator, filename = self._resolve_workplace_final_video_locator(decoded_key)
            data, content_type, filename = self._fetch_workplace_final_video(locator, filename)
        except ValueError as exc:
            message = str(exc)
            if message == "URL安全校验失败":
                return _http_json_response(
                    {"code": 3, "message": message},
                    status=400,
                )
            status = 404 if "not ready" in message or "not found" in message else 400
            if message == "下载失败":
                return _http_json_response({"code": 4, "message": message}, status=400)
            return _http_error(status, message)
        self._record_echo_video_download(decoded_key)
        safe_name = filename.replace('"', "")
        return _http_response(
            data,
            content_type=content_type,
            extra_headers=[
                ("Content-Disposition", f'attachment; filename="{safe_name}"'),
                ("Cache-Control", "private, no-store"),
                ("X-Content-Type-Options", "nosniff"),
            ],
        )

    def _handle_workplace_workflow_start_merge(self, request: WsRequest, key: str) -> Response:
        logger.info("workplace HTTP workflow/start-merge session_key={}", key)
        resolved = self._resolve_workplace_request(request, key)
        if not isinstance(resolved, tuple):
            return resolved
        decoded_key, work_id = resolved
        paths = self._workplace_paths(work_id)
        if paths is None:
            return _http_error(404, "director work not found")
        state = self._read_json_file(paths["state"], {})
        if not isinstance(state, dict):
            state = {}
        self._reconcile_workplace_state_shots(work_id, state)
        if state.get("final_output_url") or state.get("final_output_path"):
            return _http_error(409, "work already completed")
        if not state.get("story_confirmed"):
            return _http_error(400, "story is not confirmed")
        shot_entries = self._workplace_shot_entries(state)
        if not shot_entries:
            return _http_error(400, "no shots found")
        pending_kinds = {
            str(item.get("kind"))
            for item in self._workplace_pending_remote_jobs(state).values()
            if isinstance(item, dict)
        }
        if "generate_echo_shot" in pending_kinds:
            return _http_error(409, "shot generation or review still in progress")
        if pending_kinds.intersection({"merge_shot"}):
            return _http_error(409, "merge already in progress")
        approval_error = self._auto_approve_workplace_shots(decoded_key, shot_entries)
        if approval_error:
            return _http_error(400, approval_error)
        state = self._read_json_file(paths["state"], {})
        if not isinstance(state, dict):
            state = {}
        shot_entries = self._workplace_shot_entries(state)
        if not all(str(item.get("status") or "") == "approved" for item in shot_entries):
            return _http_error(400, "accept every shot before merging")
        if not state.get("review_completed_at"):
            state["review_completed_at"] = (
                datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
            )
        state["stage"] = "merging"
        self._save_workplace_state(work_id, state)
        self._schedule_workplace_start_merge_instruction(decoded_key, work_id=work_id)
        workplace = self._build_workplace_payload(decoded_key)
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self._publish_workplace_update(decoded_key))
        except RuntimeError:
            pass
        return _http_json_response(
            {
                "ok": True,
                "action": "start_merge",
                "work_id": work_id,
                "scheduled": True,
                "workplace": workplace,
            }
        )

    def _handle_promptstack_sessions(self, request: WsRequest) -> Response:
        if not self._check_api_token(request):
            return _http_error(401, "Unauthorized")
        from nanobot.agent.prompt_stacker import PromptStacker
        sessions = PromptStacker.get_sessions()
        return _http_json_response(sessions)

    def _handle_promptstack_trace(self, request: WsRequest, session_id: str) -> Response:
        if not self._check_api_token(request):
            return _http_error(401, "Unauthorized")
        from nanobot.agent.prompt_stacker import PromptStacker
        trace = PromptStacker.get_trace(session_id)
        if not trace:
            return _http_json_response({"error": "not found"}, status=404)
        return _http_json_response(trace)

    def _handle_eventstack_sessions(self, request: WsRequest) -> Response:
        if not self._check_api_token(request):
            return _http_error(401, "Unauthorized")
        from nanobot.agent.event_stacker import EventStacker
        sessions = EventStacker.get_sessions()
        return _http_json_response(sessions)

    def _handle_eventstack_trace(self, request: WsRequest, session_id: str) -> Response:
        if not self._check_api_token(request):
            return _http_error(401, "Unauthorized")
        from nanobot.agent.event_stacker import EventStacker
        trace = EventStacker.get_trace(session_id)
        if not trace:
            return _http_json_response({"error": "not found"}, status=404)
        return _http_json_response(trace)

    def _serve_static(self, request_path: str) -> Response | None:
        """Resolve *request_path* against the built SPA directory; SPA fallback to index.html."""
        assert self._static_dist_path is not None
        rel = request_path.lstrip("/")
        if not rel:
            rel = "index.html"
        # Reject path-traversal attempts and absolute targets.
        if ".." in rel.split("/") or rel.startswith("/"):
            return _http_error(403, "Forbidden")
        candidate = (self._static_dist_path / rel).resolve()
        try:
            candidate.relative_to(self._static_dist_path)
        except ValueError:
            return _http_error(403, "Forbidden")
        if not candidate.is_file():
            # SPA history-mode fallback: unknown routes serve index.html so the
            # client-side router can render them.
            index = self._static_dist_path / "index.html"
            if index.is_file():
                candidate = index
            else:
                return None
        try:
            body = candidate.read_bytes()
        except OSError as e:
            logger.warning("websocket static: failed to read {}: {}", candidate, e)
            return _http_error(500, "Internal Server Error")
        ctype, _ = mimetypes.guess_type(candidate.name)
        if ctype is None:
            ctype = "application/octet-stream"
        if ctype.startswith("text/") or ctype in {"application/javascript", "application/json"}:
            ctype = f"{ctype}; charset=utf-8"
        # Hash-named build assets are cache-friendly; index.html must stay fresh.
        if candidate.name == "index.html":
            cache = "no-cache"
        else:
            cache = "public, max-age=31536000, immutable"
        return _http_response(
            body,
            status=200,
            content_type=ctype,
            extra_headers=[("Cache-Control", cache)],
        )

    def _authorize_websocket_handshake(self, connection: Any, query: dict[str, list[str]]) -> Any:
        token = _query_first(query, "token")
        static_token = self.config.token.strip()
        if not self.config.websocket_requires_token and not static_token:
            return None
        if static_token and token and hmac.compare_digest(token, static_token):
            return None
        if self._take_issued_token_if_valid(token):
            return None
        return connection.respond(401, "Unauthorized")

    async def start(self) -> None:
        self._running = True
        self._loop = asyncio.get_running_loop()
        self._stop_event = asyncio.Event()

        ssl_context = self._build_ssl_context()
        scheme = "wss" if ssl_context else "ws"

        async def process_request(
            connection: ServerConnection,
            request: WsRequest,
        ) -> Any:
            return await self._dispatch_http(connection, request)

        async def handler(connection: ServerConnection) -> None:
            await self._connection_loop(connection)

        logger.info(
            "WebSocket server listening on {}://{}:{}{}",
            scheme,
            self.config.host,
            self.config.port,
            self.config.path,
        )
        if self.config.token_issue_path:
            logger.info(
                "WebSocket token issue route: {}://{}:{}{}",
                scheme,
                self.config.host,
                self.config.port,
                _normalize_config_path(self.config.token_issue_path),
            )

        async def runner() -> None:
            async with serve(
                handler,
                self.config.host,
                self.config.port,
                process_request=process_request,
                max_size=self.config.max_message_bytes,
                ping_interval=self.config.ping_interval_s,
                ping_timeout=self.config.ping_timeout_s,
                ssl=ssl_context,
            ):
                assert self._stop_event is not None
                await self._stop_event.wait()

        self._server_task = asyncio.create_task(runner())
        await self._server_task

    async def _connection_loop(self, connection: Any) -> None:
        request = connection.request
        path_part = request.path if request else "/"
        _, query = _parse_request_path(path_part)
        client_id_raw = _query_first(query, "client_id")
        client_id = client_id_raw.strip() if client_id_raw else ""
        if not client_id:
            client_id = f"anon-{uuid.uuid4().hex[:12]}"
        elif len(client_id) > 128:
            logger.warning("websocket: client_id too long ({} chars), truncating", len(client_id))
            client_id = client_id[:128]

        default_chat_id = str(uuid.uuid4())
        try:
            await connection.send(
                json.dumps(
                    {
                        "event": "ready",
                        "chat_id": default_chat_id,
                        "client_id": client_id,
                    },
                    ensure_ascii=False,
                )
            )
            # Register only after ready is successfully sent to avoid out-of-order sends
            self._conn_default[connection] = default_chat_id
            self._attach(connection, default_chat_id)

            async for raw in connection:
                if isinstance(raw, bytes):
                    try:
                        raw = raw.decode("utf-8")
                    except UnicodeDecodeError:
                        logger.warning("websocket: ignoring non-utf8 binary frame")
                        continue

                envelope = _parse_envelope(raw)
                if envelope is not None:
                    await self._dispatch_envelope(connection, client_id, envelope)
                    continue

                content = _parse_inbound_payload(raw)
                if content is None:
                    continue
                await self._handle_message(
                    sender_id=client_id,
                    chat_id=default_chat_id,
                    content=content,
                    metadata={"remote": getattr(connection, "remote_address", None)},
                    session_key=self._webui_session_key_for_connection(
                        connection, default_chat_id
                    ),
                )
        except Exception as e:
            logger.debug("websocket connection ended: {}", e)
        finally:
            self._cleanup_connection(connection)

    @staticmethod
    def _save_envelope_media(
        media: list[Any],
    ) -> tuple[list[str], str | None]:
        """Decode and persist ``media`` items from a ``message`` envelope.

        Returns ``(paths, None)`` on success or ``([], reason)`` on the first
        failure — the caller is expected to surface ``reason`` to the client
        and skip publishing so no half-formed message ever reaches the agent.
        On failure, any images already written to disk earlier in the same
        call are unlinked so partial ingress doesn't leak orphan files.
        ``reason`` is a short, stable token suitable for UI localization.

        Shape: ``list[{"data_url": str, "name"?: str | None}]``.
        """
        if len(media) > _MAX_IMAGES_PER_MESSAGE:
            return [], "too_many_images"
        media_dir = get_media_dir("websocket")
        paths: list[str] = []

        def _abort(reason: str) -> tuple[list[str], str]:
            for p in paths:
                try:
                    Path(p).unlink(missing_ok=True)
                except OSError as exc:
                    logger.warning(
                        "websocket: failed to unlink partial media {}: {}", p, exc
                    )
            return [], reason

        for item in media:
            if not isinstance(item, dict):
                return _abort("malformed")
            data_url = item.get("data_url")
            if not isinstance(data_url, str) or not data_url:
                return _abort("malformed")
            mime = _extract_data_url_mime(data_url)
            if mime is None:
                return _abort("decode")
            if mime not in _IMAGE_MIME_ALLOWED:
                return _abort("mime")
            try:
                saved = save_base64_data_url(
                    data_url, media_dir, max_bytes=_MAX_IMAGE_BYTES,
                )
            except FileSizeExceeded:
                return _abort("size")
            except Exception as exc:
                logger.warning("websocket: media decode failed: {}", exc)
                return _abort("decode")
            if saved is None:
                return _abort("decode")
            paths.append(saved)
        return paths, None

    async def _dispatch_envelope(
        self,
        connection: Any,
        client_id: str,
        envelope: dict[str, Any],
    ) -> None:
        """Route one typed inbound envelope (``new_chat`` / ``attach`` / ``message``)."""
        t = envelope.get("type")
        if t == "new_chat":
            new_id = str(uuid.uuid4())
            self._attach(connection, new_id)
            session_key = self._webui_session_key_for_connection(connection, new_id)
            source = self._envelope_source(envelope)
            if session_key and self._session_manager is not None:
                if source:
                    self._persist_session_source(session_key, source)
                auto_generate = resolve_auto_generate_from_wire(envelope)
                if auto_generate is not None:
                    session = self._session_manager.get_or_create(session_key)
                    if apply_auto_generate(session.metadata, auto_generate):
                        self._session_manager.save(session)
            await self._send_event(
                connection,
                "attached",
                chat_id=new_id,
                active_pe=self._active_pe_for_connection(connection, new_id),
            )
            return
        if t == "attach":
            cid = envelope.get("chat_id")
            if not _is_valid_chat_id(cid):
                await self._send_event(connection, "error", detail="invalid chat_id")
                return
            self._attach(connection, cid)
            session_key = self._webui_session_key_for_connection(connection, cid)
            source = self._envelope_source(envelope)
            if session_key and source:
                self._persist_session_source(session_key, source)
            await self._send_event(
                connection,
                "attached",
                chat_id=cid,
                active_pe=self._active_pe_for_connection(connection, cid),
            )
            return
        if t == "message":
            cid = envelope.get("chat_id")
            content = envelope.get("content")
            if not _is_valid_chat_id(cid):
                await self._send_event(connection, "error", detail="invalid chat_id")
                return
            if not isinstance(content, str):
                await self._send_event(connection, "error", detail="missing content")
                return

            raw_media = envelope.get("media")
            media_paths: list[str] = []
            if raw_media is not None:
                if not isinstance(raw_media, list):
                    await self._send_event(
                        connection, "error",
                        detail="image_rejected", reason="malformed",
                    )
                    return
                media_paths, reason = self._save_envelope_media(raw_media)
                if reason is not None:
                    await self._send_event(
                        connection, "error",
                        detail="image_rejected", reason=reason,
                    )
                    return

            # Allow image-only turns (content may be empty when media is attached).
            if not content.strip() and not media_paths:
                await self._send_event(connection, "error", detail="missing content")
                return

            # Auto-attach on first use so clients can one-shot without a separate attach.
            self._attach(connection, cid)
            session_key = self._webui_session_key_for_connection(connection, cid)
            if session_key:
                self._hydrate_session_pe(session_key)
            source = self._envelope_source(envelope)
            if session_key and not source and self._session_manager is not None:
                from nanobot.session.source import get_source

                existing = get_source(
                    self._session_manager.get_or_create(session_key).metadata
                )
                if existing:
                    source = existing
            if session_key and source:
                self._persist_session_source(session_key, source)
            msg_metadata: dict[str, Any] = {
                "remote": getattr(connection, "remote_address", None),
            }
            if source:
                msg_metadata["source"] = source
            n_shots = envelope.get("nShot")
            if n_shots is None:
                n_shots = envelope.get("nshot")
            if n_shots is not None and n_shots != "":
                msg_metadata["nShot"] = n_shots
            duration_sec = envelope.get("durationSec")
            if duration_sec is None:
                duration_sec = envelope.get("duration_sec")
            if duration_sec is not None and duration_sec != "":
                msg_metadata["duration_sec"] = duration_sec
            auto_generate = resolve_auto_generate_from_wire(envelope)
            if auto_generate is not None:
                msg_metadata["auto_generate"] = auto_generate
            for src, dst in (
                ("temperature", "temperature"),
                ("topP", "top_p"),
                ("top_p", "top_p"),
                ("topK", "top_k"),
                ("top_k", "top_k"),
            ):
                if src in envelope and envelope.get(src) not in (None, ""):
                    msg_metadata[dst] = envelope.get(src)
            # 首帧参考图 — 从 WS 消息中提取并写入 session metadata
            ref = normalize_reference_image(envelope)
            if ref is None:
                ref_url = envelope.get("reference_image_url")
                ref_name = envelope.get("reference_image_name")
                ref_w = envelope.get("reference_image_width")
                ref_h = envelope.get("reference_image_height")
                if isinstance(ref_url, str) and ref_url.strip():
                    ref = {
                        "url": ref_url.strip(),
                        "name": ref_name if isinstance(ref_name, str) else "",
                        "width": int(ref_w) if isinstance(ref_w, (int, float)) else 0,
                        "height": int(ref_h) if isinstance(ref_h, (int, float)) else 0,
                    }
            if ref:
                msg_metadata["reference_image_url"] = ref["url"]
                msg_metadata["reference_image_name"] = ref.get("name") or ""
                msg_metadata["reference_image_width"] = ref.get("width") or 0
                msg_metadata["reference_image_height"] = ref.get("height") or 0
            if self._session_manager is not None and session_key and (
                n_shots is not None or duration_sec is not None or auto_generate is not None
            ):
                from nanobot.session.generation_settings import (
                    apply_generation_settings,
                    normalize_duration_sec,
                    normalize_n_shots,
                )

                session = self._session_manager.get_or_create(session_key)
                changed = False
                if auto_generate is not None:
                    changed = apply_auto_generate(session.metadata, auto_generate) or changed
                if auto_generate is True:
                    try:
                        work_id = self._resolve_work_id_for_session(session_key)
                        locked_n = None
                        if work_id:
                            locked_n = locked_shot_count_from_state(
                                self._load_workplace_state(work_id)
                            )
                        self._sync_auto_generate_n_shots(
                            session.metadata,
                            duration_sec=duration_sec,
                            locked_shot_count=locked_n,
                        )
                        changed = True
                    except ValueError:
                        logger.error(
                            "websocket: invalid auto_generate duration_sec={} session_key={}",
                            duration_sec,
                            session_key,
                        )
                elif n_shots is not None or duration_sec is not None:
                    try:
                        apply_generation_settings(
                            session.metadata,
                            n_shots=normalize_n_shots(n_shots),
                            duration_sec=normalize_duration_sec(duration_sec),
                        )
                        changed = True
                    except ValueError:
                        pass
                if changed:
                    self._session_manager.save(session)
            # 首帧参考图持久化到 session metadata（PUT 为真源，WS 仅兜底）
            if self._session_manager is not None and session_key and ref:
                session = self._session_manager.get_or_create(session_key)
                metadata = session.metadata if isinstance(session.metadata, dict) else {}
                if not is_reference_image_locked(metadata):
                    work_id = self._resolve_work_id_for_session(session_key)
                    state_locked = False
                    if work_id:
                        state = self._load_workplace_state(work_id)
                        state_locked = is_reference_image_locked(state)
                    if not state_locked:
                        if is_blocked_local_url(str(ref.get("url") or "")):
                            logger.error(
                                "websocket: rejected local reference_image url session_key={}",
                                session_key,
                            )
                        else:
                            session.metadata["reference_image"] = ref
                            self._session_manager.save(session)
            if self._session_manager is not None and session_key and any(
                key in msg_metadata for key in ("temperature", "top_p", "top_k")
            ):
                from nanobot.session.generation_settings import apply_llm_sampling_from_wire

                session = self._session_manager.get_or_create(session_key)
                if apply_llm_sampling_from_wire(session.metadata, msg_metadata):
                    self._session_manager.save(session)
            if self._session_manager is not None and session_key:
                from nanobot.session.reference_image_gate import (
                    consume_skip_next_message,
                    is_upload_gate_answer,
                )

                session = self._session_manager.get_or_create(session_key)
                metadata = session.metadata if isinstance(session.metadata, dict) else {}
                user_answer = content
                skip_question = consume_skip_next_message(metadata)
                if skip_question:
                    self._session_manager.save(session)
                    self._persist_gate_user_reply(session_key, content)
                    await self._emit_upload_gate_mismatch_card(
                        session_key, cid, skip_question
                    )
                    return
                if is_upload_gate_answer(content):
                    gate = self._commit_reference_image_gate(session_key, content)
                    if gate is not None and gate.skip_agent and gate.mismatch_question:
                        self._persist_gate_user_reply(session_key, content)
                        await self._emit_upload_gate_mismatch_card(
                            session_key, cid, gate.mismatch_question
                        )
                        return
                    if gate is not None and gate.inject_note:
                        content = f"{gate.inject_note}\n\n{content}"
                confirm_note = self._commit_story_direction_answer(
                    session_key, user_answer
                )
                if confirm_note:
                    content = f"{confirm_note}\n\n{content}"
            await self._handle_message(
                sender_id=client_id,
                chat_id=cid,
                content=content,
                media=media_paths or None,
                metadata=msg_metadata,
                session_key=session_key,
            )
            return
        if t == "workplace_merge_up":
            await self._dispatch_workplace_beat_merge_up(connection, envelope)
            return
        if t == "workplace_remove_shot":
            await self._dispatch_workplace_beat_remove_shot(connection, envelope)
            return
        if t == "workplace_split_shot":
            await self._dispatch_workplace_beat_split_shot(connection, envelope)
            return
        if t == "workplace_save_story":
            await self._dispatch_workplace_save_story(connection, envelope)
            return
        if t == "workplace_save_story_profile":
            await self._dispatch_workplace_save_story_profile(connection, envelope)
            return
        if t == "workplace_save_reference_image":
            await self._dispatch_workplace_save_reference_image(connection, envelope)
            return
        if t in {
            "workplace_save_memory_asset",
            "workplace_create_shot_memory_asset",
            "workplace_delete_memory_asset",
            "workplace_save_shot_memory_slots",
        }:
            await self._dispatch_workplace_memory_workspace(connection, envelope)
            return
        if t == "workplace_start_generation":
            await self._dispatch_workplace_start_generation(connection, envelope)
            return
        if t == "answer_question":
            await self._dispatch_answer_question(connection, envelope)
            return
        if t == "set_pe":
            await self._dispatch_set_pe(connection, envelope)
            return
        await self._send_event(connection, "error", detail=f"unknown type: {t!r}")

    def _active_pe_for_connection(self, connection: Any, chat_id: str) -> str:
        """Resolve the PE set active for a connection's session (per-session, else global)."""
        from nanobot.prompts import PEManager

        manager = PEManager.instance()
        try:
            session_key = self._webui_session_key_for_connection(connection, chat_id)
        except Exception:
            return manager.active
        self._hydrate_session_pe(session_key)
        return manager.active_for_session(session_key)

    async def _dispatch_set_pe(self, connection: Any, envelope: dict[str, Any]) -> None:
        """Bind the active PE set for the caller's session; notify only that chat's connections."""
        from nanobot.prompts import PEManager

        name = envelope.get("name")
        if not isinstance(name, str) or not name:
            await self._send_event(connection, "error", detail="missing pe name")
            return
        chat_id = envelope.get("chat_id")
        if not _is_valid_chat_id(chat_id):
            await self._send_event(connection, "error", detail="invalid chat_id")
            return
        try:
            session_key = self._webui_session_key_for_connection(connection, chat_id)
        except Exception:
            await self._send_event(connection, "error", detail="session not found")
            return
        manager = PEManager.instance()
        if not manager.set_active_for_session(session_key, name):
            await self._send_event(connection, "error", detail=f"unknown pe set: {name!r}")
            return
        self._persist_session_pe(session_key, name)
        for conn, chats in list(self._conn_chats.items()):
            if chat_id in chats:
                await self._send_event(conn, "pe_updated", chat_id=chat_id, active=name)

    async def _dispatch_answer_question(
        self,
        connection: Any,
        envelope: dict[str, Any],
    ) -> None:
        cid = envelope.get("chat_id")
        batch_id = envelope.get("question_batch_id")
        card_id = envelope.get("card_id")
        value = envelope.get("value")
        if not _is_valid_chat_id(cid):
            await self._send_event(connection, "error", detail="invalid chat_id")
            return
        if not isinstance(batch_id, str) or not batch_id.strip():
            await self._send_event(connection, "error", detail="missing question_batch_id")
            return
        if not isinstance(card_id, str) or not card_id.strip():
            await self._send_event(connection, "error", detail="missing card_id")
            return
        if not isinstance(value, str) or not value.strip():
            await self._send_event(connection, "error", detail="missing value")
            return
        if self._session_manager is None:
            await self._send_event(connection, "error", detail="session manager unavailable")
            return

        session_key = self._webui_session_key_for_connection(connection, cid)
        if not session_key:
            await self._send_event(connection, "error", detail="session not found")
            return

        ok = self._session_manager.record_question_answer(
            session_key,
            batch_id.strip(),
            card_id.strip(),
            value.strip(),
        )
        if not ok:
            await self._send_event(
                connection,
                "error",
                chat_id=cid,
                detail="question answer not found",
            )
            return

        await self._send_event(
            connection,
            "question_answer_ok",
            chat_id=cid,
            question_batch_id=batch_id.strip(),
            card_id=card_id.strip(),
            value=value.strip(),
        )
        self._maybe_start_workflow_from_question_answer(session_key, value.strip())
        self._commit_reference_image_gate(session_key, value.strip())
        self._commit_story_direction_answer(session_key, value.strip())

    def _commit_story_direction_answer(self, session_key: str, answer: str) -> str | None:
        from nanobot.agent.tools.ask_user import (
            is_reference_image_edit_option,
            is_story_confirm_option,
        )

        if self._session_manager is None:
            return None
        if not is_story_confirm_option(answer) and not is_reference_image_edit_option(
            answer
        ):
            return None
        session = self._session_manager.get_or_create(session_key)
        if not isinstance(session.metadata, dict):
            session.metadata = {}
        note = apply_story_direction_answer(session.metadata, answer)
        self._session_manager.save(session)
        if note:
            logger.info(
                "story direction confirmed, rewrite suppressed session_key={}",
                session_key,
            )
        return note

    def _maybe_start_workflow_from_question_answer(
        self, session_key: str, value: str
    ) -> None:
        answer = (value or "").strip()
        if answer not in {_CARD_ENTER_SHOT_POLISH, _CARD_CONFIRM_AUTO_GENERATE}:
            return
        try:
            if answer == _CARD_CONFIRM_AUTO_GENERATE:
                self._apply_workplace_auto_generate(session_key)
            else:
                self._apply_workplace_start_generation(session_key)
        except EchoGeneratorUnavailableError as exc:
            self._report_echo_unavailable(session_key, exc)
            return
        except EchoGeneratorBusyError as exc:
            logger.error(
                "workplace card workflow busy session_key={} answer={} error={}",
                session_key,
                answer,
                exc,
            )
            return
        except ValueError as exc:
            logger.error(
                "workplace card workflow rejected session_key={} answer={} error={}",
                session_key,
                answer,
                exc,
            )
            return
        except Exception:
            logger.opt(exception=True).error(
                "workplace card workflow failed session_key={} answer={}",
                session_key,
                answer,
            )
            return
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(self._publish_workplace_update(session_key))
        except RuntimeError:
            pass

    async def _send_workplace_action_error(
        self,
        connection: Any,
        *,
        chat_id: str,
        request_id: str,
        detail: str,
    ) -> None:
        await self._send_event(
            connection,
            "workplace_action_error",
            chat_id=chat_id,
            request_id=request_id,
            detail=detail,
        )

    async def _send_workplace_action_ok(
        self,
        connection: Any,
        *,
        chat_id: str,
        request_id: str,
        work_id: str,
        workplace: dict[str, Any],
    ) -> None:
        await self._send_event(
            connection,
            "workplace_action_ok",
            chat_id=chat_id,
            request_id=request_id,
            work_id=work_id,
            workplace=workplace,
        )

    async def _dispatch_workplace_beat_merge_up(
        self,
        connection: Any,
        envelope: dict[str, Any],
    ) -> None:
        cid = envelope.get("chat_id")
        request_id = envelope.get("request_id")
        shot_raw = envelope.get("shot_id")
        if not _is_valid_chat_id(cid):
            await self._send_workplace_action_error(
                connection,
                chat_id="",
                request_id=str(request_id or ""),
                detail="invalid chat_id",
            )
            return
        if not isinstance(request_id, str) or not request_id.strip():
            await self._send_event(connection, "error", detail="missing request_id")
            return
        try:
            shot_id = int(shot_raw)
        except (TypeError, ValueError):
            await self._send_workplace_action_error(
                connection,
                chat_id=cid,
                request_id=request_id,
                detail="invalid shot_id",
            )
            return
        try:
            session_key = self._webui_session_key_for_connection(connection, cid)
        except ValueError as exc:
            await self._send_workplace_action_error(
                connection,
                chat_id=cid,
                request_id=request_id,
                detail=str(exc),
            )
            return
        merged_text = envelope.get("merged_text")
        if merged_text is not None and not isinstance(merged_text, str):
            merged_text = None
        try:
            merged = self._apply_workplace_shot_merge_up(
                session_key,
                shot_id,
                merged_text=merged_text,
            )
        except ValueError as exc:
            await self._send_workplace_action_error(
                connection,
                chat_id=cid,
                request_id=request_id,
                detail=str(exc),
            )
            return
        if merged is None:
            await self._send_workplace_action_error(
                connection,
                chat_id=cid,
                request_id=request_id,
                detail="beat not found",
            )
            return
        work_id, workplace = merged
        self._attach(connection, cid)
        await self._send_workplace_action_ok(
            connection,
            chat_id=cid,
            request_id=request_id,
            work_id=work_id,
            workplace=workplace,
        )

    async def _dispatch_workplace_beat_remove_shot(
        self,
        connection: Any,
        envelope: dict[str, Any],
    ) -> None:
        cid = envelope.get("chat_id")
        request_id = envelope.get("request_id")
        shot_raw = envelope.get("shot_id")
        if not _is_valid_chat_id(cid):
            await self._send_workplace_action_error(
                connection,
                chat_id="",
                request_id=str(request_id or ""),
                detail="invalid chat_id",
            )
            return
        if not isinstance(request_id, str) or not request_id.strip():
            await self._send_event(connection, "error", detail="missing request_id")
            return
        try:
            shot_id = int(shot_raw)
        except (TypeError, ValueError):
            await self._send_workplace_action_error(
                connection,
                chat_id=cid,
                request_id=request_id,
                detail="invalid shot_id",
            )
            return
        try:
            session_key = self._webui_session_key_for_connection(connection, cid)
        except ValueError as exc:
            await self._send_workplace_action_error(
                connection,
                chat_id=cid,
                request_id=request_id,
                detail=str(exc),
            )
            return
        try:
            removed = self._apply_workplace_shot_remove(session_key, shot_id)
        except ValueError as exc:
            await self._send_workplace_action_error(
                connection,
                chat_id=cid,
                request_id=request_id,
                detail=str(exc),
            )
            return
        if removed is None:
            await self._send_workplace_action_error(
                connection,
                chat_id=cid,
                request_id=request_id,
                detail="beat not found",
            )
            return
        work_id, workplace = removed
        self._attach(connection, cid)
        await self._send_workplace_action_ok(
            connection,
            chat_id=cid,
            request_id=request_id,
            work_id=work_id,
            workplace=workplace,
        )

    async def _dispatch_workplace_beat_split_shot(
        self,
        connection: Any,
        envelope: dict[str, Any],
    ) -> None:
        cid = envelope.get("chat_id")
        request_id = envelope.get("request_id")
        shot_raw = envelope.get("shot_id")
        if not _is_valid_chat_id(cid):
            await self._send_workplace_action_error(
                connection,
                chat_id="",
                request_id=str(request_id or ""),
                detail="invalid chat_id",
            )
            return
        if not isinstance(request_id, str) or not request_id.strip():
            await self._send_event(connection, "error", detail="missing request_id")
            return
        try:
            shot_id = int(shot_raw)
        except (TypeError, ValueError):
            await self._send_workplace_action_error(
                connection,
                chat_id=cid,
                request_id=request_id,
                detail="invalid shot_id",
            )
            return
        split_payload: dict[str, Any] = {}
        if "cursor_pos" in envelope:
            try:
                split_payload["cursor_pos"] = int(envelope["cursor_pos"])
            except (TypeError, ValueError):
                await self._send_workplace_action_error(
                    connection,
                    chat_id=cid,
                    request_id=request_id,
                    detail="cursor_pos must be an integer",
                )
                return
        elif "before_text" in envelope and "after_text" in envelope:
            split_payload["before_text"] = str(envelope.get("before_text") or "")
            split_payload["after_text"] = str(envelope.get("after_text") or "")
        else:
            await self._send_workplace_action_error(
                connection,
                chat_id=cid,
                request_id=request_id,
                detail="cursor_pos or before_text/after_text required",
            )
            return
        try:
            session_key = self._webui_session_key_for_connection(connection, cid)
        except ValueError as exc:
            await self._send_workplace_action_error(
                connection,
                chat_id=cid,
                request_id=request_id,
                detail=str(exc),
            )
            return
        try:
            split_result = self._apply_workplace_shot_split_shot(
                session_key,
                shot_id,
                split_payload=split_payload,
            )
        except ValueError as exc:
            await self._send_workplace_action_error(
                connection,
                chat_id=cid,
                request_id=request_id,
                detail=str(exc),
            )
            return
        if split_result is None:
            await self._send_workplace_action_error(
                connection,
                chat_id=cid,
                request_id=request_id,
                detail="beat not found",
            )
            return
        work_id, workplace = split_result
        self._attach(connection, cid)
        await self._send_workplace_action_ok(
            connection,
            chat_id=cid,
            request_id=request_id,
            work_id=work_id,
            workplace=workplace,
        )

    async def _dispatch_workplace_save_story(
        self,
        connection: Any,
        envelope: dict[str, Any],
    ) -> None:
        cid = envelope.get("chat_id")
        request_id = envelope.get("request_id")
        story_md = envelope.get("story_md")
        if not _is_valid_chat_id(cid):
            await self._send_workplace_action_error(
                connection,
                chat_id="",
                request_id=str(request_id or ""),
                detail="invalid chat_id",
            )
            return
        if not isinstance(request_id, str) or not request_id.strip():
            await self._send_event(connection, "error", detail="missing request_id")
            return
        if not isinstance(story_md, str):
            await self._send_workplace_action_error(
                connection,
                chat_id=cid,
                request_id=request_id,
                detail="story_md is required",
            )
            return
        try:
            session_key = self._webui_session_key_for_connection(connection, cid)
        except ValueError as exc:
            await self._send_workplace_action_error(
                connection,
                chat_id=cid,
                request_id=request_id,
                detail=str(exc),
            )
            return
        try:
            work_id, workplace = self._apply_workplace_story_save(session_key, story_md)
        except ValueError as exc:
            await self._send_workplace_action_error(
                connection,
                chat_id=cid,
                request_id=request_id,
                detail=str(exc),
            )
            return
        self._attach(connection, cid)
        await self._send_workplace_action_ok(
            connection,
            chat_id=cid,
            request_id=request_id,
            work_id=work_id,
            workplace=workplace,
        )

    async def _dispatch_workplace_save_story_profile(
        self,
        connection: Any,
        envelope: dict[str, Any],
    ) -> None:
        cid = envelope.get("chat_id")
        request_id = envelope.get("request_id")
        story_profile = envelope.get("story_profile")
        if not _is_valid_chat_id(cid):
            await self._send_workplace_action_error(
                connection,
                chat_id="",
                request_id=str(request_id or ""),
                detail="invalid chat_id",
            )
            return
        if not isinstance(request_id, str) or not request_id.strip():
            await self._send_event(connection, "error", detail="missing request_id")
            return
        if not isinstance(story_profile, dict):
            await self._send_workplace_action_error(
                connection,
                chat_id=cid,
                request_id=request_id,
                detail="story_profile is required",
            )
            return
        logger.info(
            "workplace WS story-profile/save chat_id={} request_id={}",
            cid,
            request_id,
        )
        try:
            session_key = self._webui_session_key_for_connection(connection, cid)
        except ValueError as exc:
            await self._send_workplace_action_error(
                connection,
                chat_id=cid,
                request_id=request_id,
                detail=str(exc),
            )
            return
        try:
            work_id, workplace = self._apply_workplace_story_profile_save(
                session_key,
                story_profile,
            )
        except ValueError as exc:
            await self._send_workplace_action_error(
                connection,
                chat_id=cid,
                request_id=request_id,
                detail=str(exc),
            )
            return
        self._attach(connection, cid)
        await self._send_workplace_action_ok(
            connection,
            chat_id=cid,
            request_id=request_id,
            work_id=work_id,
            workplace=workplace,
        )

    def _schedule_workplace_memory_recommendation_instruction(
        self,
        session_key: str,
        *,
        work_id: str,
        shot_id: int,
    ) -> None:
        content = (
            "Internal Memory Workspace task. Execute silently.\n\n"
            f"Director work_id: `{work_id}`\n"
            f"Target shot_id: `{shot_id}`\n\n"
            "1. Call `get_workplace_status(include_shots=true, include_jobs=false)` and "
            "`get_shot` for the target shot.\n"
            "2. Read only `memory_assets` entries supplied by the status tool; do not inspect binary files.\n"
            "3. Recommend zero to seven ordered slots for the target shot based on its caption, "
            "provenance, `reference_type`, `reference_label`, `identity_ids`, and `profile_text`. "
            "Treat character, scene, style, object, and other references according to their assigned role.\n"
            "4. Call `set_shot_memory_recommendations` exactly once. Use only listed asset IDs.\n\n"
            "Assets without a text profile are intentionally invisible and must not be inferred. "
            "This is a recommendation draft only: never generate the shot and never approve slots. "
            "Do not send a user-facing message."
        )
        self._schedule_workplace_workflow_instruction(
            session_key,
            work_id=work_id,
            injected_event="workplace_memory_recommendation",
            content=content,
        )

    async def _dispatch_workplace_save_reference_image(
        self,
        connection: Any,
        envelope: dict[str, Any],
    ) -> None:
        """Persist a first-frame image carried in a WebSocket frame.

        Image data URLs are intentionally sent over WebSocket. Encoding them in
        ``X-Nanobot-Body`` exceeds common HTTP header limits and causes 431s.
        """
        cid = envelope.get("chat_id")
        request_id = envelope.get("request_id")
        if not _is_valid_chat_id(cid):
            await self._send_workplace_action_error(
                connection,
                chat_id="",
                request_id=str(request_id or ""),
                detail="invalid chat_id",
            )
            return
        if not isinstance(request_id, str) or not request_id.strip():
            await self._send_event(connection, "error", detail="missing request_id")
            return
        ref = normalize_reference_image(envelope.get("image"))
        if ref is None:
            await self._send_workplace_action_error(
                connection,
                chat_id=cid,
                request_id=request_id,
                detail="reference image url is required",
            )
            return
        url = str(ref.get("url") or "")
        if url.startswith("data:image/"):
            paths, reason = self._save_envelope_media(
                [{"data_url": url, "name": ref.get("name") or "first-frame.jpg"}]
            )
            if reason is not None or not paths:
                await self._send_workplace_action_error(
                    connection,
                    chat_id=cid,
                    request_id=request_id,
                    detail=f"image_rejected:{reason or 'decode'}",
                )
                return
            # Decoding above validates MIME and size. Keep the data URL in
            # metadata so the VLM can consume it without a localhost fetch.
            for path in paths:
                try:
                    Path(path).unlink(missing_ok=True)
                except OSError:
                    pass
        else:
            if is_blocked_local_url(url):
                await self._send_workplace_action_error(
                    connection,
                    chat_id=cid,
                    request_id=request_id,
                    detail="url must be a public HTTP(S) address",
                )
                return
            try:
                configure_download_policy(self._download_url_policy())
                validate_external_url(url)
            except UrlValidationError as exc:
                await self._send_workplace_action_error(
                    connection,
                    chat_id=cid,
                    request_id=request_id,
                    detail=str(exc),
                )
                return
        try:
            session_key = self._webui_session_key_for_connection(connection, cid)
            workplace = self._persist_reference_image(session_key, ref)
        except (PermissionError, ValueError) as exc:
            await self._send_workplace_action_error(
                connection,
                chat_id=cid,
                request_id=request_id,
                detail=str(exc),
            )
            return
        self._attach(connection, cid)
        await self._send_workplace_action_ok(
            connection,
            chat_id=cid,
            request_id=request_id,
            work_id=self._resolve_work_id_for_session(session_key) or "",
            workplace=workplace,
        )

    async def _dispatch_workplace_memory_workspace(
        self,
        connection: Any,
        envelope: dict[str, Any],
    ) -> None:
        """Persist local assets or apply an ordered Memory slot draft."""
        cid = envelope.get("chat_id")
        request_id = envelope.get("request_id")
        if not _is_valid_chat_id(cid):
            await self._send_workplace_action_error(
                connection,
                chat_id="",
                request_id=str(request_id or ""),
                detail="invalid chat_id",
            )
            return
        if not isinstance(request_id, str) or not request_id.strip():
            await self._send_event(connection, "error", detail="missing request_id")
            return
        try:
            session_key = self._webui_session_key_for_connection(connection, cid)
            action = envelope.get("type")
            if action == "workplace_save_memory_asset":
                work_id, workplace = await asyncio.to_thread(
                    self._apply_workplace_memory_asset_save,
                    session_key,
                    envelope.get("asset"),
                )
            elif action == "workplace_create_shot_memory_asset":
                try:
                    shot_id = int(envelope.get("shot_id") or 0)
                except (TypeError, ValueError) as exc:
                    raise ValueError("shot_id is invalid") from exc
                if shot_id <= 0:
                    raise ValueError("shot_id is invalid")
                work_id, workplace = await asyncio.to_thread(
                    self._apply_workplace_shot_memory_asset_create,
                    session_key,
                    shot_id,
                    envelope.get("asset"),
                )
            elif action == "workplace_delete_memory_asset":
                asset_id = str(envelope.get("asset_id") or "").strip()
                if not asset_id:
                    raise ValueError("asset_id is required")
                work_id, workplace = self._apply_workplace_memory_asset_delete(
                    session_key,
                    asset_id,
                )
            else:
                try:
                    shot_id = int(envelope.get("shot_id") or 0)
                except (TypeError, ValueError) as exc:
                    raise ValueError("shot_id is invalid") from exc
                if shot_id <= 0:
                    raise ValueError("shot_id is invalid")
                work_id, workplace = self._apply_workplace_shot_memory_slots_save(
                    session_key,
                    shot_id,
                    envelope.get("slots"),
                )
        except (PermissionError, RuntimeError, ValueError) as exc:
            await self._send_workplace_action_error(
                connection,
                chat_id=cid,
                request_id=request_id,
                detail=str(exc),
            )
            return
        self._attach(connection, cid)
        await self._send_workplace_action_ok(
            connection,
            chat_id=cid,
            request_id=request_id,
            work_id=work_id,
            workplace=workplace,
        )

    async def _dispatch_workplace_start_generation(
        self,
        connection: Any,
        envelope: dict[str, Any],
    ) -> None:
        cid = envelope.get("chat_id")
        request_id = envelope.get("request_id")
        if not _is_valid_chat_id(cid):
            await self._send_workplace_action_error(
                connection,
                chat_id="",
                request_id=str(request_id or ""),
                detail="invalid chat_id",
            )
            return
        if not isinstance(request_id, str) or not request_id.strip():
            await self._send_event(connection, "error", detail="missing request_id")
            return
        logger.info(
            "workplace WS workflow/start-generation chat_id={} request_id={}",
            cid,
            request_id,
        )
        try:
            session_key = self._webui_session_key_for_connection(connection, cid)
        except ValueError as exc:
            await self._send_workplace_action_error(
                connection,
                chat_id=cid,
                request_id=request_id,
                detail=str(exc),
            )
            return
        try:
            work_id, workplace = self._apply_workplace_start_generation(session_key)
        except ValueError as exc:
            await self._send_workplace_action_error(
                connection,
                chat_id=cid,
                request_id=request_id,
                detail=str(exc),
            )
            return
        self._attach(connection, cid)
        await self._send_workplace_action_ok(
            connection,
            chat_id=cid,
            request_id=request_id,
            work_id=work_id,
            workplace=workplace,
        )

    async def stop(self) -> None:
        if not self._running:
            return
        self._running = False
        if self._stop_event:
            self._stop_event.set()
        if self._server_task:
            try:
                await self._server_task
            except Exception as e:
                logger.warning("websocket: server task error during shutdown: {}", e)
            self._server_task = None
        self._subs.clear()
        self._conn_chats.clear()
        self._conn_default.clear()
        self._issued_tokens.clear()
        self._api_tokens.clear()
        self._loop = None

    async def _safe_send_to(self, connection: Any, raw: str, *, label: str = "") -> None:
        """Send a raw frame to one connection, cleaning up on ConnectionClosed."""
        try:
            await connection.send(raw)
        except ConnectionClosed:
            self._cleanup_connection(connection)
            logger.warning("websocket{}connection gone", label)
        except Exception as e:
            logger.error("websocket{}send failed: {}", label, e)
            raise

    async def send(self, msg: OutboundMessage) -> None:
        workplace_event = msg.metadata.get("_workplace_event")
        session_key = (
            msg.metadata.get("session_key")
            or self._webui_session_key_for_chat(msg.chat_id)
        )
        if workplace_event == "updated" and isinstance(session_key, str) and session_key:
            self._schedule_auto_generate_continue(session_key)
        # Snapshot the subscriber set so ConnectionClosed cleanups mid-iteration are safe.
        conns = list(self._subs.get(msg.chat_id, ()))
        if not conns:
            logger.warning("websocket: no active subscribers for chat_id={}", msg.chat_id)
            return
        workplace_event = msg.metadata.get("_workplace_event")
        if workplace_event == "updated":
            payload: dict[str, Any] = {
                "event": "workplace_updated",
                "chat_id": msg.chat_id,
            }
            workplace_payload = None
            if (
                isinstance(session_key, str)
                and session_key
                and self._session_manager is not None
            ):
                workplace_payload = self._build_workplace_payload(session_key)
            elif isinstance(msg.metadata.get("workplace"), dict):
                workplace_payload = msg.metadata.get("workplace")
            if isinstance(workplace_payload, dict):
                payload["workplace"] = workplace_payload
            work_id = msg.metadata.get("work_id")
            if not isinstance(work_id, str) or not work_id.strip():
                work_id = workplace_payload.get("work_id") if isinstance(workplace_payload, dict) else None
            if isinstance(work_id, str) and work_id.strip():
                payload["work_id"] = work_id.strip()
        else:
            payload = {
                "event": "message",
                "chat_id": msg.chat_id,
                "text": msg.content,
            }
        if msg.media:
            payload["media"] = msg.media
            media_urls = [
                ref
                for ref in (self._public_media_entry(entry) for entry in msg.media)
                if ref is not None
            ]
            if media_urls:
                payload["media_urls"] = media_urls
        if msg.reply_to:
            payload["reply_to"] = msg.reply_to
        # Mark intermediate agent breadcrumbs (tool-call hints, generic
        # progress strings) so WS clients can render them as subordinate
        # trace rows rather than conversational replies.
        if msg.metadata.get("_tool_hint"):
            payload["kind"] = "tool_hint"
        elif msg.metadata.get("_progress"):
            payload["kind"] = "progress"
        questions = msg.metadata.get("questions")
        if isinstance(questions, list) and questions:
            payload["questions"] = questions
        batch_id = msg.metadata.get("question_batch_id")
        if isinstance(batch_id, str) and batch_id.strip():
            payload["question_batch_id"] = batch_id.strip()
        raw = json.dumps(payload, ensure_ascii=False)
        for connection in conns:
            await self._safe_send_to(connection, raw, label=" ")

    async def send_delta(
        self,
        chat_id: str,
        delta: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        conns = list(self._subs.get(chat_id, ()))
        if not conns:
            return
        meta = metadata or {}
        if meta.get("_stream_end"):
            body: dict[str, Any] = {
                "event": "stream_end",
                "chat_id": chat_id,
                "resuming": bool(meta.get("_resuming")),
            }
        else:
            body = {
                "event": "delta",
                "chat_id": chat_id,
                "text": delta,
            }
        if meta.get("_stream_id") is not None:
            body["stream_id"] = meta["_stream_id"]
        raw = json.dumps(body, ensure_ascii=False)
        for connection in conns:
            await self._safe_send_to(connection, raw, label=" stream ")
