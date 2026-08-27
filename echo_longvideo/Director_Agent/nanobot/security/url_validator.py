"""External URL validation for server-side HTTP downloads (SSRF protection).

Standalone module — copy into other services without nanobot context.
Configure :func:`configure_download_policy` before calling :func:`validate_external_url`.
"""

from __future__ import annotations

import ipaddress
import socket
from dataclasses import dataclass, field
from urllib.parse import urlparse

_ALLOWED_SCHEMES = frozenset({"http", "https"})
_DEFAULT_DOMAIN_SUFFIXES: tuple[str, ...] = ()
_DEFAULT_ALLOWED_PORTS = frozenset({"", "80", "443"})
_EXTRA_PRIVATE_NETWORKS = (
    ipaddress.ip_network("100.64.0.0/10"),
    ipaddress.ip_network("198.18.0.0/15"),
)
_DNS_TIMEOUT_S = 3.0


@dataclass
class DownloadUrlPolicy:
    """Configurable SSRF policy for outbound HTTP(S) downloads."""

    allowed_domain_suffixes: tuple[str, ...] = _DEFAULT_DOMAIN_SUFFIXES
    trusted_internal_domains: frozenset[str] = frozenset()
    allowed_ports: frozenset[str] = _DEFAULT_ALLOWED_PORTS
    reject_ip_literal_hosts: bool = True
    reject_userinfo: bool = True


_policy = DownloadUrlPolicy()


def configure_download_policy(policy: DownloadUrlPolicy) -> None:
    """Replace the active download URL policy (call once at process startup)."""
    global _policy
    _policy = policy


def get_download_policy() -> DownloadUrlPolicy:
    return _policy


class UrlValidationError(ValueError):
    """Raised when an external URL fails SSRF validation."""


def validate_external_url(raw_url: str) -> str:
    """Validate *raw_url* and return the normalized URL string.

  Steps (order matters):
    1. Parse URL
    2. Reject userinfo (``http://user@host`` bypass)
    3. Scheme whitelist (http/https only)
    4. Hostname required
    5. Reject IP-literal hosts (optional, default on)
    6. Domain suffix whitelist
    7. Port whitelist (80/443/default)
    8. DNS resolve + private/reserved IP block (skipped for trusted internal domains)
    """
    policy = _policy
    try:
        parsed = urlparse(raw_url.strip())
    except Exception as exc:
        raise UrlValidationError("URL格式无效") from exc

    if policy.reject_userinfo and parsed.username is not None:
        raise UrlValidationError("URL不允许包含用户名密码")

    scheme = (parsed.scheme or "").lower()
    if scheme not in _ALLOWED_SCHEMES:
        raise UrlValidationError(f"不允许的协议: {parsed.scheme or '(none)'}")

    host = (parsed.hostname or "").lower()
    if not host:
        raise UrlValidationError("URL缺少主机名")

    if policy.reject_ip_literal_hosts:
        try:
            ipaddress.ip_address(host)
        except ValueError:
            pass
        else:
            raise UrlValidationError(f"不允许使用IP地址访问: {host}")

    if not _is_domain_allowed(host, policy.allowed_domain_suffixes):
        raise UrlValidationError(f"域名不在白名单中: {host}")

    port = parsed.port
    port_text = "" if port is None else str(port)
    if port_text not in policy.allowed_ports:
        raise UrlValidationError(f"不允许的端口: {port_text or '(default)'}")

    if host not in policy.trusted_internal_domains:
        _assert_host_resolves_to_public_ip(host)

    return raw_url.strip()


def _is_domain_allowed(host: str, suffixes: tuple[str, ...]) -> bool:
    for suffix in suffixes:
        root = suffix[1:] if suffix.startswith(".") else suffix
        if host == root or host.endswith(suffix):
            return True
    return False


def _assert_host_resolves_to_public_ip(host: str) -> None:
    try:
        socket.setdefaulttimeout(_DNS_TIMEOUT_S)
        infos = socket.getaddrinfo(host, None, socket.AF_UNSPEC, socket.SOCK_STREAM)
    except socket.gaierror as exc:
        raise UrlValidationError("DNS解析失败") from exc
    finally:
        socket.setdefaulttimeout(None)

    if not infos:
        raise UrlValidationError("DNS解析失败")

    for info in infos:
        ip_text = info[4][0]
        try:
            addr = ipaddress.ip_address(ip_text)
        except ValueError:
            continue
        if _is_private_or_reserved(addr):
            raise UrlValidationError(f"禁止访问内网地址: {host} -> {addr}")


def _is_private_or_reserved(addr: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    if addr.is_private or addr.is_loopback or addr.is_link_local or addr.is_unspecified:
        return True
    if addr.is_reserved:
        return True
    for network in _EXTRA_PRIVATE_NETWORKS:
        if addr in network:
            return True
    return False
