"""Scope configuration + enforcement for authorized live-target assessment.

Nothing active happens without an explicit Scope. Every URL is checked against the scope BEFORE a
request, and the browser layer aborts out-of-scope subresources, navigations, and server-side
redirect targets before they leave the browser; blocked URLs are logged. Paths are compared after
percent-decoding and dot-segment removal, so `/a/../admin` or `/%61dmin` cannot slip past an
exclusion. Non-loopback hosts are out of scope unless the Scope is explicitly authorized for
external testing. A bounded request budget and a rate limit cap activity. The scope never
auto-expands.
"""
from __future__ import annotations

import posixpath
import time
from dataclasses import dataclass, field
from urllib.parse import unquote, urlparse

LOOPBACK_HOSTS = ("127.0.0.1", "localhost", "::1")

# Test classes: reflected (GET query), dom (URL fragment), post (POST form fields; may change server
# state), stored (submit -> view; persists data). The last two must be opted into explicitly.
TEST_CLASSES = ("reflected", "dom", "post", "stored")
DEFAULT_TEST_CLASSES = ("reflected", "dom")


def is_loopback(url_or_host: str) -> bool:
    host = urlparse(url_or_host).hostname if "://" in url_or_host else url_or_host
    return (host or "").lower() in LOOPBACK_HOSTS


def _normalize_path(path: str) -> str:
    """Resolve dot segments the way a browser would, keeping a trailing slash."""
    path = path or "/"
    norm = posixpath.normpath("/" + path.lstrip("/"))
    norm = "/" + norm.lstrip("/")          # normpath keeps a leading '//' as-is
    if path.endswith("/") and norm != "/":
        norm += "/"
    return norm


def path_variants(path: str) -> set[str]:
    """Every interpretation of `path` a browser or server might act on: as sent, and fully
    percent-decoded (with backslashes treated as slashes), each with dot segments resolved."""
    decoded = path or "/"
    for _ in range(5):                     # undo nested encodings such as %252e
        nxt = unquote(decoded)
        if nxt == decoded:
            break
        decoded = nxt
    return {_normalize_path(path), _normalize_path(decoded.replace("\\", "/"))}


def under_prefix(path: str, prefix: str) -> bool:
    """Segment-aware prefix match: '/app' covers '/app' and '/app/x', not '/application'."""
    base = prefix.rstrip("/")
    return base == "" or path == base or path.startswith(base + "/")


@dataclass
class Scope:
    base_url: str
    allowed_hosts: list[str] = field(default_factory=list)
    allowed_subdomains: list[str] = field(default_factory=list)   # suffixes, e.g. "example.com"
    allowed_prefixes: list[str] = field(default_factory=list)     # URL path prefixes
    excluded_paths: list[str] = field(default_factory=list)
    max_depth: int = 3
    max_requests: int = 300
    rate_limit_rps: float = 5.0
    auth: dict = field(default_factory=dict)                      # only when explicitly supplied
    extra_headers: dict = field(default_factory=dict)
    cookies: list[dict] = field(default_factory=list)
    authorized_external: bool = False   # must be explicitly set to test a non-loopback target
    allowed_test_classes: list[str] = field(default_factory=lambda: list(DEFAULT_TEST_CLASSES))

    def __post_init__(self):
        unknown = set(self.allowed_test_classes) - set(TEST_CLASSES)
        if unknown:
            raise ValueError(f"unknown test classes: {sorted(unknown)}")
        host = urlparse(self.base_url).hostname
        if host and host not in self.allowed_hosts:
            self.allowed_hosts.append(host)
        if not self.allowed_prefixes:
            self.allowed_prefixes = [urlparse(self.base_url).path or "/"]

    def in_scope(self, url: str) -> tuple[bool, str]:
        try:
            p = urlparse(url)
        except Exception:
            return False, "unparseable"
        if p.scheme not in ("http", "https"):
            return False, f"scheme {p.scheme!r}"
        host = p.hostname or ""
        host_ok = host in self.allowed_hosts or any(
            host == d or host.endswith("." + d) for d in self.allowed_subdomains)
        if not host_ok:
            return False, f"host {host!r} not allowed"
        if not self.authorized_external and not is_loopback(host):
            return False, f"host {host!r} is external and the scope is not authorized for it"
        variants = path_variants(p.path)
        excluded = [_normalize_path(x) for x in self.excluded_paths]
        if any(v.startswith(x) for v in variants for x in excluded):
            return False, "excluded path"
        if self.allowed_prefixes and not all(
                any(under_prefix(v, x) for x in self.allowed_prefixes) for v in variants):
            return False, "outside allowed prefixes"
        return True, "in_scope"

    def allows(self, test_class: str) -> bool:
        return test_class in self.allowed_test_classes


@dataclass
class Enforcer:
    """Wraps a Scope with the live request budget, rate limiter, and request/blocked logs."""
    scope: Scope
    requests_made: int = 0
    request_log: list[dict] = field(default_factory=list)
    blocked_log: list[dict] = field(default_factory=list)
    _last_req_t: float = 0.0

    def budget_left(self) -> int:
        return max(0, self.scope.max_requests - self.requests_made)

    def check(self, url: str, kind: str = "get", depth: int | None = None) -> tuple[bool, str]:
        ok, reason = self.scope.in_scope(url)
        if not ok:
            self.blocked_log.append({"url": url, "reason": reason, "kind": kind})
            return False, reason
        if depth is not None and depth > self.scope.max_depth:
            self.blocked_log.append({"url": url, "reason": "max_depth", "kind": kind})
            return False, "max_depth"
        if self.budget_left() <= 0:
            return False, "max_requests"
        return True, "ok"

    def note_request(self, url: str, final_url: str = "", status: int | None = None,
                     kind: str = "get", depth: int | None = None):
        """Record a request and enforce the rate limit. Also checks the FINAL url (post-redirect)
        stayed in scope; an out-of-scope redirect is logged as blocked."""
        now = time.time()
        min_gap = 1.0 / max(0.1, self.scope.rate_limit_rps)
        wait = self._last_req_t + min_gap - now
        if wait > 0:
            time.sleep(wait)
        self._last_req_t = time.time()
        self.requests_made += 1
        rec = {"url": url, "final_url": final_url or url, "status": status,
               "kind": kind, "depth": depth, "n": self.requests_made}
        self.request_log.append(rec)
        if final_url and final_url != url:
            ok, reason = self.scope.in_scope(final_url)
            if not ok:
                self.blocked_log.append({"url": final_url, "reason": f"redirect_out_of_scope:{reason}",
                                         "from": url})
                rec["redirect_blocked"] = True
        return rec
