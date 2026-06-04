from __future__ import annotations

import json
import logging
import re
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

from app_config import APP_NAME


LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class ReleaseAsset:
    name: str
    url: str
    size: int


@dataclass(frozen=True)
class ReleaseInfo:
    tag: str
    version: str
    html_url: str
    published_at: str
    body: str
    assets: tuple[ReleaseAsset, ...]


def _normalize_version(value: str) -> str:
    text = str(value or "").strip()
    if text.startswith("v"):
        text = text[1:]
    return text


def _parse_version_tuple(value: str) -> tuple[int, int, int]:
    text = _normalize_version(value)
    if not text:
        return (0, 0, 0)
    parts = []
    for token in text.split("."):
        match = re.match(r"(\d+)", token)
        parts.append(int(match.group(1)) if match else 0)
    while len(parts) < 3:
        parts.append(0)
    return tuple(parts[:3])


def is_newer_version(candidate: str, current: str) -> bool:
    return _parse_version_tuple(candidate) > _parse_version_tuple(current)


def fetch_latest_release(repo: str, timeout: float = 4.0) -> Optional[ReleaseInfo]:
    repo = (repo or "").strip()
    if not repo:
        return None

    url = f"https://api.github.com/repos/{repo}/releases/latest"
    request = urllib.request.Request(
        url,
        headers={
            "Accept": "application/vnd.github+json",
            "User-Agent": f"{APP_NAME}-updater",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        LOGGER.info("Update check HTTP error (%s): %s", exc.code, exc.reason)
        return None
    except urllib.error.URLError as exc:
        LOGGER.info("Update check URL error: %s", exc.reason)
        return None
    except Exception as exc:
        LOGGER.info("Update check failed: %s", exc)
        return None

    tag = str(payload.get("tag_name") or "").strip()
    if not tag:
        return None

    assets: list[ReleaseAsset] = []
    for item in payload.get("assets", []) or []:
        name = str(item.get("name") or "").strip()
        download_url = str(item.get("browser_download_url") or "").strip()
        if not name or not download_url:
            continue
        assets.append(
            ReleaseAsset(
                name=name,
                url=download_url,
                size=int(item.get("size") or 0),
            )
        )

    return ReleaseInfo(
        tag=tag,
        version=_normalize_version(tag),
        html_url=str(payload.get("html_url") or "").strip(),
        published_at=str(payload.get("published_at") or "").strip(),
        body=str(payload.get("body") or "").strip(),
        assets=tuple(assets),
    )


def get_available_update(repo: str, current_version: str, timeout: float = 4.0) -> Optional[ReleaseInfo]:
    latest = fetch_latest_release(repo, timeout=timeout)
    if not latest:
        return None
    if is_newer_version(latest.version, current_version):
        return latest
    return None


def preferred_asset_suffix() -> str:
    if sys.platform.startswith("win"):
        return "-win.zip"
    if sys.platform.startswith("linux"):
        return "-linux.zip"
    if sys.platform == "darwin":
        return "-mac.zip"
    return ".zip"


def select_preferred_asset(assets: Iterable[ReleaseAsset]) -> Optional[ReleaseAsset]:
    suffix = preferred_asset_suffix()
    for asset in assets:
        if asset.name.endswith(suffix):
            return asset
    for asset in assets:
        if asset.name.endswith(".zip"):
            return asset
    return None


def download_release_asset(url: str, destination: Path, timeout: float = 30.0) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(
        url,
        headers={"User-Agent": f"{APP_NAME}-updater"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = response.read()
    destination.write_bytes(payload)
    return destination
