"""Strict resolved user-image ingress; no filesystem or network resolution here.

CLI/gateway image_routing owns local-file preparation. Native stdin accepts only
its inline raster data URLs, never an arbitrary remote URL or a path to fetch.
"""
from __future__ import annotations

import base64
import binascii
import warnings
from io import BytesIO

from agent.image_routing import (
    _UNIVERSALLY_SUPPORTED_MIMES, _sniff_mime_from_bytes,
)

# Claude Code resizes every base64 image block of a stream-json user message itself
# (to the API's pixel and 5 MiB limits) before calling the API, so large screenshots
# and phone photos pass through unchanged, as in the fork (2338745987). These caps
# only protect stdin and memory: the per-image one is the CLI's own 32 MB of base64.
MAX_IMAGE_BASE64_CHARS = 32 * 1024 * 1024
MAX_IMAGE_BYTES = MAX_IMAGE_BASE64_CHARS // 4 * 3
MAX_REQUEST_IMAGE_BYTES = 64 * 1024 * 1024
MAX_IMAGES = 100


def image_block(part: dict) -> dict:
    value = part.get("image_url")
    url = value.get("url") if isinstance(value, dict) else value
    if not isinstance(url, str) or not url.startswith("data:"):
        raise ValueError("Claude native images require resolved base64 data URLs; paths and remote schemes are unsupported")
    header, separator, data = url.partition(",")
    mime = header[5:].removesuffix(";base64")
    mime = "image/jpeg" if mime == "image/jpg" else mime
    if not separator or not header.endswith(";base64") or mime not in _UNIVERSALLY_SUPPORTED_MIMES:
        raise ValueError("Claude native requires supported raster MIME and base64 encoding")
    if len(data) > MAX_IMAGE_BASE64_CHARS:
        raise ValueError("Claude native image exceeds the CLI's 32 MB limit")
    try:
        raw = base64.b64decode(data, validate=True)
    except (ValueError, binascii.Error) as exc:
        raise ValueError("Claude native image has invalid base64 data") from exc
    if not raw or len(raw) > MAX_IMAGE_BYTES:
        raise ValueError("Claude native image is empty or exceeds the CLI's 32 MB limit")
    if _sniff_mime_from_bytes(raw) != mime:
        raise ValueError("Claude native image is corrupt or does not match its MIME")
    # Header sniffing alone does not validate JPEG/GIF/WebP. Require the existing
    # Pillow decoder, fail closed on corruption/bombs, and do not transcode here.
    from PIL import Image
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(BytesIO(raw)) as image:
                image.verify()
            with Image.open(BytesIO(raw)) as image:
                image.load()
    except (OSError, SyntaxError, ValueError, Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
        raise ValueError("Claude native image is corrupt or has unsupported dimensions") from exc
    return {"type": "image", "source": {"type": "base64", "media_type": mime,
                                          "data": base64.b64encode(raw).decode("ascii")}}
