from pathlib import Path

cache_path = Path("app/services/material_cache.py")
cache_text = cache_path.read_text(encoding="utf-8")

old = "_CACHE_FORMAT_VERSION = 2"
new = "_CACHE_FORMAT_VERSION = 3"
assert old in cache_text, "cache format version marker not found"
cache_text = cache_text.replace(old, new, 1)

needle = '''    return urlunsplit((parsed.scheme, parsed.netloc, parsed.path, "", ""))\n\n\ndef _cached_source_info(item: MaterialInfo) -> dict | None:\n'''
replacement = '''    return urlunsplit((parsed.scheme, parsed.netloc, parsed.path, "", ""))\n\n\n_ALLOWED_PREVIEW_HOST_SUFFIXES = (\n    "pexels.com",\n    "pixabay.com",\n    "coverr.co",\n)\n_MAX_CACHED_PREVIEW_IMAGES = 4\n\n\ndef _safe_preview_url(value) -> str | None:\n    """Keep only public stock-provider preview URLs and strip query credentials."""\n    sanitized = _safe_public_url(value)\n    if not sanitized:\n        return None\n    try:\n        hostname = (urlsplit(sanitized).hostname or "").lower().rstrip(".")\n    except ValueError:\n        return None\n    if not any(\n        hostname == suffix or hostname.endswith(f".{suffix}")\n        for suffix in _ALLOWED_PREVIEW_HOST_SUFFIXES\n    ):\n        return None\n    return sanitized\n\n\ndef _cached_source_info(item: MaterialInfo) -> dict | None:\n'''
assert needle in cache_text, "safe URL insertion marker not found"
cache_text = cache_text.replace(needle, replacement, 1)

needle = '''        if rendition:\n            cached["rendition"] = rendition\n    return cached\n'''
replacement = '''        if rendition:\n            cached["rendition"] = rendition\n\n    raw_previews = source.get("preview_images")\n    if isinstance(raw_previews, list):\n        previews: list[str] = []\n        for value in raw_previews:\n            preview = _safe_preview_url(value)\n            if not preview or preview in previews:\n                continue\n            previews.append(preview)\n            if len(previews) >= _MAX_CACHED_PREVIEW_IMAGES:\n                break\n        if previews:\n            cached["preview_images"] = previews\n    return cached\n'''
assert needle in cache_text, "cached source info marker not found"
cache_text = cache_text.replace(needle, replacement, 1)
cache_path.write_text(cache_text, encoding="utf-8")

test_path = Path("test/services/test_material_cache.py")
test_text = test_path.read_text(encoding="utf-8")

needle = '''                "rendition": {\n                    "id": "large",\n                    "width": 1080,\n                    "height": 1920,\n                },\n            },\n'''
replacement = '''                "rendition": {\n                    "id": "large",\n                    "width": 1080,\n                    "height": 1920,\n                },\n                "preview_images": [\n                    "https://cdn.pixabay.com/video/2026/09/07/preview.jpg?token=drop",\n                ],\n            },\n'''
assert needle in test_text, "test material fixture marker not found"
test_text = test_text.replace(needle, replacement, 1)

needle = '''        self.assertEqual(\n            loaded[0].source_info["creator"]["profile_page"],\n            "https://pixabay.com/users/creator-456/",\n        )\n\n    def test_expired_cache_is_removed_and_treated_as_miss(self):\n'''
replacement = '''        self.assertEqual(\n            loaded[0].source_info["creator"]["profile_page"],\n            "https://pixabay.com/users/creator-456/",\n        )\n        self.assertEqual(\n            loaded[0].source_info["preview_images"],\n            ["https://cdn.pixabay.com/video/2026/09/07/preview.jpg"],\n        )\n\n    def test_expired_cache_is_removed_and_treated_as_miss(self):\n'''
assert needle in test_text, "round-trip assertion marker not found"
test_text = test_text.replace(needle, replacement, 1)

needle = '''    def test_cache_key_separates_provider_duration_and_aspect(self):\n'''
insert = '''    def test_version_two_cache_is_invalidated(self):\n        """V2 omitted semantic preview URLs, so it must refresh from the provider."""\n        cache_path = self._cache_path()\n        cache_path.write_text(\n            json.dumps(\n                {\n                    "version": 2,\n                    "items": [\n                        {\n                            "provider": "pixabay",\n                            "url": "https://example.com/old-v2.mp4",\n                            "duration": 12,\n                            "source_info": {\n                                "provider": "pixabay",\n                                "asset_id": "old-v2",\n                                "rendition": {\n                                    "id": "large",\n                                    "width": 1080,\n                                    "height": 1920,\n                                },\n                            },\n                        }\n                    ],\n                }\n            ),\n            encoding="utf-8",\n        )\n\n        loaded = material_cache.load_material_search_cache(\n            provider="pixabay",\n            search_term="nature",\n            minimum_duration=5,\n            video_aspect=VideoAspect.portrait,\n        )\n\n        self.assertIsNone(loaded)\n        self.assertFalse(cache_path.exists())\n\n    def test_cache_key_separates_provider_duration_and_aspect(self):\n'''
assert needle in test_text, "version-two test insertion marker not found"
test_text = test_text.replace(needle, insert, 1)

test_path.write_text(test_text, encoding="utf-8")
