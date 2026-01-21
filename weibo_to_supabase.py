"""Weibo crawler export -> Supabase row conversion utilities."""

from __future__ import annotations

import html
import json
import re
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

try:
    from zoneinfo import ZoneInfo
except ImportError:  # pragma: no cover - for Python < 3.9
    ZoneInfo = None


SOURCE_NAME = "weibo-crawler-dpz"
SHANGHAI_TZ = ZoneInfo("Asia/Shanghai") if ZoneInfo else timezone(timedelta(hours=8))

USER_NAMESPACE = uuid.uuid5(uuid.NAMESPACE_URL, f"{SOURCE_NAME}:user")
POST_NAMESPACE = uuid.uuid5(uuid.NAMESPACE_URL, f"{SOURCE_NAME}:post")
MEDIA_NAMESPACE = uuid.uuid5(uuid.NAMESPACE_URL, f"{SOURCE_NAME}:media")


def generate_uuid5(namespace: uuid.UUID | str, name: str) -> str:
    """Generate a deterministic UUIDv5 string.

    Args:
        namespace: UUID namespace (uuid.UUID instance or UUID string).
        name: Name input to derive the UUID from (will be coerced to str).

    Returns:
        UUIDv5 string, deterministic for the same namespace + name.
    """
    namespace_uuid = uuid.UUID(namespace) if isinstance(namespace, str) else namespace
    return str(uuid.uuid5(namespace_uuid, str(name)))


def parse_weibo_timestamp(time_str: Optional[str]) -> Optional[str]:
    """Parse a Weibo crawler timestamp string and return a UTC ISO 8601 string.

    The Weibo crawler normalizes time to Asia/Shanghai local time with formats like:
    - "YYYY-MM-DD"
    - "YYYY-MM-DDTHH:MM:SS"
    - "YYYY-MM-DD HH:MM:SS"
    - "YYYY-MM-DDTHH:MM:SS.mmmmmm"
    - "YYYY-MM-DD HH:MM:SS.mmmmmm"

    Args:
        time_str: Raw time string from the crawler. When empty/None, returns None.

    Returns:
        UTC ISO 8601 string with timezone offset (e.g., "2025-09-14T16:10:46+00:00"),
        or None if parsing fails.
    """
    if not time_str or not str(time_str).strip():
        return None

    raw = str(time_str).strip()
    if raw.endswith("Z"):
        raw = f"{raw[:-1]}+00:00"

    dt = _try_parse_datetime(raw)
    if dt is None:
        return None

    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=SHANGHAI_TZ)

    return dt.astimezone(timezone.utc).isoformat()


def build_media_tasks(weibo_item: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Create a media_tasks array from a Weibo item.

    Splits media URLs from:
    - pics (comma-separated)
    - video_url (single URL)
    - live_photo_url (semicolon-separated)

    Args:
        weibo_item: Weibo post dict with media fields and optional IDs.

    Returns:
        A list of media task dicts compatible with the Supabase posts schema.
    """
    tasks: List[Dict[str, Any]] = []
    weibo_post_id = _coerce_id(weibo_item.get("id"))
    weibo_user_id = _coerce_id(weibo_item.get("user_id"))
    media_from = weibo_item.get("_media_from") or weibo_item.get("media_from") or "post"

    pics = _normalize_urls(weibo_item.get("pics"), ",")
    videos = _normalize_urls(weibo_item.get("video_url"), ",")
    live_photos = _normalize_urls(weibo_item.get("live_photo_url"), ";")

    tasks.extend(
        _build_media_task_items(
            kind="image",
            urls=pics,
            weibo_post_id=weibo_post_id,
            weibo_user_id=weibo_user_id,
            media_from=media_from,
        )
    )
    tasks.extend(
        _build_media_task_items(
            kind="video",
            urls=videos,
            weibo_post_id=weibo_post_id,
            weibo_user_id=weibo_user_id,
            media_from=media_from,
        )
    )
    tasks.extend(
        _build_media_task_items(
            kind="live_photo",
            urls=live_photos,
            weibo_post_id=weibo_post_id,
            weibo_user_id=weibo_user_id,
            media_from=media_from,
        )
    )

    return tasks


def convert_user(wb_user: Dict[str, Any], idx: int = 1) -> Dict[str, Any]:
    """Convert a Weibo user dict to a Supabase user row.

    The Weibo-specific fields are stored in the `metadata` JSON string.

    Args:
        wb_user: Weibo user dict (from wb.user).
        idx: 1-based row index for the export batch.

    Returns:
        Dict with Supabase users table columns populated.

    Raises:
        ValueError: If the Weibo user id is missing or empty.
    """
    weibo_user_id = _coerce_id(wb_user.get("id"))
    if not weibo_user_id:
        raise ValueError("Weibo user id is required for UUID mapping.")

    imported_at = datetime.now(timezone.utc).isoformat()
    username = (wb_user.get("screen_name") or "").strip() or f"weibo_{weibo_user_id}"
    avatar_url = wb_user.get("avatar_hd") or wb_user.get("profile_image_url")

    created_at = parse_weibo_timestamp(wb_user.get("registration_time")) or imported_at

    metadata = {
        "source": SOURCE_NAME,
        "imported_at": imported_at,
        "weibo_user_id": weibo_user_id,
        "profile_url": wb_user.get("profile_url") or "",
        "gender": wb_user.get("gender") or "",
        "birthday": wb_user.get("birthday") or "",
        "location": wb_user.get("location") or "",
        "ip_location": wb_user.get("ip_location") or "",
        "education": wb_user.get("education") or "",
        "company": wb_user.get("company") or "",
        "sunshine": wb_user.get("sunshine") or "",
        "registration_time": wb_user.get("registration_time") or "",
        "statuses_count": _to_int(wb_user.get("statuses_count")),
        "followers_count": _to_int(wb_user.get("followers_count")),
        "follow_count": _to_int(wb_user.get("follow_count")),
        "description": wb_user.get("description") or "",
        "verified": bool(wb_user.get("verified")),
        "verified_type": _to_int(wb_user.get("verified_type"), default=-1),
        "verified_reason": wb_user.get("verified_reason") or "",
        "urank": _to_int(wb_user.get("urank")),
        "mbrank": _to_int(wb_user.get("mbrank")),
        "profile_image_url": wb_user.get("profile_image_url") or "",
        "avatar_hd": wb_user.get("avatar_hd") or "",
        "avatar_url": avatar_url,
        "full_avatar_url": wb_user.get("avatar_hd") or avatar_url,
    }

    return {
        "idx": idx,
        "id": generate_uuid5(USER_NAMESPACE, weibo_user_id),
        "username": username,
        "full_name": None,
        "first_name": None,
        "email": None,
        "avatar_url": avatar_url,
        "created_at": created_at,
        "updated_at": imported_at,
        "last_activity_at": None,
        "invitation_code": f"{idx:06d}",
        "expo_token": None,
        "onboard_finished": False,
        "hide": False,
        "metadata": json.dumps(metadata, ensure_ascii=False),
    }


def convert_post(
    wb_weibo_item: Dict[str, Any],
    user_id_map: Dict[str, str],
    idx: int = 1,
) -> Dict[str, Any]:
    """Convert a Weibo post dict to a Supabase post row.

    This flattens retweets into a single content string and merges media tasks.

    Args:
        wb_weibo_item: Weibo post dict (from wb.weibo list).
        user_id_map: Mapping of Weibo user id (string) -> Supabase user UUID string.
        idx: 1-based row index for the export batch.

    Returns:
        Dict with Supabase posts table columns populated.

    Raises:
        ValueError: If the Weibo post id or user mapping is missing.
    """
    weibo_post_id = _coerce_id(wb_weibo_item.get("id"))
    if not weibo_post_id:
        raise ValueError("Weibo post id is required for UUID mapping.")

    weibo_user_id = _coerce_id(wb_weibo_item.get("user_id"))
    if not weibo_user_id:
        raise ValueError("Weibo user id is required for post mapping.")

    user_id = user_id_map.get(weibo_user_id) or user_id_map.get(str(weibo_user_id))
    if not user_id:
        raise ValueError(f"User id mapping missing for Weibo user id: {weibo_user_id}")

    imported_at = datetime.now(timezone.utc).isoformat()
    created_at = (
        parse_weibo_timestamp(wb_weibo_item.get("full_created_at"))
        or parse_weibo_timestamp(wb_weibo_item.get("created_at"))
        or imported_at
    )

    raw_text = wb_weibo_item.get("text") or ""
    retweet = wb_weibo_item.get("retweet") or {}
    raw_retweet_text = retweet.get("text") or ""

    content = _flatten_text(
        _strip_html(raw_text),
        _strip_html(raw_retweet_text),
        retweet.get("screen_name") or "",
    )

    html_content = _flatten_html(
        raw_text,
        raw_retweet_text,
        retweet.get("screen_name") or "",
    )

    media_tasks = build_media_tasks(wb_weibo_item)
    if retweet:
        retweet_copy = dict(retweet)
        retweet_copy["_media_from"] = "retweet"
        media_tasks.extend(build_media_tasks(retweet_copy))

    summary = None
    llm_analysis = wb_weibo_item.get("llm_analysis") or {}
    if isinstance(llm_analysis, dict):
        summary = llm_analysis.get("summary")

    return {
        "idx": idx,
        "id": generate_uuid5(POST_NAMESPACE, weibo_post_id),
        "user_id": user_id,
        "content": content or "",
        "type": "text",
        "created_at": created_at,
        "html_content": html_content,
        "summary": summary,
        "purpose_y": None,
        "focus_domain_x": None,
        "status": None,
        "mongodb_id": weibo_post_id,
        "media_tasks": media_tasks,
        "task_ids": None,
    }


def convert_weibo_export(wb_data: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    """Convert a Weibo crawler export into Supabase-ready users + posts.

    Args:
        wb_data: Dict containing wb.user and wb.weibo fields, for example:
            {
              "user": { ... },
              "weibo": [ ... ]
            }

    Returns:
        Dict with "users" and "posts" lists, ready for insertion.
    """
    users: List[Dict[str, Any]] = []
    posts: List[Dict[str, Any]] = []
    user_id_map: Dict[str, str] = {}

    wb_user = wb_data.get("user") or {}
    if wb_user:
        try:
            user_row = convert_user(wb_user, idx=1)
        except ValueError:
            user_row = None
        if user_row:
            users.append(user_row)
            weibo_user_id = _coerce_id(wb_user.get("id"))
            if weibo_user_id:
                user_id_map[weibo_user_id] = user_row["id"]

    wb_weibo_list = wb_data.get("weibo") or []
    for idx, wb_weibo_item in enumerate(wb_weibo_list, start=1):
        try:
            post_row = convert_post(wb_weibo_item, user_id_map, idx=idx)
        except ValueError:
            continue
        posts.append(post_row)

    _attach_last_activity_at(users, posts)

    return {"users": users, "posts": posts}


def _try_parse_datetime(raw: str) -> Optional[datetime]:
    try:
        return datetime.fromisoformat(raw)
    except (TypeError, ValueError):
        pass

    patterns = [
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%dT%H:%M:%S.%f",
        "%Y-%m-%d %H:%M:%S.%f",
        "%Y-%m-%d",
        "%Y-%m-%dT%H:%M",
        "%Y-%m-%d %H:%M",
    ]
    for fmt in patterns:
        try:
            return datetime.strptime(raw, fmt)
        except ValueError:
            continue
    return None


def _normalize_urls(value: Any, separator: str) -> List[str]:
    if not value:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str):
        return [item.strip() for item in value.split(separator) if item.strip()]
    return [str(value).strip()] if str(value).strip() else []


def _build_media_task_items(
    kind: str,
    urls: List[str],
    weibo_post_id: str,
    weibo_user_id: str,
    media_from: str,
) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    for idx, url in enumerate(urls):
        metadata = {
            "source": "weibo",
            "kind": kind,
            "original_url": url,
            "weibo_post_id": weibo_post_id,
            "weibo_user_id": weibo_user_id,
            "index": idx,
            "from": media_from,
        }
        items.append(
            {
                "task_id": None,
                "media_id": generate_uuid5(MEDIA_NAMESPACE, url),
                "metadata": metadata,
                "media_url": url,
                "media_type": "simple",
                "media_sub_type": "rich_text",
            }
        )
    return items


def _looks_like_html(text: str) -> bool:
    return bool(re.search(r"<[^>]+>", text or ""))


def _strip_html(text: str) -> str:
    if not text:
        return ""
    if not _looks_like_html(text):
        return text.strip()
    stripped = re.sub(r"<[^>]+>", "", text)
    return html.unescape(stripped).strip()


def _flatten_text(main_text: str, retweet_text: str, retweet_name: str) -> str:
    retweet_line = ""
    if retweet_text:
        if retweet_name:
            retweet_line = f"// @{retweet_name}: {retweet_text}"
        else:
            retweet_line = f"// {retweet_text}"
    elif retweet_name:
        retweet_line = f"// @{retweet_name}"

    if main_text and retweet_line:
        return f"{main_text}\n\n{retweet_line}"
    return main_text or retweet_line


def _flatten_html(main_html: str, retweet_html: str, retweet_name: str) -> Optional[str]:
    if not (_looks_like_html(main_html) or _looks_like_html(retweet_html)):
        return None

    parts: List[str] = []
    if main_html:
        if _looks_like_html(main_html):
            parts.append(main_html)
        else:
            parts.append(f"<p>{html.escape(main_html)}</p>")

    if retweet_html or retweet_name:
        header = f"// @{retweet_name}:" if retweet_name else "//"
        header_html = html.escape(header)
        body = ""
        if retweet_html:
            if _looks_like_html(retweet_html):
                body = retweet_html
            else:
                body = f"<p>{html.escape(retweet_html)}</p>"
        parts.append(f"<blockquote><p>{header_html}</p>{body}</blockquote>")

    return "<br><br>".join(part for part in parts if part)


def _coerce_id(value: Any) -> str:
    if value is None:
        return ""
    value_str = str(value).strip()
    return value_str


def _to_int(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def _parse_iso_datetime(value: Optional[str]) -> Optional[datetime]:
    if not value or not str(value).strip():
        return None
    raw = str(value).strip()
    if raw.endswith("Z"):
        raw = f"{raw[:-1]}+00:00"
    try:
        return datetime.fromisoformat(raw)
    except ValueError:
        return None


def _attach_last_activity_at(
    users: List[Dict[str, Any]], posts: List[Dict[str, Any]]
) -> None:
    if not users or not posts:
        return

    latest_by_user: Dict[str, datetime] = {}
    for post in posts:
        user_id = post.get("user_id")
        created_at = _parse_iso_datetime(post.get("created_at"))
        if not user_id or not created_at:
            continue
        prev = latest_by_user.get(user_id)
        if not prev or created_at > prev:
            latest_by_user[user_id] = created_at

    for user in users:
        user_id = user.get("id")
        if user_id in latest_by_user:
            user["last_activity_at"] = latest_by_user[user_id].isoformat()


if __name__ == "__main__":
    example_export = {
        "user": {
            "id": "1234567890",
            "screen_name": "example_user",
            "registration_time": "2020-01-01",
            "profile_image_url": "https://example.com/avatar.jpg",
            "avatar_hd": "",
            "followers_count": 10,
            "follow_count": 2,
            "statuses_count": 3,
        },
        "weibo": [
            {
                "id": 9876543210,
                "user_id": "1234567890",
                "text": "Hello from Weibo!",
                "created_at": "2020-01-02T12:30:00",
                "pics": "https://example.com/image1.jpg,https://example.com/image2.jpg",
                "video_url": "",
                "live_photo_url": "",
            }
        ],
    }

    converted = convert_weibo_export(example_export)
    print(json.dumps(converted, ensure_ascii=True, indent=2))
