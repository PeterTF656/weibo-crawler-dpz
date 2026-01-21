"""Validation script for weibo_to_supabase conversions."""

from __future__ import annotations

import json
import sys
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import weibo_to_supabase as wts


class Reporter:
    def __init__(self) -> None:
        self.failures: List[str] = []
        self.notes: List[str] = []
        self.checks = 0

    def check(self, condition: bool, message: str) -> None:
        self.checks += 1
        if not condition:
            self.failures.append(message)

    def note(self, message: str) -> None:
        self.notes.append(message)


def _is_uuid(value: Any) -> bool:
    if not value or not isinstance(value, str):
        return False
    try:
        uuid.UUID(value)
        return True
    except (ValueError, TypeError):
        return False


def _parse_iso(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    raw = value.strip()
    if raw.endswith("Z"):
        raw = f"{raw[:-1]}+00:00"
    try:
        return datetime.fromisoformat(raw)
    except ValueError:
        return None


def _is_utc_isoformat(value: Optional[str]) -> bool:
    dt = _parse_iso(value)
    if not dt or not dt.tzinfo:
        return False
    return dt.utcoffset() == timedelta(0)


def _require_keys(
    report: Reporter, payload: Dict[str, Any], keys: List[str], context: str
) -> None:
    for key in keys:
        report.check(key in payload, f"{context}: missing field '{key}'")


def _validate_media_tasks(
    report: Reporter, media_tasks: Any, context: str
) -> None:
    report.check(isinstance(media_tasks, list), f"{context}: media_tasks not a list")
    if not isinstance(media_tasks, list):
        return
    for idx, task in enumerate(media_tasks):
        entry_context = f"{context}: media_tasks[{idx}]"
        report.check(isinstance(task, dict), f"{entry_context} not a dict")
        if not isinstance(task, dict):
            continue
        _require_keys(
            report,
            task,
            ["task_id", "media_id", "metadata", "media_url", "media_type", "media_sub_type"],
            entry_context,
        )
        report.check(_is_uuid(task.get("media_id")), f"{entry_context} media_id not UUID")
        report.check(
            isinstance(task.get("media_url"), str),
            f"{entry_context} media_url not string",
        )
        report.check(
            task.get("media_type") == "simple",
            f"{entry_context} media_type not 'simple'",
        )
        report.check(
            task.get("media_sub_type") == "rich_text",
            f"{entry_context} media_sub_type not 'rich_text'",
        )
        metadata = task.get("metadata")
        if metadata is None:
            continue
        report.check(isinstance(metadata, dict), f"{entry_context} metadata not dict")
        if isinstance(metadata, dict):
            _require_keys(
                report,
                metadata,
                ["source", "kind", "original_url", "weibo_post_id", "weibo_user_id", "index", "from"],
                entry_context,
            )
            report.check(
                metadata.get("from") in {"post", "retweet"},
                f"{entry_context} metadata.from invalid",
            )


def _validate_user_row(report: Reporter, row: Dict[str, Any], context: str) -> None:
    required = [
        "idx",
        "id",
        "username",
        "full_name",
        "first_name",
        "email",
        "avatar_url",
        "created_at",
        "updated_at",
        "last_activity_at",
        "invitation_code",
        "expo_token",
        "onboard_finished",
        "hide",
        "metadata",
    ]
    _require_keys(report, row, required, context)
    report.check(_is_uuid(row.get("id")), f"{context}: id not UUID")
    report.check(
        isinstance(row.get("username"), str) and row.get("username"),
        f"{context}: username empty",
    )
    report.check(
        _is_utc_isoformat(row.get("created_at")),
        f"{context}: created_at not UTC ISO",
    )
    report.check(
        _is_utc_isoformat(row.get("updated_at")),
        f"{context}: updated_at not UTC ISO",
    )
    metadata_raw = row.get("metadata")
    report.check(isinstance(metadata_raw, str), f"{context}: metadata not string")
    if isinstance(metadata_raw, str):
        try:
            metadata = json.loads(metadata_raw)
        except json.JSONDecodeError:
            report.check(False, f"{context}: metadata not valid JSON")
        else:
            _require_keys(report, metadata, ["source", "weibo_user_id", "imported_at"], context)
            report.check(
                metadata.get("source") == wts.SOURCE_NAME,
                f"{context}: metadata.source mismatch",
            )


def _validate_post_row(report: Reporter, row: Dict[str, Any], context: str) -> None:
    required = [
        "idx",
        "id",
        "user_id",
        "content",
        "type",
        "created_at",
        "html_content",
        "summary",
        "purpose_y",
        "focus_domain_x",
        "status",
        "mongodb_id",
        "media_tasks",
        "task_ids",
    ]
    _require_keys(report, row, required, context)
    report.check(_is_uuid(row.get("id")), f"{context}: id not UUID")
    report.check(_is_uuid(row.get("user_id")), f"{context}: user_id not UUID")
    report.check(
        _is_utc_isoformat(row.get("created_at")),
        f"{context}: created_at not UTC ISO",
    )
    report.check(isinstance(row.get("content"), str), f"{context}: content not string")
    report.check(isinstance(row.get("mongodb_id"), str), f"{context}: mongodb_id not string")
    _validate_media_tasks(report, row.get("media_tasks"), context)


def _find_weibo_exports(root: Path) -> List[Path]:
    if not root.exists():
        return []
    return [
        path
        for path in root.rglob("*.json")
        if path.is_file() and path.name != "schema.json"
    ]


def test_real_exports(report: Reporter) -> None:
    export_paths = _find_weibo_exports(Path("weibo"))
    if not export_paths:
        report.note("No weibo JSON exports found; skipped real-data tests.")
        return

    for path in export_paths:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            report.check(False, f"{path}: invalid JSON")
            continue

        report.check(isinstance(data, dict), f"{path}: export not dict")
        if not isinstance(data, dict):
            continue

        converted = wts.convert_weibo_export(data)
        report.check(isinstance(converted, dict), f"{path}: conversion not dict")
        users = converted.get("users")
        posts = converted.get("posts")
        report.check(isinstance(users, list), f"{path}: users not list")
        report.check(isinstance(posts, list), f"{path}: posts not list")

        if data.get("user"):
            report.check(users, f"{path}: expected user row")

        for user in users or []:
            _validate_user_row(report, user, f"{path}: user")

        for post in posts or []:
            _validate_post_row(report, post, f"{path}: post")

        if users and len(users) == 1 and posts:
            user_id = users[0].get("id")
            for post in posts:
                report.check(
                    post.get("user_id") == user_id,
                    f"{path}: post user_id mismatch",
                )

            latest = max(
                (_parse_iso(p.get("created_at")) for p in posts),
                default=None,
            )
            if latest:
                report.check(
                    users[0].get("last_activity_at") == latest.isoformat(),
                    f"{path}: last_activity_at mismatch",
                )

        converted_again = wts.convert_weibo_export(data)
        ids_first = [row.get("id") for row in converted.get("users", [])]
        ids_second = [row.get("id") for row in converted_again.get("users", [])]
        report.check(ids_first == ids_second, f"{path}: user UUIDs not deterministic")
        post_ids_first = [row.get("id") for row in converted.get("posts", [])]
        post_ids_second = [row.get("id") for row in converted_again.get("posts", [])]
        report.check(post_ids_first == post_ids_second, f"{path}: post UUIDs not deterministic")

        weibo_list = data.get("weibo") or []
        for wb_item, post_row in zip(weibo_list, posts or []):
            retweet = wb_item.get("retweet") or {}
            if retweet and (retweet.get("text") or retweet.get("screen_name")):
                report.check(
                    "//" in post_row.get("content", ""),
                    f"{path}: retweet content missing flatten marker",
                )


def test_field_mapping_and_determinism(report: Reporter) -> None:
    wb_user = {"id": "123", "screen_name": "", "registration_time": "2020-01-01"}
    user_row = wts.convert_user(wb_user, idx=5)
    user_row_again = wts.convert_user(wb_user, idx=6)
    report.check(user_row["id"] == user_row_again["id"], "User UUID not deterministic")
    report.check(
        user_row["username"] == "weibo_123",
        "Username fallback not used for empty screen_name",
    )

    user_map = {"123": user_row["id"]}
    wb_post = {
        "id": "999",
        "user_id": "123",
        "text": "hello",
        "created_at": "2020-01-02 12:30:00",
    }
    post_row = wts.convert_post(wb_post, user_map, idx=1)
    post_row_again = wts.convert_post(wb_post, user_map, idx=2)
    report.check(post_row["id"] == post_row_again["id"], "Post UUID not deterministic")
    report.check(_is_utc_isoformat(post_row["created_at"]), "Post created_at not UTC ISO")
    _validate_user_row(report, user_row, "field-mapping: user")
    _validate_post_row(report, post_row, "field-mapping: post")


def test_retweet_handling(report: Reporter) -> None:
    wb_user = {"id": "42", "screen_name": "poster"}
    user_row = wts.convert_user(wb_user, idx=1)
    user_map = {"42": user_row["id"]}
    wb_post = {
        "id": "100",
        "user_id": "42",
        "text": "Main <b>post</b> &amp; stuff",
        "created_at": "2024-01-02 03:04:05",
        "pics": "http://example.com/a.jpg",
        "retweet": {
            "text": "Retweet <i>text</i>",
            "screen_name": "rt_user",
            "pics": "http://example.com/rt.jpg",
            "video_url": "http://example.com/rt.mp4",
        },
    }
    post_row = wts.convert_post(wb_post, user_map, idx=1)
    content = post_row.get("content", "")
    report.check("Main post & stuff" in content, "Retweet flatten stripped main text mismatch")
    report.check("// @rt_user" in content, "Retweet flatten missing marker")
    report.check(
        post_row.get("html_content") is not None,
        "Retweet html_content missing when HTML present",
    )

    media_tasks = post_row.get("media_tasks", [])
    report.check(len(media_tasks) == 3, "Retweet media tasks count mismatch")
    retweet_media = [
        task
        for task in media_tasks
        if isinstance(task, dict) and task.get("metadata", {}).get("from") == "retweet"
    ]
    report.check(len(retweet_media) == 2, "Retweet media not included")


def test_edge_cases(report: Reporter) -> None:
    try:
        wts.convert_user({})
    except ValueError:
        report.check(True, "convert_user missing id raises")
    else:
        report.check(False, "convert_user missing id did not raise")

    try:
        wts.convert_post({"user_id": "1"}, {"1": "x"}, idx=1)
    except ValueError:
        report.check(True, "convert_post missing post id raises")
    else:
        report.check(False, "convert_post missing post id did not raise")

    try:
        wts.convert_post({"id": "1", "user_id": "1"}, {}, idx=1)
    except ValueError:
        report.check(True, "convert_post missing user mapping raises")
    else:
        report.check(False, "convert_post missing user mapping did not raise")

    wb_user = {"id": "77", "screen_name": "tester"}
    user_row = wts.convert_user(wb_user, idx=1)
    user_map = {"77": user_row["id"]}
    wb_post = {"id": "200", "user_id": "77", "text": "", "created_at": ""}
    post_row = wts.convert_post(wb_post, user_map, idx=1)
    report.check(post_row.get("content") == "", "Empty text not preserved")


def test_timestamp_handling(report: Reporter) -> None:
    report.check(wts.parse_weibo_timestamp("not-a-time") is None, "Invalid timestamp parsed")
    report.check(
        wts.parse_weibo_timestamp("2020-01-02 12:30:00") == "2020-01-02T04:30:00+00:00",
        "Shanghai timestamp conversion mismatch",
    )
    report.check(
        wts.parse_weibo_timestamp("2020-01-02T12:30:00") == "2020-01-02T04:30:00+00:00",
        "Shanghai timestamp conversion mismatch (T format)",
    )
    report.check(
        wts.parse_weibo_timestamp("2020-01-02") == "2020-01-01T16:00:00+00:00",
        "Date-only timestamp conversion mismatch",
    )
    report.check(
        wts.parse_weibo_timestamp("2020-01-02T12:30:00Z")
        == "2020-01-02T12:30:00+00:00",
        "Z timestamp conversion mismatch",
    )

    wb_user = {"id": "88", "screen_name": "tester"}
    user_row = wts.convert_user(wb_user, idx=1)
    user_map = {"88": user_row["id"]}
    wb_post = {"id": "300", "user_id": "88", "created_at": "bad-date"}
    start = datetime.now(timezone.utc) - timedelta(seconds=2)
    post_row = wts.convert_post(wb_post, user_map, idx=1)
    end = datetime.now(timezone.utc) + timedelta(seconds=2)
    created_at = _parse_iso(post_row.get("created_at"))
    report.check(created_at is not None, "Malformed timestamp fallback missing")
    if created_at:
        report.check(start <= created_at <= end, "Malformed timestamp fallback outside window")


def main() -> int:
    report = Reporter()

    test_real_exports(report)
    test_field_mapping_and_determinism(report)
    test_retweet_handling(report)
    test_edge_cases(report)
    test_timestamp_handling(report)

    print(f"Checks: {report.checks}")
    if report.notes:
        print("Notes:")
        for note in report.notes:
            print(f"- {note}")
    if report.failures:
        print(f"Failures: {len(report.failures)}")
        for idx, failure in enumerate(report.failures, start=1):
            print(f"{idx}. {failure}")
        return 1
    print("Failures: 0")
    return 0


if __name__ == "__main__":
    sys.exit(main())
