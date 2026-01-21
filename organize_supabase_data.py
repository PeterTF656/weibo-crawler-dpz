"""Reorganize Supabase data exports into a structured layout."""

from __future__ import annotations

import argparse
import json
import logging
import logging.config
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


DEFAULT_INPUT_DIR = "supabase/data"
DEFAULT_OUTPUT_FORMAT = "both"

BY_USER_DIRNAME = "by-user"
COMBINED_DIRNAME = "combined"
BY_USER_USER_FILENAME = "user.json"
BY_USER_POSTS_FILENAME = "posts.json"
BY_USER_METADATA_FILENAME = "metadata.json"
COMBINED_USERS_FILENAME = "users.json"
COMBINED_POSTS_FILENAME = "posts.json"
INDEX_FILENAME = "index.json"

logger = logging.getLogger("organize_supabase_data")


def configure_logging() -> None:
    config_path = Path(__file__).with_name("logging.conf")
    if config_path.exists():
        logging.config.fileConfig(config_path, disable_existing_loggers=False)
    else:
        logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Input file not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {path}: {exc}") from exc


def _write_json(path: Path, payload: Any, pretty: bool, dry_run: bool) -> None:
    if dry_run:
        logger.info("Dry run: would write %s", path)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2 if pretty else None)


def _extract_user_id(path: Path, prefix: str) -> Optional[str]:
    name = path.name
    if not (name.startswith(prefix) and name.endswith(".json")):
        return None
    return name[len(prefix) : -len(".json")]


def _collect_source_files(
    input_dir: Path,
) -> Tuple[Dict[str, Path], Dict[str, Path]]:
    user_files: Dict[str, Path] = {}
    post_files: Dict[str, Path] = {}
    for path in input_dir.glob("supabase_user_*.json"):
        user_id = _extract_user_id(path, "supabase_user_")
        if user_id:
            user_files[user_id] = path
    for path in input_dir.glob("supabase_posts_*.json"):
        user_id = _extract_user_id(path, "supabase_posts_")
        if user_id:
            post_files[user_id] = path
    return user_files, post_files


def _parse_metadata_field(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str) and value.strip():
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return {}
    return {}


def _extract_user_row(payload: Any) -> Optional[Dict[str, Any]]:
    if isinstance(payload, list):
        for item in payload:
            if isinstance(item, dict):
                return item
        return None
    if isinstance(payload, dict):
        return payload
    return None


def _as_list(payload: Any) -> List[Any]:
    if payload is None:
        return []
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        return [payload]
    logger.warning("Skipping non-list payload while combining: %s", type(payload))
    return []


def _count_rows(payload: Any) -> int:
    if payload is None:
        return 0
    if isinstance(payload, list):
        return len(payload)
    if isinstance(payload, dict):
        return 1
    return 0


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


def _post_date_range(posts_payload: Any) -> Optional[Dict[str, str]]:
    if not isinstance(posts_payload, list):
        return None
    parsed: List[datetime] = []
    for post in posts_payload:
        if not isinstance(post, dict):
            continue
        created_at = _parse_iso_datetime(post.get("created_at"))
        if created_at:
            parsed.append(created_at)
    if not parsed:
        return None
    return {"min": min(parsed).isoformat(), "max": max(parsed).isoformat()}


def _relative_path(path: Optional[Path], base_dir: Path) -> Optional[str]:
    if not path:
        return None
    try:
        return str(path.relative_to(base_dir))
    except ValueError:
        return str(path)


def _build_user_metadata(
    user_id: str,
    user_payload: Any,
    posts_payload: Any,
    source_files: Dict[str, str],
    output_files: Dict[str, str],
    generated_at: str,
    missing_sources: List[str],
) -> Dict[str, Any]:
    user_row = _extract_user_row(user_payload) or {}
    metadata = _parse_metadata_field(user_row.get("metadata"))
    if metadata.get("weibo_user_id") and str(metadata.get("weibo_user_id")) != user_id:
        logger.warning(
            "User id mismatch for %s (metadata %s).",
            user_id,
            metadata.get("weibo_user_id"),
        )
    user_metadata = {
        "generated_at": generated_at,
        "weibo_user_id": metadata.get("weibo_user_id") or user_id,
        "supabase_user_id": user_row.get("id"),
        "username": user_row.get("username"),
        "user_rows": _count_rows(user_payload),
        "post_rows": _count_rows(posts_payload),
        "post_date_range": _post_date_range(posts_payload),
        "source_files": source_files,
        "output_files": output_files,
    }
    if missing_sources:
        user_metadata["missing_sources"] = missing_sources
    return user_metadata


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Reorganize Supabase data exports into a structured layout.",
    )
    parser.add_argument(
        "--input-dir",
        default=DEFAULT_INPUT_DIR,
        help="Directory containing supabase_user_*.json and supabase_posts_*.json.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory to write the reorganized structure (default: input dir).",
    )
    parser.add_argument(
        "--format",
        default=DEFAULT_OUTPUT_FORMAT,
        choices=("by-user", "combined", "both"),
        help="Output layout to generate.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview the reorganization without writing files.",
    )
    format_group = parser.add_mutually_exclusive_group()
    format_group.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print JSON output (default).",
    )
    format_group.add_argument(
        "--compact",
        action="store_true",
        help="Write JSON without indentation.",
    )
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    configure_logging()
    parser = _build_parser()
    args = parser.parse_args(argv)

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir) if args.output_dir else input_dir

    user_files, post_files = _collect_source_files(input_dir)
    user_ids = sorted(set(user_files) | set(post_files))

    if not user_ids:
        logger.error("No supabase_user_*.json or supabase_posts_*.json found in %s", input_dir)
        return 2

    generate_by_user = args.format in ("by-user", "both")
    generate_combined = args.format in ("combined", "both")
    pretty = not args.compact

    generated_at = datetime.now(timezone.utc).isoformat()
    combined_users: List[Any] = []
    combined_posts: List[Any] = []
    index_users: List[Dict[str, Any]] = []

    for user_id in user_ids:
        user_file = user_files.get(user_id)
        post_file = post_files.get(user_id)
        missing_sources: List[str] = []

        if not user_file:
            missing_sources.append("user")
            logger.warning("Missing user file for %s", user_id)
        if not post_file:
            missing_sources.append("posts")
            logger.warning("Missing posts file for %s", user_id)

        user_payload = _read_json(user_file) if user_file else []
        posts_payload = _read_json(post_file) if post_file else []

        if generate_combined:
            combined_users.extend(_as_list(user_payload))
            combined_posts.extend(_as_list(posts_payload))

        source_files = {
            key: value
            for key, value in {
                "user": _relative_path(user_file, input_dir),
                "posts": _relative_path(post_file, input_dir),
            }.items()
            if value
        }
        output_files: Dict[str, str] = {}

        if generate_by_user:
            base_dir = output_dir / BY_USER_DIRNAME / user_id
            user_out = base_dir / BY_USER_USER_FILENAME
            posts_out = base_dir / BY_USER_POSTS_FILENAME
            metadata_out = base_dir / BY_USER_METADATA_FILENAME

            _write_json(user_out, user_payload, pretty, args.dry_run)
            _write_json(posts_out, posts_payload, pretty, args.dry_run)
            output_files = {
                "user": _relative_path(user_out, output_dir) or str(user_out),
                "posts": _relative_path(posts_out, output_dir) or str(posts_out),
                "metadata": _relative_path(metadata_out, output_dir) or str(metadata_out),
            }

        user_metadata = _build_user_metadata(
            user_id=user_id,
            user_payload=user_payload,
            posts_payload=posts_payload,
            source_files=source_files,
            output_files=output_files,
            generated_at=generated_at,
            missing_sources=missing_sources,
        )
        index_users.append(user_metadata)

        if generate_by_user:
            metadata_path = output_dir / BY_USER_DIRNAME / user_id / BY_USER_METADATA_FILENAME
            _write_json(metadata_path, user_metadata, pretty, args.dry_run)

    combined_info: Optional[Dict[str, Any]] = None
    if generate_combined:
        combined_dir = output_dir / COMBINED_DIRNAME
        users_path = combined_dir / COMBINED_USERS_FILENAME
        posts_path = combined_dir / COMBINED_POSTS_FILENAME
        _write_json(users_path, combined_users, pretty, args.dry_run)
        _write_json(posts_path, combined_posts, pretty, args.dry_run)
        combined_info = {
            "users_file": _relative_path(users_path, output_dir),
            "posts_file": _relative_path(posts_path, output_dir),
            "user_rows": len(combined_users),
            "post_rows": len(combined_posts),
        }

    total_user_rows = sum(
        meta.get("user_rows", 0)
        for meta in index_users
        if isinstance(meta.get("user_rows"), int)
    )
    total_post_rows = sum(
        meta.get("post_rows", 0)
        for meta in index_users
        if isinstance(meta.get("post_rows"), int)
    )

    index_payload: Dict[str, Any] = {
        "generated_at": generated_at,
        "format_version": 1,
        "source_dir": str(input_dir),
        "output_dir": str(output_dir),
        "format": args.format,
        "user_count": len(user_ids),
        "user_rows": total_user_rows,
        "post_rows": total_post_rows,
        "users": index_users,
    }
    if combined_info:
        index_payload["combined"] = combined_info

    index_path = output_dir / INDEX_FILENAME
    _write_json(index_path, index_payload, pretty, args.dry_run)

    logger.info("Processed %d user ids", len(user_ids))
    if generate_combined:
        logger.info(
            "Combined rows: %d users, %d posts",
            len(combined_users),
            len(combined_posts),
        )
    logger.info("Index written to %s", index_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
