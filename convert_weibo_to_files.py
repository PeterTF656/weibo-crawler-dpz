"""Convert Weibo crawler exports into Supabase-ready JSON files."""

from __future__ import annotations

import argparse
import json
import logging
import logging.config
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import weibo_to_supabase as wts

DEFAULT_USER_OUT = "supabase_user.json"
DEFAULT_POSTS_OUT = "supabase_posts.json"
DEFAULT_OUTPUT_FORMAT = "flat"

BY_USER_DIRNAME = "by-user"
COMBINED_DIRNAME = "combined"
BY_USER_USER_FILENAME = "user.json"
BY_USER_POSTS_FILENAME = "posts.json"
BY_USER_METADATA_FILENAME = "metadata.json"
COMBINED_USERS_FILENAME = "users.json"
COMBINED_POSTS_FILENAME = "posts.json"

logger = logging.getLogger("convert_weibo_to_files")


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


def _extract_user(payload: Any, source: str) -> Dict[str, Any]:
    if isinstance(payload, dict) and isinstance(payload.get("user"), dict):
        return payload["user"]
    if isinstance(payload, dict):
        return payload
    raise ValueError(f"{source} must be a JSON object (wb.user or export.user).")


def _extract_weibo(payload: Any, source: str) -> List[Dict[str, Any]]:
    if isinstance(payload, dict) and isinstance(payload.get("weibo"), list):
        return payload["weibo"]
    if isinstance(payload, list):
        return payload
    raise ValueError(f"{source} must be a JSON array (wb.weibo or export.weibo).")


def _resolve_output_path(output_dir: Path, name: str) -> Path:
    path = Path(name)
    if path.is_absolute():
        return path
    return output_dir / path


def _resolve_weibo_user_id(export: Dict[str, Any]) -> str:
    user = export.get("user") or {}
    user_id = user.get("id")
    if user_id:
        return str(user_id).strip()
    weibo_list = export.get("weibo") or []
    if weibo_list and isinstance(weibo_list, list):
        first = weibo_list[0]
        if isinstance(first, dict) and first.get("user_id"):
            return str(first.get("user_id")).strip()
    return ""


def _parse_metadata_field(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str) and value.strip():
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return {}
    return {}


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


def _post_date_range(posts: List[Dict[str, Any]]) -> Optional[Dict[str, str]]:
    if not posts:
        return None
    parsed: List[datetime] = []
    for post in posts:
        if not isinstance(post, dict):
            continue
        created_at = _parse_iso_datetime(post.get("created_at"))
        if created_at:
            parsed.append(created_at)
    if not parsed:
        return None
    return {"min": min(parsed).isoformat(), "max": max(parsed).isoformat()}


def _build_user_metadata(
    users: List[Dict[str, Any]],
    posts: List[Dict[str, Any]],
    weibo_user_id: str,
) -> Dict[str, Any]:
    user_row = users[0] if users else {}
    metadata = _parse_metadata_field(user_row.get("metadata"))
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "weibo_user_id": metadata.get("weibo_user_id") or weibo_user_id or "",
        "supabase_user_id": user_row.get("id"),
        "username": user_row.get("username"),
        "user_rows": len(users),
        "post_rows": len(posts),
        "post_date_range": _post_date_range(posts),
    }


def _resolve_output_paths(
    output_dir: Path,
    output_format: str,
    user_out: str,
    posts_out: str,
    export: Dict[str, Any],
) -> Tuple[Path, Path, Optional[Path], str]:
    if output_format == "flat":
        user_path = _resolve_output_path(output_dir, user_out)
        posts_path = _resolve_output_path(output_dir, posts_out)
        return user_path, posts_path, None, ""
    if output_format == "by-user":
        weibo_user_id = _resolve_weibo_user_id(export)
        if not weibo_user_id:
            raise ValueError("Weibo user id is required for --format by-user.")
        base_dir = output_dir / BY_USER_DIRNAME / weibo_user_id
        return (
            base_dir / BY_USER_USER_FILENAME,
            base_dir / BY_USER_POSTS_FILENAME,
            base_dir / BY_USER_METADATA_FILENAME,
            weibo_user_id,
        )
    if output_format == "combined":
        base_dir = output_dir / COMBINED_DIRNAME
        return (
            base_dir / COMBINED_USERS_FILENAME,
            base_dir / COMBINED_POSTS_FILENAME,
            None,
            "",
        )
    raise ValueError(f"Unsupported output format: {output_format}")


def _write_json(path: Path, payload: Any, pretty: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2 if pretty else None)


def convert_and_write_export(
    wb_data: Dict[str, Any],
    output_dir: str | Path = ".",
    user_out: str = DEFAULT_USER_OUT,
    posts_out: str = DEFAULT_POSTS_OUT,
    pretty: bool = True,
    output_format: str = DEFAULT_OUTPUT_FORMAT,
) -> Tuple[Path, Path, int, int]:
    """Convert a full Weibo export dict and write Supabase JSON files."""
    output_dir = Path(output_dir)
    user_path, posts_path, metadata_path, weibo_user_id = _resolve_output_paths(
        output_dir,
        output_format,
        user_out,
        posts_out,
        wb_data,
    )

    converted = wts.convert_weibo_export(wb_data)
    users = converted.get("users", [])
    posts = converted.get("posts", [])

    _write_json(user_path, users, pretty)
    _write_json(posts_path, posts, pretty)
    if metadata_path:
        metadata = _build_user_metadata(users, posts, weibo_user_id)
        _write_json(metadata_path, metadata, pretty)
        logger.info("Wrote metadata to %s", metadata_path)

    return user_path, posts_path, len(users), len(posts)


def convert_and_write(
    wb_user: Dict[str, Any],
    wb_weibo: List[Dict[str, Any]],
    output_dir: str | Path = ".",
    user_out: str = DEFAULT_USER_OUT,
    posts_out: str = DEFAULT_POSTS_OUT,
    pretty: bool = True,
    output_format: str = DEFAULT_OUTPUT_FORMAT,
) -> Tuple[Path, Path, int, int]:
    """Convert wb.user + wb.weibo and write Supabase JSON files."""
    export = {"user": wb_user, "weibo": wb_weibo}
    return convert_and_write_export(
        export,
        output_dir=output_dir,
        user_out=user_out,
        posts_out=posts_out,
        pretty=pretty,
        output_format=output_format,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert Weibo crawler JSON into Supabase-ready JSON files.",
        epilog=(
            "Examples:\n"
            "  python convert_weibo_to_files.py --input weibo/123/123.json\n"
            "  python convert_weibo_to_files.py --user-file user.json --weibo-file weibo.json\n"
            "  python convert_weibo_to_files.py --input weibo/123/123.json --output-dir out\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input",
        help="Path to Weibo crawler JSON export containing 'user' and 'weibo'.",
    )
    parser.add_argument(
        "--user-file",
        help="Path to JSON file containing wb.user (or export.user).",
    )
    parser.add_argument(
        "--weibo-file",
        help="Path to JSON file containing wb.weibo list (or export.weibo).",
    )
    parser.add_argument(
        "--output-dir",
        default=".",
        help="Directory to write Supabase JSON output files.",
    )
    parser.add_argument(
        "--user-out",
        default=DEFAULT_USER_OUT,
        help=(
            f"User output filename (default: {DEFAULT_USER_OUT}). "
            "Only used with --format flat."
        ),
    )
    parser.add_argument(
        "--posts-out",
        default=DEFAULT_POSTS_OUT,
        help=(
            f"Posts output filename (default: {DEFAULT_POSTS_OUT}). "
            "Only used with --format flat."
        ),
    )
    parser.add_argument(
        "--format",
        default=DEFAULT_OUTPUT_FORMAT,
        choices=("flat", "by-user", "combined"),
        help=(
            "Output layout: flat (legacy), by-user (grouped directories), "
            "or combined (batch-friendly files)."
        ),
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


def _load_export_from_args(args: argparse.Namespace) -> Dict[str, Any]:
    if args.input:
        payload = _read_json(Path(args.input))
        if not isinstance(payload, dict):
            raise ValueError("--input must be a JSON object with 'user' and 'weibo'.")
        return {
            "user": payload.get("user") or {},
            "weibo": payload.get("weibo") or [],
        }

    if not args.user_file and not args.weibo_file:
        raise ValueError("Provide --input or both --user-file and --weibo-file.")

    if not args.user_file or not args.weibo_file:
        raise ValueError("Both --user-file and --weibo-file are required together.")

    user_payload = _read_json(Path(args.user_file))
    weibo_payload = _read_json(Path(args.weibo_file))
    user = _extract_user(user_payload, "--user-file")
    weibo_list = _extract_weibo(weibo_payload, "--weibo-file")
    return {"user": user, "weibo": weibo_list}


def main(argv: Optional[List[str]] = None) -> int:
    configure_logging()
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        export = _load_export_from_args(args)
    except ValueError as exc:
        parser.error(str(exc))
        return 2

    pretty = not args.compact
    user_path, posts_path, user_count, post_count = convert_and_write_export(
        export,
        output_dir=args.output_dir,
        user_out=args.user_out,
        posts_out=args.posts_out,
        pretty=pretty,
        output_format=args.format,
    )

    logger.info("Wrote %d user rows to %s", user_count, user_path)
    logger.info("Wrote %d post rows to %s", post_count, posts_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
