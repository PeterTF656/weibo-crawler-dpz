# Weibo to Supabase Conversion Guide

This guide shows how to convert Weibo crawler JSON into Supabase-ready JSON files.
It includes CLI usage, input/output formats, and common workflows.

## Quick Start

1. Run the crawler with JSON output enabled (write_mode includes `json`).
2. Locate the export file (example: `weibo/<user_id>/<user_id>.json`).
3. Convert it:

```bash
python convert_weibo_to_files.py --input weibo/1669879400/1669879400.json
```

Outputs (in the current directory by default):
- `supabase_user.json`
- `supabase_posts.json`

## Script Usage

Basic usage:

```bash
python convert_weibo_to_files.py --help
```

Convert a full export file:

```bash
python convert_weibo_to_files.py --input weibo/1669879400/1669879400.json
```

Convert from separate `wb.user` and `wb.weibo` files:

```bash
python convert_weibo_to_files.py --user-file user.json --weibo-file weibo.json
```

Write to a custom output directory and keep pretty JSON:

```bash
python convert_weibo_to_files.py --input weibo/1669879400/1669879400.json --output-dir out --pretty
```

Write compact JSON:

```bash
python convert_weibo_to_files.py --input weibo/1669879400/1669879400.json --compact
```

## Direct Conversion (Python)

If you already have `wb.user` and `wb.weibo` in memory:

```python
from convert_weibo_to_files import convert_and_write

convert_and_write(wb.user, wb.weibo, output_dir="out")
```

## Expected Input Format

### Full export file (recommended)

`weibo/<user_id>/<user_id>.json` from the crawler:

```json
{
  "user": {
    "id": "1669879400",
    "screen_name": "example_user",
    "registration_time": "2020-01-01"
  },
  "weibo": [
    {
      "id": "4454572602912349",
      "user_id": "1669879400",
      "text": "Hello world",
      "created_at": "2020-01-02 12:30:00",
      "pics": "https://example.com/a.jpg"
    }
  ]
}
```

### Separate files

`user.json` must be a JSON object (wb.user).  
`weibo.json` must be a JSON array (wb.weibo).

## Expected Output Format

The script writes two JSON arrays:

`supabase_user.json`:

```json
[
  {
    "idx": 1,
    "id": "uuid-string",
    "username": "example_user",
    "avatar_url": "https://example.com/avatar.jpg",
    "created_at": "2020-01-01T00:00:00+00:00",
    "updated_at": "2025-01-01T00:00:00+00:00",
    "metadata": "{...}"
  }
]
```

`supabase_posts.json`:

```json
[
  {
    "idx": 1,
    "id": "uuid-string",
    "user_id": "uuid-string",
    "content": "Hello world",
    "created_at": "2020-01-02T04:30:00+00:00",
    "mongodb_id": "4454572602912349",
    "media_tasks": []
  }
]
```

Notes:
- IDs are deterministic UUIDv5 values.
- Timestamps are converted to UTC ISO 8601.
- Weibo-specific fields are stored in `metadata`.

## Common Use Cases

Single user:

```bash
python convert_weibo_to_files.py --input weibo/1669879400/1669879400.json
```

Multiple users (one command per export file):

```bash
python convert_weibo_to_files.py --input weibo/123/123.json --output-dir out/123
python convert_weibo_to_files.py --input weibo/456/456.json --output-dir out/456
```

Batch processing (loop over exports):

```bash
for export in weibo/*/*.json; do
  user_id=$(basename "$export" .json)
  python convert_weibo_to_files.py --input "$export" --output-dir "out/$user_id"
done
```

If your export files use a different pattern, adjust the loop accordingly.
