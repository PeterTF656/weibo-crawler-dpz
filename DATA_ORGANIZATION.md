# Supabase Data Organization

## Current Structure

Files live directly under `supabase/data/`:

- `supabase_user_{user_id}.json`
- `supabase_posts_{user_id}.json`

Pros:
- Simple and flat to generate.
- Easy to list all files in one directory.

Cons:
- User data is split across two filenames instead of one folder.
- Hard to find a specific user's complete data quickly.
- No metadata or index for discovery and batch operations.
- Batch inserts require manual concatenation.

## Proposed Structure

```
supabase/data/
  index.json
  by-user/
    5643495004/
      user.json
      posts.json
      metadata.json
    6607408050/
      user.json
      posts.json
      metadata.json
  combined/
    users.json
    posts.json
```

Notes:
- `by-user/` keeps each user in a dedicated folder.
- `combined/` provides batch-friendly files.
- `index.json` summarizes all users and points to output files.
- `metadata.json` contains per-user counts and IDs.

## Benefits

- Faster discovery: jump straight to `by-user/{user_id}`.
- Cleaner batch processing with `combined/` files.
- Structured metadata for quick audits and automation.
- Compatible with existing JSON payloads (no data loss).

## Reorganize Existing Data

Dry run first:

```
python organize_supabase_data.py --dry-run
```

Generate both layouts (keeps existing files intact):

```
python organize_supabase_data.py --format both
```

Write to a separate output directory:

```
python organize_supabase_data.py --output-dir supabase/data_v2 --format both
```

## Using the New Structure

By-user access:

- User row: `supabase/data/by-user/{user_id}/user.json`
- Posts: `supabase/data/by-user/{user_id}/posts.json`
- Metadata: `supabase/data/by-user/{user_id}/metadata.json`

Batch operations:

- Users: `supabase/data/combined/users.json`
- Posts: `supabase/data/combined/posts.json`

Index and discovery:

- `supabase/data/index.json` lists user ids, counts, and file paths.

Generate new exports directly into the structure:

```
python convert_weibo_to_files.py --input weibo/123/123.json --output-dir supabase/data --format by-user
python convert_weibo_to_files.py --input weibo/123/123.json --output-dir supabase/data --format combined
```
