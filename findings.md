# Findings: Weibo to Supabase Conversion

## Supabase User Table Structure

**Source:** `target_user_row.json`

| Column | Type | Purpose | Required | Example |
|--------|------|---------|----------|---------|
| `idx` | integer | Row index | ✓ | 2 |
| `id` | UUID | Primary key | ✓ | e28a6122-631b-496d-99f4-d28670a26e35 |
| `username` | string | Display name | ✓ | 开心的小七 |
| `full_name` | string | Full legal name | ✗ | null |
| `first_name` | string | First name | ✗ | null |
| `email` | string | Email address | ✗ | null |
| `avatar_url` | string | Profile picture URL | ✗ | https://wpggtjzeecbyrmwansjg.supabase.co/... |
| `created_at` | timestamp | Account creation | ✓ | 2025-09-14 16:10:46.761894+00 |
| `updated_at` | timestamp | Last update | ✓ | 2026-01-14 07:02:07.758285+00 |
| `last_activity_at` | timestamp | Last activity | ✗ | 2026-01-14 07:02:17.324+00 |
| `invitation_code` | string | Invite code | ✗ | 000046 |
| `expo_token` | string | Push notification token | ✗ | null |
| `onboard_finished` | boolean | Onboarding status | ✗ | false |
| `hide` | boolean | Hidden profile flag | ✗ | false |
| `metadata` | JSON string | Additional data | ✗ | {"avatar_url": "...", "full_avatar_url": "..."} |

## Supabase Post Table Structure

**Source:** `target_post_row.json`

| Column | Type | Purpose | Required | Example |
|--------|------|---------|----------|---------|
| `idx` | integer | Row index | ✓ | 88 |
| `id` | UUID | Primary key | ✓ | 3da8c4ed-b9ba-4a5c-b175-83885cdec1e5 |
| `user_id` | UUID | Foreign key to users | ✓ | dcd3a197-3a1b-4a2e-960c-f71b7e7ba83c |
| `content` | string | Post text content | ✓ | 服了，太难吃了吧 |
| `type` | string | Content type | ✓ | text |
| `created_at` | timestamp | Post creation time | ✓ | 2025-09-08 15:10:51.377398+00 |
| `html_content` | string | HTML formatted content | ✗ | null |
| `summary` | string | Post summary | ✗ | null |
| `purpose_y` | ? | Purpose field Y | ✗ | null |
| `focus_domain_x` | ? | Focus domain X | ✗ | null |
| `status` | string | Post status | ✗ | null |
| `mongodb_id` | string | Legacy MongoDB ID | ✗ | null |
| `media_tasks` | JSON array | Media attachments | ✗ | [{task_id, media_id, media_url, ...}] |
| `task_ids` | UUID array | Related task IDs | ✗ | [b4366e7e-5d7d-4672-8383-e8915e272f38] |

### Media Tasks Structure
Each item in `media_tasks` array:
```json
{
  "task_id": null,
  "media_id": "UUID",
  "metadata": null,
  "media_url": "https://...",
  "media_type": "simple",
  "media_sub_type": "rich_text"
}
```

## Key Observations

### User Table
- Uses UUIDs for all IDs
- Supports metadata as JSON string for extensibility
- Has timestamp tracking (created, updated, last_activity)
- Email is optional (not all users may have email)
- `invitation_code` format: 6-digit string (000046)

### Post Table
- Links to users via `user_id` (UUID foreign key)
- Supports multiple media attachments via `media_tasks` array
- `type` field indicates content type (text, image, video, etc.)
- Has optional HTML content separate from plain text
- `task_ids` array suggests posts can be linked to tasks/workflows
- Fields like `purpose_y` and `focus_domain_x` suggest app-specific categorization

### Critical Differences from Weibo
1. **ID Format**: Supabase uses UUIDs, Weibo uses numeric string IDs
2. **Media Storage**: Supabase stores URLs, Weibo downloads to filesystem
3. **Timestamps**: Supabase uses timezone-aware timestamps; Weibo crawler emits local-time strings (see below)
4. **User Fields**: Supabase has app-specific fields (invitation_code, expo_token, onboard_finished)
5. **Post Structure**: Supabase has nested media_tasks, Weibo has flat pic/video URLs

## Weibo User Data Structure

`wb.user` is a dict (created in `Weibo.get_user_info()`), then passed through `Weibo.standardize_info()` (removes `\u200b` and strips characters that can’t be encoded to the current stdout encoding).

| Field | Type | Meaning | Example (README) |
|------|------|---------|------------------|
| `id` | `str` (sometimes `int`) | Weibo user id | `"1669879400"` |
| `screen_name` | `str` | Nickname | `"Dear-迪丽热巴"` |
| `gender` | `str` | `"f"`/`"m"` | `"f"` |
| `birthday` | `str` | Birthday / zodiac / empty | `"双子座"` |
| `location` | `str` | Location / empty | `"上海"` |
| `ip_location` | `str` | IP属地 / empty | (not shown in README example) |
| `education` | `str` | Education (combined) / empty | `"上海戏剧学院"` |
| `company` | `str` | Company / empty | `"嘉行传媒"` |
| `sunshine` | `str` | Sunshine credit / empty | `"信用极好"` |
| `registration_time` | `str` | Registration date (often `YYYY-MM-DD`) / empty | `"2010-07-02"` |
| `statuses_count` | `int` | Total statuses count | `1121` |
| `followers_count` | `int` | Followers count | `66395881` |
| `follow_count` | `int` | Following count | `250` |
| `description` | `str` | Bio / empty | `"一只喜欢默默表演的小透明。工作联系..."` |
| `profile_url` | `str` | Mobile profile URL | `"https://m.weibo.cn/u/1669879400?..."` |
| `profile_image_url` | `str` | Avatar URL | `"https://tvax2.sinaimg.cn/...jpg?...` |
| `avatar_hd` | `str` | HD avatar URL | `"https://wx2.sinaimg.cn/orj480/...jpg"` |
| `urank` | `int` | Weibo level | `44` |
| `mbrank` | `int` | Membership level | `7` |
| `verified` | `bool` | Verified or not | `true` |
| `verified_type` | `int` | Verified type (`-1` when unverified) | `0` |
| `verified_reason` | `str` | Verification reason / empty | `"嘉行传媒签约演员　"` |

## Weibo Post Data Structure

`wb.weibo` is a list of dicts (each item returned by `Weibo.get_one_weibo()`).

### Base Weibo Item (original or retweet wrapper)
| Field | Type | Meaning | Example (README) |
|------|------|---------|------------------|
| `user_id` | `int` (sometimes `str`) | Author user id | `1669879400` |
| `screen_name` | `str` | Author nickname | `"Dear-迪丽热巴"` |
| `id` | `int` | Weibo numeric id | `4454572602912349` |
| `bid` | `str` | Weibo bid | `"ImTGkcdDn"` |
| `text` | `str` | Content (HTML or plain text depending on `remove_html_tag`) | `"今天的#星光大赏#"` |
| `article_url` | `str` | Headline article URL or `""` | (not shown in README example) |
| `pics` | `str` | Comma-separated image URLs or `""` | `"https://wx3.sinaimg.cn/large/...jpg,https://wx4...jpg"` |
| `video_url` | `str` | Video URL or `""` | `""` / `"http://f.video.weibocdn.com/...mp4?...` |
| `live_photo_url` | `str` | Semicolon-separated Live Photo video URLs or `""` | (not shown in README example) |
| `location` | `str` | Location string or `""` | `""` |
| `created_at` | `str` | Standardized time: `%Y-%m-%dT%H:%M:%S` | README shows date-only: `"2019-12-28"` |
| `full_created_at` | `str` | Standardized time: `%Y-%m-%d %H:%M:%S` | (not shown in README example) |
| `source` | `str` | Posting client/source or `""` | `""` / `"微博 weibo.com"` |
| `attitudes_count` | `int` | Likes | `551894` |
| `comments_count` | `int` | Comments | `182010` |
| `reposts_count` | `int` | Reposts | `1000000` |
| `topics` | `str` | Comma-separated topic names (no `#`) or `""` | `"星光大赏"` |
| `at_users` | `str` | Comma-separated @user names or `""` | `""` / `"Dear-迪丽热巴,法国娇韵诗"` |
| `edited` | `bool` | Whether edited (`edit_count > 0`) | (not shown in README example) |
| `edit_count` | `int` | Edit count | (not shown in README example) |
| `llm_analysis` | `dict` (optional) | Added only when `llm_config` is enabled | (not shown in README example) |

### Retweet / Original Post Structure
- If the item is a retweet, it contains `retweet: dict` representing the retweeted/original post.
- `retweet` dict is built by `parse_weibo(retweeted_status)` and then gets `created_at`/`full_created_at` standardized; it generally has the same fields as the base item (but does not add another nested `retweet` level, and does not set `edited`/`edit_count`).

### Media Storage (URLs vs Filesystem)
- In-memory/output structures store media as URLs (`pics`, `video_url`, `live_photo_url`), not local file paths.
- When download options are enabled, files are saved under `weibo/<user_dir>/img`, `weibo/<user_dir>/video`, `weibo/<user_dir>/live_photo` (naming is based on `created_at` timestamps, e.g. `YYYY-MM-DD_HH-MM-SS.jpg` for images).

## Phase 3: Mapping Strategy (Weibo → Supabase)

### 1) User Mapping Table

#### Users Table Columns (`users`)

| Supabase column | Weibo source | Transform / rule | If missing |
|---|---|---|---|
| `idx` | (generated) | 1-based row counter in the export batch | Must generate |
| `id` | `wb.user.id` | Deterministic UUID (UUIDv5) derived from Weibo user id; also store original in `metadata.weibo_user_id` | Skip row (no stable key) |
| `username` | `wb.user.screen_name` | Use as-is (strip/standardize); fallback `weibo_<weibo_user_id>` | Fallback to `weibo_<weibo_user_id>` |
| `full_name` | (none) | `null` | `null` |
| `first_name` | (none) | `null` | `null` |
| `email` | (none) | `null` | `null` |
| `avatar_url` | `wb.user.avatar_hd` (preferred), else `wb.user.profile_image_url` | If uploading to Supabase Storage: store public URL; else store Weibo URL directly | `null` |
| `created_at` | `wb.user.registration_time` (optional) | If parseable `YYYY-MM-DD`: interpret as Asia/Shanghai `00:00:00`, convert to UTC ISO 8601; else use `imported_at` (UTC now) | Use `imported_at` |
| `updated_at` | (generated) | `imported_at` (UTC ISO 8601) | Must generate |
| `last_activity_at` | (computed from posts) | Max `posts.created_at` for this user (UTC ISO 8601) | `null` (or set to `updated_at`) |
| `invitation_code` | (generated, optional) | If needed: `f"{idx:06d}"` to match sample format | `null` |
| `expo_token` | (none) | `null` | `null` |
| `onboard_finished` | (none) | `false` | `false` |
| `hide` | (none) | `false` | `false` |
| `metadata` | (generated) | JSON **string** (see below) | `"{}"` |

#### Users Metadata (`users.metadata` JSON string)

Store `metadata` as a JSON-serialized string (example shows it is not a JSON object). Recommended keys:

| Metadata key | Weibo source | Transform / rule | If missing |
|---|---|---|---|
| `source` | (constant) | `"weibo-crawler-dpz"` | Must set |
| `imported_at` | (generated) | UTC ISO 8601 timestamp | Must set |
| `weibo_user_id` | `wb.user.id` | Always store as string | `""` |
| `profile_url` | `wb.user.profile_url` | As-is | `""` |
| `gender` | `wb.user.gender` | As-is (`"m"`, `"f"`, or `""`) | `""` |
| `birthday` | `wb.user.birthday` | As-is | `""` |
| `location` | `wb.user.location` | As-is | `""` |
| `ip_location` | `wb.user.ip_location` | As-is | `""` |
| `education` | `wb.user.education` | As-is | `""` |
| `company` | `wb.user.company` | As-is | `""` |
| `sunshine` | `wb.user.sunshine` | As-is | `""` |
| `registration_time` | `wb.user.registration_time` | As-is (raw), even if `created_at` is derived from it | `""` |
| `statuses_count` | `wb.user.statuses_count` | Integer | `0` |
| `followers_count` | `wb.user.followers_count` | Integer | `0` |
| `follow_count` | `wb.user.follow_count` | Integer | `0` |
| `description` | `wb.user.description` | As-is | `""` |
| `verified` | `wb.user.verified` | Boolean | `false` |
| `verified_type` | `wb.user.verified_type` | Integer | `-1` |
| `verified_reason` | `wb.user.verified_reason` | As-is | `""` |
| `urank` | `wb.user.urank` | Integer | `0` |
| `mbrank` | `wb.user.mbrank` | Integer | `0` |
| `profile_image_url` | `wb.user.profile_image_url` | As-is (original Weibo URL) | `""` |
| `avatar_hd` | `wb.user.avatar_hd` | As-is (original Weibo URL) | `""` |
| `avatar_url` | (derived) | Copy of `users.avatar_url` for convenience | `null` |
| `full_avatar_url` | (derived) | Prefer “HD” uploaded URL; else copy of `users.avatar_url` | `null` |
| `weibo_raw` | `wb.user` | Optional: store full standardized user dict to avoid losing new/unknown fields | Omit |

### 2) Post Mapping Table

#### Posts Table Columns (`posts`)

| Supabase column | Weibo source | Transform / rule | If missing |
|---|---|---|---|
| `idx` | (generated) | 1-based row counter in the export batch | Must generate |
| `id` | `wb.weibo[i].id` | Deterministic UUID (UUIDv5) derived from Weibo post id; keep original id separately (see `mongodb_id`) | Skip row (no stable key) |
| `user_id` | `wb.weibo[i].user_id` | Map Weibo user id → Supabase user UUID using the same UUIDv5 rule as `users.id` | Skip row (or create placeholder user) |
| `content` | `wb.weibo[i].text` (+ optional `retweet`) | Store **plain text**. If the crawler kept HTML, strip tags for `content`. If `retweet` exists, flatten into `content` (see Retweet section). | `""` (or `"(no content)"` if non-empty required) |
| `type` | (derived) | Use `"text"` for all imported Weibo posts (media is represented via `media_tasks`) | `"text"` |
| `created_at` | `wb.weibo[i].created_at` / `full_created_at` | Parse as Asia/Shanghai local time, convert to UTC ISO 8601 | Use `imported_at` |
| `html_content` | `wb.weibo[i].text` (+ optional `retweet`) | If HTML is available, store HTML (optionally with a rendered retweet blockquote); else `null` | `null` |
| `summary` | `wb.weibo[i].llm_analysis.summary` (optional) | Copy summary if present | `null` |
| `purpose_y` | (none) | `null` | `null` |
| `focus_domain_x` | (none) | `null` | `null` |
| `status` | (none) | `null` | `null` |
| `mongodb_id` | `wb.weibo[i].id` | Store original Weibo numeric post id as string (acts as `source_id`) | `null` |
| `media_tasks` | `pics` / `video_url` / `live_photo_url` (+ optional `retweet`) | Build `media_tasks` array (see below) | `[]` |
| `task_ids` | (none) | `null` (or `[]` if your DB prefers empty arrays) | `null` |

#### Weibo Post Fields Coverage (field-by-field)

This table makes explicit where each Weibo field ends up given the current `posts` schema.

| Weibo field | Supabase destination | Notes |
|---|---|---|
| `user_id` | `posts.user_id` | Convert Weibo user id → Supabase user UUID (UUIDv5). |
| `screen_name` | (not stored) | Derivable from `posts.user_id → users.username`. |
| `id` | `posts.id` + `posts.mongodb_id` | `posts.id` is UUIDv5; `posts.mongodb_id` stores original Weibo numeric id as string. |
| `bid` | (not stored) | Recommended future: `posts.metadata.bid`. |
| `text` | `posts.content` (+ `posts.html_content`) | `content` is plain text; `html_content` stores raw HTML if available. |
| `article_url` | (not stored) | Recommended future: `posts.metadata.article_url`. |
| `pics` | `posts.media_tasks[]` | One `media_tasks` entry per URL (`metadata.kind="image"`). |
| `video_url` | `posts.media_tasks[]` | One `media_tasks` entry if non-empty (`metadata.kind="video"`). |
| `live_photo_url` | `posts.media_tasks[]` | One entry per URL (`metadata.kind="live_photo"`). |
| `location` | (not stored) | Recommended future: `posts.metadata.location`. |
| `created_at` | `posts.created_at` | Convert Asia/Shanghai → UTC ISO 8601. |
| `full_created_at` | (used during parsing) | Not stored; recommended future: `posts.metadata.full_created_at`. |
| `source` | (not stored) | Recommended future: `posts.metadata.source`. |
| `attitudes_count` | (not stored) | Recommended future: `posts.metadata.attitudes_count`. |
| `comments_count` | (not stored) | Recommended future: `posts.metadata.comments_count`. |
| `reposts_count` | (not stored) | Recommended future: `posts.metadata.reposts_count`. |
| `topics` | (not stored) | Often already present in `text` (e.g., `#话题#`). Recommended future: `posts.metadata.topics`. |
| `at_users` | (not stored) | Often already present in `text` (e.g., `@用户`). Recommended future: `posts.metadata.at_users`. |
| `edited` | (not stored) | Recommended future: `posts.metadata.edited`. |
| `edit_count` | (not stored) | Recommended future: `posts.metadata.edit_count`. |
| `llm_analysis` | `posts.summary` (partial) | Store `llm_analysis.summary` in `posts.summary`; keep full dict in future `posts.metadata.llm_analysis`. |
| `retweet` | `posts.content` / `posts.html_content` + `posts.media_tasks[]` | Flatten retweet content for display; include retweet media with `metadata.from="retweet"`. |

#### Media Tasks (`posts.media_tasks`)

Create one `media_tasks[]` entry per media URL. Use the sample shape and keep Weibo-specific details in each item’s `metadata`.

**Base shape (per item):**
```json
{
  "task_id": null,
  "media_id": "UUID",
  "metadata": { "source": "weibo", "...": "..." },
  "media_url": "https://...",
  "media_type": "simple",
  "media_sub_type": "rich_text"
}
```

**`media_id` rule:**
- Prefer deterministic UUIDv5 derived from `metadata.storage_path` (if uploaded) else `metadata.original_url` (stable across re-imports).
- If stability is not required, UUIDv4 per item is acceptable.

**URL parsing rules:**
- `pics`: split by `,` → image items (preserve order).
- `video_url`: single URL → video item (if non-empty).
- `live_photo_url`: split by `;` → live_photo items (preserve order).

**Per-item `metadata` recommended keys:**
- `kind`: `"image" | "video" | "live_photo"`
- `original_url`: original Weibo CDN URL
- `weibo_post_id`: numeric post id as string
- `weibo_user_id`: numeric user id as string
- `index`: 0-based position within its kind
- `from`: `"post" | "retweet"` (if the media came from the retweeted/original status)
- `local_path`: local filesystem path (only if downloaded)
- `storage_path`: Supabase Storage object path (only if uploaded)

**Local file path → URL strategy:**
- If `local_path` exists (downloaded media), upload to Supabase Storage and set `media_url` to the resulting public URL. Store the object key as `metadata.storage_path`.
- If no local file exists, set `media_url = original_url` and keep `metadata.original_url`.

#### Retweet Handling

Decision: **flatten retweets for storage/display**, and optionally keep structured data elsewhere if the schema is extended later.

- `content` flattening:
  - If `weibo.retweet` exists, set `content` to:
    - `{weibo.text}\n\n// @{retweet.screen_name}: {retweet.text}`
  - If the wrapper `weibo.text` is empty, use only the retweet part.
- `html_content` flattening (when you have HTML available):
  - Render a `<blockquote>` for the retweeted/original post.
- Include retweet media in the same `media_tasks` array with `metadata.from = "retweet"`.

### 3) Critical Decisions

| Topic | Decision |
|---|---|
| ID strategy | Use deterministic UUIDv5 for `users.id` and `posts.id` derived from Weibo ids; also store original Weibo ids (`users.metadata.weibo_user_id`, `posts.mongodb_id`) for traceability. |
| Media strategy | Prefer uploading downloaded media to Supabase Storage and storing the resulting public `media_url`; if no local file exists, fall back to the original Weibo URL and keep `original_url` in `media_tasks[].metadata`. |
| Timestamp | Treat crawler timestamps as Asia/Shanghai local time, convert to UTC, store timezone-aware ISO 8601 strings in Supabase (`...Z` / `+00:00`). |
| Missing required fields | Generate `idx`; default `type="text"`; default `content=""` if absent; default timestamps to `imported_at` when parsing fails. |
| Retweets | Flatten retweets into `content` / `html_content`; include retweet media in `media_tasks` with `metadata.from="retweet"`. |
| `type` values | Keep `posts.type="text"` for all imported Weibo posts; treat media/retweet richness as `media_tasks` + `html_content`. |
| Media IDs | Use deterministic UUIDv5 for `media_tasks[].media_id` based on storage path / original URL to avoid duplicates across repeated imports. |
