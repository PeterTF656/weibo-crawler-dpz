# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Weibo (微博) crawler written in Python that can scrape user information, posts (weibos), comments, reposts, images, and videos from Weibo.cn. The crawler supports multiple output formats (CSV, JSON, Markdown, MySQL, MongoDB, SQLite) and includes anti-ban mechanisms to avoid rate limiting.

## Commands

### Running the Crawler

```bash
# Basic run with config.json settings
python weibo.py

# Run with scheduled interval (in minutes)
python __main__.py <interval_in_minutes>

# Run the API service
python service.py
```

### Dependencies

```bash
# Install all dependencies
pip install -r requirements.txt
```

### Docker

```bash
# Build image
docker build -t weibo-crawler .

# Run with docker
docker run -it -d \
  -v path/to/config.json:/app/config.json \
  -v path/to/user_id_list.txt:/app/user_id_list.txt \
  -v path/to/weibo:/app/weibo \
  -e schedule_interval=1 \
  weibo-crawler

# Or use docker-compose
docker-compose up
```

## Configuration

### Primary Config File: `config.json`

The main configuration file controls all crawler behavior. Key settings:

- **user_id_list**: Can be an array of user IDs or a path to a text file containing user IDs (one per line)
- **since_date**: Start date (YYYY-MM-DD) or integer (number of days ago)
- **only_crawl_original**: 1 = only original posts, 0 = all posts (original + reposts)
- **write_mode**: Array of output formats: `["csv", "json", "markdown", "sqlite", "mysql", "mongo", "post"]`
- **cookie**: Required for full access and downloading reposts
- **anti_ban_config**: Anti-rate-limiting settings including delays, batch sizes, and random user agents

### Secondary Config File: `const.py`

Controls operational mode and cookie validation:

- **const.MODE**: `"overwrite"` (full crawl) or `"append"` (incremental crawl)
- **const.CHECK_COOKIE**: Cookie validation settings
- **const.NOTIFY**: Push notification settings (using PushDeer)

### User ID List File

When `user_id_list` points to a text file, each line format:
```
<user_id> [username] [since_date] [query_keywords]
```

Example:
```
1669879400 迪丽热巴 2020-01-18
1223178222 胡歌 10 梦想,希望
```

The program automatically updates this file with the last crawl timestamp.

## Architecture

### Main Components

**weibo.py** (~3100 lines) - Core crawler implementation:
- `Weibo` class: Main crawler logic
  - Request handling with retry logic and rate limiting
  - User info and weibo parsing from mobile Weibo API
  - Media (images/videos/Live Photo) download
  - Comments and reposts fetching
  - Multiple output format writers (CSV, JSON, Markdown, SQLite, MySQL, MongoDB, HTTP POST)
  - Anti-ban mechanisms (dynamic delays, random headers, batch processing, session management)

**service.py** - Flask REST API:
- Provides endpoints to trigger crawls, check task status, and query weibo data
- Uses ThreadPoolExecutor for async crawling
- Maintains task state in memory
- See API.md for full documentation

**__main__.py** - Scheduled execution:
- Uses `schedule` library to run crawler at intervals
- Supports continuous operation with exception handling

**const.py** - Configuration constants:
- Runtime mode (append vs overwrite)
- Cookie validation settings
- Notification settings

### Utility Modules (util/)

- **csvutil.py**: CSV file operations with UTF-8 BOM encoding
- **dateutil.py**: Date parsing utilities
- **llm_analyzer.py**: Optional LLM-based content analysis (sentiment, summary, anomaly detection)
- **notify.py**: PushDeer notification integration

### Key Data Flow

1. **Configuration Loading**: `get_config()` reads and validates `config.json`
2. **User Config Processing**: Parses user_id_list (file or array) into individual user configs
3. **Per-User Crawling**:
   - `initialize_info(user_config)`: Reset state for each user
   - `get_user_info()`: Fetch user profile
   - `get_pages()`: Determine page count based on `since_date`
   - `get_one_page(page)`: Fetch and parse weibos from a page
   - `get_one_weibo(info)`: Parse individual weibo
   - Optional: `get_weibo_comments()`, `get_weibo_reposts()`
   - Media download: `download_files()` for images/videos
4. **Data Writing**: `write_data()` calls appropriate writer(s) based on `write_mode`
5. **User List Update**: Updates user_id_list file with latest crawl timestamp

### Anti-Ban System

The crawler includes sophisticated anti-ban mechanisms:

- **Dynamic Request Delays**: Increases delay based on request count and session duration
- **Batch Processing**: Pauses after N weibos to simulate human behavior
- **Random Headers**: Rotates User-Agent, Accept-Language, and Referer
- **Session Limits**: Max weibos per session, max session time, max API errors
- **Random Rest Probability**: Occasionally pauses randomly to appear more human-like

Configure via `anti_ban_config` in config.json.

### Database Schema

**SQLite/MySQL/MongoDB** store two main entities:

1. **user table**: User profile information (id, screen_name, gender, location, followers_count, etc.)
2. **weibo table**: Post information (id, user_id, text, pics, video_url, created_at, attitudes_count, comments_count, reposts_count, etc.)

SQLite additionally has:
- **comment table**: Comments on weibos (when `download_comment` enabled)
- **repost table**: Reposts of weibos (when `download_repost` enabled)

Append mode (const.MODE="append") uses SQLite tracking to avoid re-fetching old weibos.

## Important Patterns

### Error Handling

- Network errors are retried with exponential backoff
- API errors increment `crawl_stats["api_errors"]` and may trigger session pause
- Invalid responses are logged but don't crash the crawler

### File Organization

Output files are organized as:
```
weibo/
  <username or user_id>/
    <user_id>.csv
    <user_id>.json
    <user_id>.md
    img/
      <yyyymmdd>_<weibo_id>_<seq>.jpg
    video/
      <yyyymmdd>_<weibo_id>.mp4
```

Controlled by `user_id_as_folder_name` and `output_directory` config.

### Cookie Handling

- **Without cookie**: Can fetch most public weibos (90%+) and user info
- **With cookie**: Required for:
  - Full access to all weibos
  - Downloading reposts (`download_repost`)
  - Keyword search (`query_list`)
  - Some private user profiles

Get cookie from browser DevTools (Network tab) after logging into m.weibo.cn or weibo.cn.

### Mode: Append vs Overwrite

- **Overwrite** (default): Fetches all weibos from `since_date` to now
- **Append**: Only fetches new weibos since last run (requires SQLite, skips pinned posts)

## Special Features

### LLM Analysis (Optional)

If configured, `llm_analyzer.py` can analyze weibo content using an LLM API:
- Sentiment analysis (positive/neutral/negative)
- Summary generation
- Anomaly detection (rumors, ads, sensitive content)

Configure via `llm_config` in config.json with your API endpoint and key.

### EXIF Metadata

When `write_time_in_exif: 1`, the crawler writes the weibo publish time into downloaded image EXIF data (CreateDate and DateTimeOriginal tags).

### Markdown Export

The markdown writer creates a formatted document with:
- User profile header
- Each weibo as a section with metadata table
- Embedded images (if downloaded)
- Original weibo content for reposts

## Testing

No formal test suite exists. Manual testing workflow:

1. Configure a test user ID in config.json
2. Set `since_date` to recent date (e.g., 7 days ago)
3. Set `write_mode: ["csv", "json"]` for quick validation
4. Disable downloads for faster iteration
5. Run `python weibo.py`
6. Verify output files in `weibo/<username>/`

## Debugging

- Logs are written to `log/` directory
- Set logging level in `logging.conf`
- Use `logger.info()` / `logger.error()` throughout code
- Check `crawl_stats` dict for anti-ban metrics
- If hitting rate limits, increase delays in `anti_ban_config`

## Common Issues

1. **Cookie invalid**: Use cookie checker (see const.py and README)
2. **Rate limited**: Adjust anti_ban_config delays or reduce max_weibo_per_session
3. **Missing weibos**: Some weibos require login (add cookie)
4. **Database errors**: Ensure DB is installed and config is correct
5. **Encoding issues**: All files use UTF-8, Windows may need BOM (csvutil handles this)

## API Service

`service.py` provides a REST API with endpoints:

- `POST /refresh`: Trigger a crawl for specific user IDs
- `GET /task/<task_id>`: Check task status
- `GET /weibo`: Query all weibos from SQLite DB
- `GET /weibo/<weibo_id>`: Get specific weibo details

See API.md for full documentation.
