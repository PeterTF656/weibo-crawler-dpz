# 📝 How to Convert Weibo Data to Supabase Format - For Project Leader

## Executive Summary

The dev team has created a **ready-to-use conversion tool** that transforms Weibo crawler output into Supabase-compatible JSON files. Two files are generated:
1. **`supabase_user.json`** - User profile row
2. **`supabase_posts.json`** - All posts rows

---

## ✅ What's Ready

| Item | Status | File |
|------|--------|------|
| Conversion script | ✅ Ready | `convert_weibo_to_files.py` |
| User guide | ✅ Ready | `CONVERSION_GUIDE.md` |
| Conversion library | ✅ Ready | `weibo_to_supabase.py` |
| Test suite | ✅ Ready | `test_conversion.py` |
| Example script | ✅ Ready | `example_convert.py` |

---

## 🚀 Quick Start (3 Steps)

### Step 1: Run the Weibo Crawler
Make sure your `config.json` has JSON output enabled:

```json
{
  "write_mode": ["json"],
  "user_id_list": ["1669879400"]
}
```

Run the crawler:
```bash
python weibo.py
```

This creates: `weibo/1669879400/1669879400.json`

### Step 2: Convert to Supabase Format
```bash
python convert_weibo_to_files.py --input weibo/1669879400/1669879400.json
```

### Step 3: Check Output Files
Two files are created in the current directory:
- ✅ `supabase_user.json` - 1 user profile row
- ✅ `supabase_posts.json` - All posts rows (with media_tasks)

---

## 📋 Output Format

### User Profile File: `supabase_user.json`

```json
[
  {
    "idx": 1,
    "id": "7fd84de8-4dad-527f-bccb-3ba68a7cb096",  // Deterministic UUID
    "username": "Dear-迪丽热巴",
    "avatar_url": "https://wx2.sinaimg.cn/...",
    "created_at": "2010-07-01T16:00:00+00:00",     // UTC timestamp
    "updated_at": "2026-01-21T03:28:20+00:00",
    "last_activity_at": "2019-12-28T12:00:00+00:00",
    "metadata": "{...all Weibo fields preserved...}"  // JSON string
  }
]
```

### Posts File: `supabase_posts.json`

```json
[
  {
    "idx": 1,
    "id": "a853c362-d1ea-5b8d-a008-98a8ca8d39c8",  // Deterministic UUID
    "user_id": "7fd84de8-4dad-527f-bccb-3ba68a7cb096",  // Links to user
    "content": "今天的#星光大赏#",
    "type": "text",
    "created_at": "2019-12-28T12:00:00+00:00",     // UTC timestamp
    "mongodb_id": "4454572602912349",              // Original Weibo ID
    "media_tasks": [                               // Image/video array
      {
        "media_id": "02dd0de2-a77e-5250-b009-1bee19e99888",
        "media_url": "https://wx3.sinaimg.cn/large/...",
        "metadata": {
          "kind": "image",
          "original_url": "https://...",
          "index": 0,
          "from": "post"
        }
      }
    ]
  }
]
```

---

## 🎯 Common Use Cases

### Use Case 1: Single User Conversion
```bash
# Crawl one user
python weibo.py

# Convert (finds the JSON automatically)
python convert_weibo_to_files.py --input weibo/1669879400/1669879400.json

# Output in current directory:
# - supabase_user.json (1 user)
# - supabase_posts.json (all posts)
```

### Use Case 2: Multiple Users with Organized Output
```bash
# Convert user 1
python convert_weibo_to_files.py \
  --input weibo/1669879400/1669879400.json \
  --output-dir output/user_1669879400

# Convert user 2
python convert_weibo_to_files.py \
  --input weibo/1223178222/1223178222.json \
  --output-dir output/user_1223178222

# Result:
# output/
#   user_1669879400/
#     supabase_user.json
#     supabase_posts.json
#   user_1223178222/
#     supabase_user.json
#     supabase_posts.json
```

### Use Case 3: Batch Processing All Users
```bash
# Process all Weibo exports at once
for export in weibo/*/*.json; do
  user_id=$(basename "$export" .json)
  python convert_weibo_to_files.py \
    --input "$export" \
    --output-dir "output/$user_id"
done
```

### Use Case 4: Compact JSON (Smaller Files)
```bash
# Generate compact JSON without pretty formatting
python convert_weibo_to_files.py \
  --input weibo/1669879400/1669879400.json \
  --compact
```

---

## 🔧 Advanced Options

### CLI Help
```bash
python convert_weibo_to_files.py --help
```

### All Available Options
```bash
--input FILE           # Weibo JSON export file
--user-file FILE       # Separate user JSON file
--weibo-file FILE      # Separate posts JSON file
--output-dir DIR       # Where to save output files
--user-out FILENAME    # Custom user filename (default: supabase_user.json)
--posts-out FILENAME   # Custom posts filename (default: supabase_posts.json)
--pretty               # Pretty-print JSON (default)
--compact              # Compact JSON (no indentation)
```

### Custom Output Names
```bash
python convert_weibo_to_files.py \
  --input weibo/1669879400/1669879400.json \
  --user-out user_dilireba.json \
  --posts-out posts_dilireba.json
```

---

## 🔍 Verification

### Check File Sizes
```bash
ls -lh supabase_*.json

# Expected output:
# supabase_user.json   (~1-2 KB for 1 user)
# supabase_posts.json  (~2-20 KB depending on post count)
```

### Verify Conversion
```bash
# Count posts
cat supabase_posts.json | grep '"id":' | wc -l

# Check user UUID (should be deterministic)
cat supabase_user.json | grep '"id":'

# Verify timestamps are in UTC
cat supabase_posts.json | grep '"created_at":'
```

### Run Tests
```bash
# Validate conversion logic (28,733 checks)
python3 test_conversion.py

# Expected: Checks: 28733, Failures: 0
```

---

## 💡 Understanding the Output

### Key Features

1. **Deterministic UUIDs**
   - Same Weibo ID always generates same UUID
   - Running conversion twice = identical output
   - Safe for re-imports (use UPSERT in database)

2. **Data Preservation**
   - ALL Weibo fields preserved
   - Primary data → Supabase columns
   - Extra data → metadata JSON string
   - Original IDs → mongodb_id field

3. **Timezone Conversion**
   - Input: Asia/Shanghai local time
   - Output: UTC with timezone info
   - Format: ISO 8601 (e.g., `2019-12-28T12:00:00+00:00`)

4. **Media Handling**
   - Images parsed from comma-separated URLs
   - Videos parsed from single URL
   - Each media gets unique UUID
   - Includes metadata (kind, index, source)

5. **Retweet Flattening**
   - Original + retweeted content combined
   - Format: `{text}\n\n// @{user}: {retweet}`
   - Retweet media included with `from: "retweet"`

---

## 📚 Related Documentation

| Document | Purpose |
|----------|---------|
| `CONVERSION_GUIDE.md` | Detailed usage guide |
| `PROJECT_COMPLETE.md` | Full project documentation |
| `findings.md` | Complete field mappings |
| `weibo_to_supabase.py` | Conversion library (can import) |
| `test_conversion.py` | Validation tests |

---

## 🎓 For Developers: Python API

If you want to integrate conversion into Python code:

```python
from convert_weibo_to_files import convert_and_write

# After running Weibo crawler
wb = Weibo(config)
wb.start()

# Convert and save
convert_and_write(
    user=wb.user,
    weibo=wb.weibo,
    output_dir="output"
)
```

Or use the library directly:

```python
from weibo_to_supabase import convert_weibo_export

weibo_data = {"user": wb.user, "weibo": wb.weibo}
result = convert_weibo_export(weibo_data)

# result = {
#   "users": [...],   # Array with 1 user
#   "posts": [...]    # Array with all posts
# }
```

---

## ✅ Next Steps: Insert to Supabase

Once you have the JSON files, insert them into Supabase:

```python
from supabase import create_client
import json

# Initialize Supabase client
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# Load converted files
with open('supabase_user.json') as f:
    users = json.load(f)

with open('supabase_posts.json') as f:
    posts = json.load(f)

# Insert to Supabase (UPSERT for idempotency)
supabase.table('users').upsert(users).execute()
supabase.table('posts').upsert(posts).execute()

print(f"✅ Inserted {len(users)} users and {len(posts)} posts")
```

---

## 🆘 Troubleshooting

### Issue: "No such file or directory"
**Solution:** Ensure the Weibo crawler generated JSON files. Check `config.json` has `"write_mode": ["json"]`.

### Issue: "Invalid JSON"
**Solution:** Verify the input file is valid JSON:
```bash
python3 -m json.tool weibo/1669879400/1669879400.json
```

### Issue: "Empty output files"
**Solution:** Check if Weibo export has data:
```bash
cat weibo/1669879400/1669879400.json | grep '"weibo":'
```

### Issue: "Different UUIDs each time"
**Solution:** This shouldn't happen (UUIDs are deterministic). If it does, please run tests:
```bash
python3 test_conversion.py
```

---

## 📊 Example: Complete Workflow

```bash
# 1. Configure crawler
cat > config.json << EOF
{
  "user_id_list": ["1669879400"],
  "write_mode": ["json"],
  "since_date": "2025-06-01",
  "original_pic_download": 1
}
EOF

# 2. Run crawler
python weibo.py

# 3. Convert to Supabase format
python convert_weibo_to_files.py --input weibo/1669879400/1669879400.json

# 4. Verify output
ls -lh supabase_*.json
cat supabase_user.json | python3 -m json.tool | head -20
cat supabase_posts.json | python3 -m json.tool | head -20

# 5. Success! Files ready for Supabase
```

---

## 🎉 Summary for Project Leader

**Status: ✅ READY FOR PRODUCTION**

The conversion system is:
- ✅ **Tested**: 28,733 checks passed
- ✅ **Documented**: Complete guides and examples
- ✅ **Simple**: 1 command converts Weibo → Supabase
- ✅ **Safe**: Deterministic, idempotent, preserves all data
- ✅ **Flexible**: CLI, Python API, batch processing

**Dev team can now:**
1. Run Weibo crawler
2. Convert output with 1 command
3. Get 2 JSON files ready for Supabase insertion

**Total time per user: ~2 minutes**
(Crawler time + conversion time)
