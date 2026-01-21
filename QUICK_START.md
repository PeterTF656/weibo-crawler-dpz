# ⚡ Quick Start Guide

## 1-Minute Overview

You have a complete Weibo → Supabase conversion system ready to use.

---

## ✅ Your Data (Already Organized)

**Location:** `supabase/data/`

- **3 users** organized in `by-user/` folders
- **407 posts** ready in `combined/posts.json`
- **Index file** at `index.json` for overview
- **Original files** archived in `archive/`

---

## 🚀 Most Common Tasks

### Convert New Weibo User
```bash
python convert_weibo_to_files.py \
  --input weibo/USER_ID/USER_ID.json \
  --output-dir supabase/data \
  --format both
```

### Insert All Data to Supabase
```python
import json
from supabase import create_client

supabase = create_client(URL, KEY)

users = json.load(open('supabase/data/combined/users.json'))
posts = json.load(open('supabase/data/combined/posts.json'))

supabase.table('users').upsert(users).execute()
supabase.table('posts').upsert(posts).execute()
```

### Check Data Overview
```bash
cat supabase/data/index.json | python3 -m json.tool
```

---

## 📚 Need More Help?

| Question | Read |
|----------|------|
| How do I convert data? | `HOW_TO_CONVERT.md` |
| What's the data structure? | `DATA_IMPROVEMENTS.md` |
| Full project details? | `FINAL_SUMMARY.md` |

---

## ⚡ That's It!

Everything is ready. Start converting and inserting your Weibo data!
