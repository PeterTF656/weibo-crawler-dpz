# Supabase with Cloudflare R2 Storage Integration

**Source**: Official Supabase documentation, GitHub discussions, and community implementations
**Retrieved**: December 20, 2025
**Last Updated**: December 2025

## Summary

Supabase does not natively support Cloudflare R2 as a direct storage backend replacement for its hosted Storage service. However, there are multiple approaches to integrate R2 with Supabase, including using S3-compatible wrappers for database queries, self-hosted configurations, and hybrid architectures. Cloudflare R2 offers significant cost savings (zero egress fees, 10GB free storage) compared to traditional S3 storage.

## Key Concepts

### Native Support Status

- **Hosted Supabase**: Does NOT support replacing the built-in Storage service with Cloudflare R2
- **Self-Hosted Supabase**: CAN be configured to use R2 as the storage backend (requires custom configuration)
- **Database Wrappers**: Supabase provides Foreign Data Wrappers (FDW) that can query R2 via S3-compatible API

### Why Use Cloudflare R2?

1. **Zero Egress Fees**: No bandwidth charges for downloads (major cost saver)
2. **Generous Free Tier**: 10GB free storage vs Supabase's 1GB free tier
3. **S3 Compatibility**: Fully compatible with S3 API
4. **Cost Savings**: Real-world example: $354/year → $0.48/year for storage costs

## Integration Approaches

### 1. Database-Level Integration (PostgreSQL Wrappers)

Use Supabase's Foreign Data Wrappers to query R2 data directly from PostgreSQL.

#### AWS S3 Wrapper Configuration

```sql
-- Enable the Wrappers extension
create extension if not exists wrappers;

-- Create S3 foreign data wrapper server
create foreign data wrapper s3_wrapper
  handler s3_fdw_handler
  validator s3_fdw_validator;

-- Create server pointing to R2
create server r2_server
  foreign data wrapper s3_wrapper
  options (
    endpoint_url 'https://<ACCOUNT_ID>.r2.cloudflarestorage.com',
    region 'auto',  -- R2 uses 'auto' as region
    path_style_url 'false'
  );

-- Create user mapping with R2 credentials
create user mapping for current_user
  server r2_server
  options (
    access_key_id '<R2_ACCESS_KEY_ID>',
    secret_access_key '<R2_SECRET_ACCESS_KEY>'
  );

-- Create foreign table to query R2 bucket
create foreign table r2_files (
  name text,
  size bigint,
  last_modified timestamp
)
  server r2_server
  options (
    bucket 'my-bucket',
    object_key 'prefix/*'
  );
```

**Key Parameters**:
- `endpoint_url`: Points to your R2 account endpoint
- `region`: Use `'auto'` for R2 (required by Cloudflare)
- `path_style_url`: Whether to use path-style URLs (usually `false`)

#### DuckDB Wrapper for Analytics

For analytical queries on R2 data:

```sql
-- Enable DuckDB wrapper
create extension if not exists wrappers;

-- Create DuckDB foreign data wrapper
create foreign data wrapper duckdb_wrapper
  handler duckdb_fdw_handler
  validator duckdb_fdw_validator;

-- Create server with R2 S3 configuration
create server duckdb_server
  foreign data wrapper duckdb_wrapper
  options (
    s3_endpoint '<ACCOUNT_ID>.r2.cloudflarestorage.com',
    s3_access_key_id '<R2_ACCESS_KEY_ID>',
    s3_secret_access_key '<R2_SECRET_ACCESS_KEY>',
    s3_region 'auto'
  );

-- Query Parquet files directly from R2
create foreign table r2_analytics (
  id bigint,
  data jsonb
)
  server duckdb_server
  options (
    file 's3://my-bucket/analytics/*.parquet'
  );
```

### 2. Self-Hosted Supabase Configuration

For self-hosted Supabase instances, you can configure R2 as the storage backend.

#### Environment Variables

Add these to your `.env` or `docker-compose.yml`:

```bash
# Storage Backend Configuration
STORAGE_BACKEND=s3
STORAGE_S3_ENDPOINT=https://<ACCOUNT_ID>.r2.cloudflarestorage.com
STORAGE_S3_REGION=auto
STORAGE_S3_FORCE_PATH_STYLE=false

# R2 Credentials
AWS_ACCESS_KEY_ID=<R2_ACCESS_KEY_ID>
AWS_SECRET_ACCESS_KEY=<R2_SECRET_ACCESS_KEY>

# Bucket Configuration
STORAGE_S3_BUCKET=<your-bucket-name>
```

#### Docker Compose Example

```yaml
services:
  storage:
    image: supabase/storage-api:latest
    environment:
      STORAGE_BACKEND: s3
      STORAGE_S3_ENDPOINT: https://${R2_ACCOUNT_ID}.r2.cloudflarestorage.com
      STORAGE_S3_REGION: auto
      STORAGE_S3_FORCE_PATH_STYLE: false
      AWS_ACCESS_KEY_ID: ${R2_ACCESS_KEY_ID}
      AWS_SECRET_ACCESS_KEY: ${R2_SECRET_ACCESS_KEY}
      STORAGE_S3_BUCKET: ${R2_BUCKET_NAME}
```

**Important Notes**:
- Files are uploaded to your server first, then forwarded to R2
- URLs returned by Supabase point to your server domain, not R2 directly
- This may consume bandwidth on your server for file delivery

### 3. Hybrid Architecture (Recommended Pattern)

Use both Supabase and R2 for their strengths:

```typescript
// Store file metadata in Supabase PostgreSQL
const { data: metadata, error } = await supabase
  .from('files')
  .insert({
    user_id: userId,
    file_name: 'image.jpg',
    file_size: 1024000,
    r2_url: 'https://pub-xxx.r2.dev/images/image.jpg',
    created_at: new Date()
  });

// Upload file directly to R2 using S3 SDK
import { S3Client, PutObjectCommand } from '@aws-sdk/client-s3';

const s3Client = new S3Client({
  region: 'auto',
  endpoint: `https://${accountId}.r2.cloudflarestorage.com`,
  credentials: {
    accessKeyId: R2_ACCESS_KEY_ID,
    secretAccessKey: R2_SECRET_ACCESS_KEY,
  },
});

await s3Client.send(
  new PutObjectCommand({
    Bucket: 'my-bucket',
    Key: 'images/image.jpg',
    Body: fileBuffer,
  })
);
```

**Benefits**:
- Supabase handles: Auth, RLS policies, metadata queries, relationships
- R2 handles: Actual file storage, CDN delivery, zero egress costs
- Best of both worlds: Security + Cost Efficiency

### 4. Application-Level Integration

Handle R2 uploads in your application code while using Supabase for everything else.

```typescript
// Example: Next.js API Route
export async function POST(request: Request) {
  // 1. Authenticate with Supabase
  const supabase = createClient();
  const { data: { user } } = await supabase.auth.getUser();

  if (!user) {
    return new Response('Unauthorized', { status: 401 });
  }

  // 2. Upload to R2
  const formData = await request.formData();
  const file = formData.get('file') as File;

  const s3Client = new S3Client({
    region: 'auto',
    endpoint: process.env.R2_ENDPOINT,
    credentials: {
      accessKeyId: process.env.R2_ACCESS_KEY_ID!,
      secretAccessKey: process.env.R2_SECRET_ACCESS_KEY!,
    },
  });

  const key = `uploads/${user.id}/${file.name}`;

  await s3Client.send(
    new PutObjectCommand({
      Bucket: process.env.R2_BUCKET_NAME,
      Key: key,
      Body: Buffer.from(await file.arrayBuffer()),
      ContentType: file.type,
    })
  );

  // 3. Store metadata in Supabase
  const publicUrl = `https://pub-xxx.r2.dev/${key}`;

  await supabase.from('files').insert({
    user_id: user.id,
    file_name: file.name,
    file_url: publicUrl,
    file_type: file.type,
    file_size: file.size,
  });

  return Response.json({ url: publicUrl });
}
```

## R2 Setup Guide

### 1. Create R2 Bucket

1. Log in to Cloudflare Dashboard
2. Navigate to R2 Object Storage
3. Create a new bucket
4. Configure public access if needed (or use signed URLs)

### 2. Generate API Tokens

1. Go to R2 → Manage R2 API Tokens
2. Create API Token with read/write permissions
3. Note down:
   - Access Key ID
   - Secret Access Key
   - Account ID (visible in R2 settings)

### 3. Configure R2 Endpoint

```
Format: https://<ACCOUNT_ID>.r2.cloudflarestorage.com
```

### 4. Public Access (Optional)

For public file access:
1. Enable Custom Domains or R2.dev subdomain
2. Configure bucket as public
3. Use CDN-friendly URLs: `https://pub-<hash>.r2.dev/<key>`

## Common Patterns

### Pattern 1: Direct Upload to R2 + Metadata in Supabase

**Use Case**: User-generated content (images, videos)

```typescript
async function uploadFile(file: File, userId: string) {
  // 1. Upload to R2
  const key = `users/${userId}/${Date.now()}-${file.name}`;
  await uploadToR2(key, file);

  // 2. Save metadata to Supabase
  const { data } = await supabase
    .from('user_files')
    .insert({
      user_id: userId,
      file_key: key,
      file_name: file.name,
      file_size: file.size,
      r2_url: `https://pub-xxx.r2.dev/${key}`,
    })
    .select()
    .single();

  return data;
}
```

### Pattern 2: Signed URLs for Private Files

**Use Case**: Private files requiring access control

```typescript
import { getSignedUrl } from '@aws-sdk/s3-request-presigner';
import { GetObjectCommand } from '@aws-sdk/client-s3';

async function generateSignedUrl(fileKey: string, expiresIn = 3600) {
  // 1. Check authorization in Supabase
  const { data: file } = await supabase
    .from('user_files')
    .select('*')
    .eq('file_key', fileKey)
    .single();

  if (!file) throw new Error('File not found');

  // 2. Generate R2 signed URL
  const command = new GetObjectCommand({
    Bucket: R2_BUCKET_NAME,
    Key: fileKey,
  });

  const signedUrl = await getSignedUrl(s3Client, command, { expiresIn });

  return signedUrl;
}
```

### Pattern 3: Analytics with Foreign Data Wrappers

**Use Case**: Query R2 logs or analytics data from PostgreSQL

```sql
-- Query R2 data directly in PostgreSQL
SELECT
  date_trunc('day', last_modified) as day,
  count(*) as file_count,
  sum(size) as total_size
FROM r2_files
WHERE last_modified >= now() - interval '30 days'
GROUP BY day
ORDER BY day DESC;
```

## Gotchas & Limitations

### ⚠️ Hosted Supabase Limitations

1. **No Native Backend Replacement**: You cannot replace Supabase's built-in Storage with R2 on hosted plans
2. **Storage Pricing**: Still charged for Supabase Storage if using their service
3. **No Built-in Migration**: Must manually migrate files if switching to R2

### ⚠️ Self-Hosted Considerations

1. **URL Routing**: Files route through your server, not directly from R2
   - **Impact**: Consumes your server's bandwidth
   - **Solution**: Use R2 public URLs directly or configure CDN

2. **Uptime Responsibility**: You manage server availability and maintenance

3. **Configuration Complexity**: Requires Docker/infrastructure knowledge

### ⚠️ R2-Specific Considerations

1. **Region Handling**: Always use `region: 'auto'` for R2 (not a specific AWS region)

2. **Path-Style URLs**: Some S3 clients default to virtual-hosted-style URLs
   - R2 supports both styles
   - Set `path_style_url: false` for modern clients

3. **CORS Configuration**: Must configure CORS on R2 bucket for browser uploads
   ```json
   [
     {
       "AllowedOrigins": ["https://yourdomain.com"],
       "AllowedMethods": ["GET", "PUT", "POST", "DELETE"],
       "AllowedHeaders": ["*"],
       "MaxAgeSeconds": 3600
     }
   ]
   ```

4. **Public Access vs Private**:
   - Public: Use R2.dev subdomain or custom domain
   - Private: Use signed URLs (pre-signed S3 URLs work with R2)

### ⚠️ Cost Considerations

**R2 Free Tier**:
- 10 GB storage
- 1 million Class A operations/month (writes)
- 10 million Class B operations/month (reads)
- Zero egress fees (unlimited)

**Paid Tier** (after free tier):
- $0.015/GB/month storage
- $4.50 per million Class A operations
- $0.36 per million Class B operations
- **Still zero egress fees**

**Comparison**:
- Supabase Free: 1GB storage, 2GB egress
- Supabase Pro: 100GB storage, 200GB egress, then $0.021/GB egress
- **R2 Advantage**: Zero egress can save hundreds/thousands for high-traffic apps

### ⚠️ Feature Parity

**R2 Limitations vs Supabase Storage**:
- ❌ No built-in RLS (Row Level Security) - handle in application
- ❌ No built-in image transformations - use Cloudflare Images separately
- ❌ No automatic thumbnail generation
- ✅ S3 API compatibility (wide ecosystem support)
- ✅ Zero egress fees
- ✅ Global edge network via Cloudflare CDN

## Best Practices

### 1. Use Hybrid Architecture

✅ **Recommended**:
```
- Supabase: Auth, database, RLS policies, metadata
- R2: File storage, delivery, large objects
```

❌ **Not Recommended**:
```
- Trying to replace Supabase Storage completely on hosted plans
- Storing file metadata only in R2 (no relational queries)
```

### 2. Implement Proper Access Control

```typescript
// Check authorization before generating R2 URLs
async function getFileUrl(fileId: string, userId: string) {
  // 1. Verify ownership in Supabase
  const { data: file, error } = await supabase
    .from('files')
    .select('*')
    .eq('id', fileId)
    .eq('user_id', userId)
    .single();

  if (error || !file) {
    throw new Error('Unauthorized');
  }

  // 2. Return R2 URL (public or signed)
  return file.r2_url;
}
```

### 3. Use CDN for Public Files

Configure R2 with custom domain + Cloudflare CDN:
```
R2 Bucket → Custom Domain → Cloudflare CDN → Users
```

Benefits:
- Faster global delivery
- Caching at edge
- DDoS protection
- Free SSL

### 4. Optimize for Cost

1. **Use R2 for large files** (videos, images, archives)
2. **Use Supabase Storage for small files** requiring tight integration
3. **Implement lifecycle policies** to delete old files
4. **Compress before upload** to reduce storage costs

### 5. Security Hardening

```typescript
// Generate time-limited signed URLs for sensitive files
const signedUrl = await getSignedUrl(
  s3Client,
  new GetObjectCommand({
    Bucket: R2_BUCKET_NAME,
    Key: fileKey,
  }),
  {
    expiresIn: 900, // 15 minutes
  }
);

// Store audit logs in Supabase
await supabase.from('file_access_logs').insert({
  user_id: userId,
  file_id: fileId,
  action: 'download',
  ip_address: request.ip,
  timestamp: new Date(),
});
```

### 6. Error Handling

```typescript
async function uploadToR2WithRetry(key: string, file: File, maxRetries = 3) {
  for (let attempt = 1; attempt <= maxRetries; attempt++) {
    try {
      await s3Client.send(
        new PutObjectCommand({
          Bucket: R2_BUCKET_NAME,
          Key: key,
          Body: Buffer.from(await file.arrayBuffer()),
        })
      );
      return { success: true, key };
    } catch (error) {
      console.error(`Upload attempt ${attempt} failed:`, error);

      if (attempt === maxRetries) {
        // Log failure to Supabase for monitoring
        await supabase.from('upload_failures').insert({
          file_name: file.name,
          error_message: error.message,
          attempts: maxRetries,
        });

        throw error;
      }

      // Exponential backoff
      await new Promise(resolve => setTimeout(resolve, 1000 * Math.pow(2, attempt)));
    }
  }
}
```

## Migration Path from Supabase Storage to R2

### Step 1: Setup R2

1. Create R2 bucket
2. Generate API credentials
3. Configure CORS if needed

### Step 2: Implement Dual Write

```typescript
// Write to both Supabase and R2 during migration
async function uploadFileDualWrite(file: File, userId: string) {
  const fileName = `${Date.now()}-${file.name}`;

  // Upload to Supabase Storage (existing)
  const { data: supabaseFile } = await supabase.storage
    .from('uploads')
    .upload(fileName, file);

  // Also upload to R2 (new)
  const r2Key = `uploads/${fileName}`;
  await uploadToR2(r2Key, file);

  // Store both URLs
  await supabase.from('files').insert({
    user_id: userId,
    file_name: file.name,
    supabase_url: supabaseFile.path,
    r2_url: `https://pub-xxx.r2.dev/${r2Key}`,
  });
}
```

### Step 3: Migrate Existing Files

```typescript
// Batch migration script
async function migrateFilesToR2(batchSize = 100) {
  let offset = 0;

  while (true) {
    // Get batch of files from Supabase
    const { data: files } = await supabase
      .from('files')
      .select('*')
      .is('r2_url', null)
      .range(offset, offset + batchSize - 1);

    if (!files || files.length === 0) break;

    for (const file of files) {
      try {
        // Download from Supabase Storage
        const { data: blob } = await supabase.storage
          .from('uploads')
          .download(file.supabase_url);

        // Upload to R2
        const r2Key = `migrated/${file.supabase_url}`;
        await uploadToR2(r2Key, blob);

        // Update metadata
        await supabase
          .from('files')
          .update({ r2_url: `https://pub-xxx.r2.dev/${r2Key}` })
          .eq('id', file.id);

        console.log(`Migrated: ${file.file_name}`);
      } catch (error) {
        console.error(`Failed to migrate ${file.file_name}:`, error);
      }
    }

    offset += batchSize;
  }
}
```

### Step 4: Switch to R2-Only

```typescript
// Once migration complete, use R2 only
async function uploadFileR2Only(file: File, userId: string) {
  const r2Key = `uploads/${userId}/${Date.now()}-${file.name}`;
  await uploadToR2(r2Key, file);

  const { data } = await supabase.from('files').insert({
    user_id: userId,
    file_name: file.name,
    r2_url: `https://pub-xxx.r2.dev/${r2Key}`,
  });

  return data;
}
```

## Related Topics

- **S3 Compatibility**: Supabase Storage S3 protocol support
- **Foreign Data Wrappers**: PostgreSQL extensions for external data
- **Supabase Auth**: User authentication and RLS policies
- **Cloudflare Workers**: Edge computing for file processing
- **CDN Integration**: Content delivery optimization

## Additional Resources

- [Supabase S3 Wrapper Documentation](https://supabase.com/docs/guides/database/extensions/wrappers/s3)
- [Supabase Storage Self-hosting Config](https://supabase.com/docs/guides/self-hosting/storage/config)
- [Cloudflare R2 S3 API Compatibility](https://developers.cloudflare.com/r2/api/s3/api/)
- [GitHub Discussion: R2 with Self-Hosted Supabase](https://github.com/orgs/supabase/discussions/22534)
- [GitHub Discussion: Storage Options Feature Request](https://github.com/orgs/supabase/discussions/982)

---

**Last Verified**: December 20, 2025
**API Compatibility**: Supabase Storage S3 Protocol (Public Alpha), R2 S3 API (GA)
