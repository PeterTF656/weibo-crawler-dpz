# Supabase Realtime: Private Channels and Broadcast Authorization

**Source**: Official Supabase Documentation
**Retrieved**: 2026-01-07
**Key URLs**:
- https://supabase.com/docs/guides/realtime/broadcast
- https://supabase.com/docs/guides/realtime/authorization

## Summary

Supabase Realtime supports both **public** and **private** channels. Private channels require authentication and Row Level Security (RLS) policies on the `realtime.messages` table. The critical requirement is that **broadcast messages must match the channel type**: private broadcasts only reach private channels, and public broadcasts only reach public channels. This is the most common cause of "messages not received" issues.

## Key Concepts

### Public vs Private Channels

**Public Channels:**
- Anyone can subscribe without authentication
- No RLS policies required
- Messages sent as public broadcasts reach these channels
- Default behavior when no configuration specified

**Private Channels:**
- Require authentication (JWT token)
- Require RLS policies on `realtime.messages` table
- Clients must subscribe with `config: { private: true }`
- Messages sent as private broadcasts reach these channels
- **By default, all database broadcasts are private**

### Critical Matching Rule

> **The `private` flag on the broadcast MUST match the channel subscription configuration.**

- If frontend subscribes with `config: { private: true }`, backend must send with `private: true`
- If frontend subscribes without `private: true` (public), backend must send as public
- Mismatched configurations result in messages not being delivered
- Channels with the same topic name but different public/private settings are treated as **separate channels**

## Row Level Security (RLS) Setup

Private channels require RLS policies on the `realtime.messages` table to control authorization.

### Basic RLS Policies

Allow authenticated users to receive broadcasts:
```sql
CREATE POLICY "authenticated_users_can_receive"
ON realtime.messages
FOR SELECT
TO authenticated
USING (true);
```

Allow authenticated users to send broadcasts:
```sql
CREATE POLICY "authenticated_users_can_send"
ON realtime.messages
FOR INSERT
TO authenticated
WITH CHECK (true);
```

### How Authorization Works

1. When a client connects via WebSocket and joins a channel topic, Realtime:
   - Performs a query on `realtime.messages` table (then rolls it back)
   - Evaluates RLS policies based on:
     - The user's Auth JWT token
     - Request headers
     - The channel topic being joined

2. Realtime does **not** store messages in `realtime.messages` table
3. The table is used only for policy validation
4. Increased RLS complexity can impact connection performance

## REST API Endpoints for Broadcasting

### Option 1: `/rest/v1/rpc/broadcast` (Recommended for Private Channels)

**Endpoint**: `https://<project>.supabase.co/rest/v1/rpc/broadcast`

**Headers:**
```http
Content-Type: application/json
Authorization: Bearer <service-role-key>
apikey: <service-role-key>
```

**Request Body:**
```json
{
  "topic": "user:5f08568b-a81a-426e-b93c-dfffd705e25f:broadcast",
  "event": "message_sent",
  "payload": {
    "text": "Hello from server!",
    "user": "system",
    "timestamp": "2023-10-27T10:00:00Z"
  },
  "private": true
}
```

**Key Parameter:**
- `private` (boolean, required): Set to `true` for private channels, `false` for public

**Example (cURL):**
```bash
curl -X POST 'https://<project>.supabase.co/rest/v1/rpc/broadcast' \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer <service-role-key>' \
  -H 'apikey: <service-role-key>' \
  -d '{
    "topic": "user:123:broadcast",
    "event": "notification",
    "payload": {"message": "Test"},
    "private": true
  }'
```

### Option 2: `/realtime/v1/api/broadcast`

**Endpoint**: `https://<project>.supabase.co/realtime/v1/api/broadcast`

**Headers:**
```http
Content-Type: application/json
apikey: <supabase-token>
```

**Request Body:**
```json
{
  "messages": [
    {
      "topic": "user:xxx:broadcast",
      "event": "event_name",
      "payload": {"test": "data"}
    }
  ]
}
```

**Note**: This endpoint does **not** explicitly document a `private` parameter in the messages array. It may default to public broadcasts. **For private channels, use `/rest/v1/rpc/broadcast` instead.**

## Client-Side Subscription (Frontend)

### Subscribing to a Private Channel

```javascript
const supabase = createClient('URL', 'ANON_KEY')

const channel = supabase.channel('user:123:broadcast', {
  config: {
    private: true  // REQUIRED for private channels
  }
})

channel
  .on('broadcast', { event: 'notification' }, (payload) => {
    console.log('Received:', payload)
  })
  .subscribe((status, err) => {
    if (status === 'SUBSCRIBED') {
      console.log('Connected to private channel!')
    } else {
      console.error('Subscription error:', err)
    }
  })
```

### Subscribing to a Public Channel

```javascript
const channel = supabase.channel('public:room:lobby')
// No config.private needed for public channels

channel
  .on('broadcast', { event: 'message' }, (payload) => {
    console.log('Received:', payload)
  })
  .subscribe()
```

## Common Patterns

### Server-to-User Private Broadcasting

**Use Case**: Backend sends notifications to specific users via private channels

**Channel Naming**: `user:{user_id}:broadcast`

**Backend (Python example):**
```python
import httpx

async def broadcast_to_user(user_id: str, event: str, payload: dict):
    url = f"{supabase_url}/rest/v1/rpc/broadcast"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {service_role_key}",
        "apikey": service_role_key
    }
    data = {
        "topic": f"user:{user_id}:broadcast",
        "event": event,
        "payload": payload,
        "private": True  # CRITICAL for private channels
    }

    async with httpx.AsyncClient() as client:
        response = await client.post(url, headers=headers, json=data)
        return response.status_code
```

**Frontend (JavaScript):**
```javascript
// User must be authenticated
const { data: { user } } = await supabase.auth.getUser()

const channel = supabase.channel(`user:${user.id}:broadcast`, {
  config: { private: true }  // Must match backend's private: true
})

channel
  .on('broadcast', { event: 'notification' }, (payload) => {
    console.log('User notification:', payload)
  })
  .subscribe()
```

### RLS Policy for User-Specific Channels

```sql
-- Allow users to receive broadcasts only on their own channels
CREATE POLICY "users_receive_own_broadcasts"
ON realtime.messages
FOR SELECT
TO authenticated
USING (
  (realtime.topic())::text = 'user:' || auth.uid()::text || ':broadcast'
);

-- Allow service to send to any user channel
CREATE POLICY "service_can_send_user_broadcasts"
ON realtime.messages
FOR INSERT
TO service_role
WITH CHECK (
  (realtime.topic())::text LIKE 'user:%:broadcast'
);
```

## Gotchas and Important Considerations

### 1. **202 Accepted Doesn't Mean Delivered**

A `202 Accepted` response from the broadcast endpoint means:
- The server accepted the request
- The message was queued for delivery
- **It does NOT guarantee any clients received it**

If clients don't receive messages:
- Check if the `private` flag matches client subscription
- Verify clients are subscribed with correct authentication
- Check RLS policies allow the operation
- Verify channel topic names match exactly

### 2. **Public/Private Mismatch is Silent**

If you send a private broadcast but clients are subscribed to a public channel (or vice versa):
- No error is thrown
- Messages simply don't arrive
- The server returns success (202)
- **This is the most common mistake**

### 3. **Service Role Key vs User JWT**

- Backend broadcasts typically use **service role key** (bypasses RLS for sending)
- Frontend subscriptions use **user JWT** (subject to RLS policies)
- RLS policies on `realtime.messages` control who can **join** channels
- The service role can always send, but users must have SELECT permission to receive

### 4. **Channel Topic Names**

- Public and private channels with the **same topic name** are treated as **different channels**
- `room:123` (public) ≠ `room:123` (private)
- Use consistent naming conventions to avoid confusion

### 5. **Authentication Required for Private Channels**

- Clients must be authenticated with a valid JWT
- Anonymous/unauthenticated clients cannot join private channels
- Ensure `supabase.auth.signIn()` or equivalent is called before subscribing

### 6. **RLS Policy Performance**

- Complex RLS policies slow down channel join operations
- Keep policies simple for high-traffic channels
- Consider using separate tables or logic for complex authorization
- Test connection latency under load

### 7. **Database Broadcasts Default to Private**

When using Postgres triggers with `realtime.broadcast_changes()`:
- Database-triggered broadcasts are **private by default**
- The `is_private` parameter in `realtime.send()` controls this
- Example: `realtime.send(payload, event, topic, true)` for private

## Testing Checklist

When debugging "messages not received" issues:

- [ ] Verify backend uses `private: true` in broadcast payload
- [ ] Verify frontend subscribes with `config: { private: true }`
- [ ] Check RLS policies exist on `realtime.messages` table
- [ ] Confirm user is authenticated (check JWT token)
- [ ] Verify channel topic names match exactly (case-sensitive)
- [ ] Test RLS policies allow user to SELECT from `realtime.messages`
- [ ] Check service role key has permissions (if using service role)
- [ ] Verify Realtime is enabled in Supabase project settings
- [ ] Check for typos in event names (case-sensitive)
- [ ] Look at Realtime logs in Supabase dashboard

## Related Topics

- **Supabase Realtime Presence**: Track online users in channels
- **Supabase Realtime Postgres Changes**: Subscribe to database events
- **Supabase Auth**: User authentication required for private channels
- **PostgreSQL Row Level Security**: Security model underlying Realtime authorization
- **WebSocket vs REST**: When to use HTTP broadcast vs WebSocket `channel.send()`

## Version Notes

- Realtime Authorization is in **Public Beta**
- Requires `supabase-js` v2.44.0 or later for authorization features
- Broadcast via REST API available since `supabase-js` v2.37.0
- Broadcast Replay (for private channels) is in **Public Alpha**

---

**Last Updated**: 2026-01-07
