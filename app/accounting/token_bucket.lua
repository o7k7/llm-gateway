-- Atomic token-bucket refill + consume.
--
-- KEYS[1] = bucket hash key (e.g. "tb:tenant123:tpm")
--
-- ARGV[1] = capacity       (max tokens the bucket can hold)
-- ARGV[2] = refill_per_sec (tokens added per second)
-- ARGV[3] = now_ms         (current time from the client, in ms since epoch)
-- ARGV[4] = cost           (tokens the caller wants to consume)
-- ARGV[5] = ttl_ms         (expire the key after this many ms of inactivity)
--
-- Returns: {allowed, remaining}
--   allowed   = 1 if the request is allowed and tokens were deducted; 0 otherwise
--   remaining = tokens left in the bucket after the operation (int)

local capacity       = tonumber(ARGV[1])
local refill_per_sec = tonumber(ARGV[2])
local now_ms         = tonumber(ARGV[3])
local cost           = tonumber(ARGV[4])
local ttl_ms         = tonumber(ARGV[5])

if capacity == nil or refill_per_sec == nil or now_ms == nil or cost == nil or ttl_ms == nil then
    return redis.error_reply("token_bucket: bad ARGV")
end

if cost < 0 then
    return redis.error_reply("token_bucket: negative cost")
end

local state = redis.call("HMGET", KEYS[1], "tokens", "ts")
local tokens = tonumber(state[1])
local last_ts = tonumber(state[2])

if tokens == nil then
    tokens = capacity
    last_ts = now_ms
end

-- Refill based on elapsed time (guard against clock skew going backwards)
local elapsed_ms = now_ms - last_ts
if elapsed_ms > 0 then
    local refill = (elapsed_ms / 1000.0) * refill_per_sec
    tokens = math.min(capacity, tokens + refill)
end

local allowed = 0
if tokens >= cost then
    tokens = tokens - cost
    allowed = 1
end

redis.call("HMSET", KEYS[1], "tokens", tokens, "ts", now_ms)
redis.call("PEXPIRE", KEYS[1], ttl_ms)

return { allowed, math.floor(tokens) }
