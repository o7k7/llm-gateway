-- Return unused tokens to a bucket after a request completes.
--
-- KEYS[1] = bucket hash key
-- ARGV[1] = capacity   (to cap the bucket; can't exceed this)
-- ARGV[2] = amount     (tokens to add back; clamped to >= 0)
-- ARGV[3] = now_ms
-- ARGV[4] = ttl_ms
--
-- Returns: remaining tokens after refund (int)

local capacity = tonumber(ARGV[1])
local amount   = tonumber(ARGV[2])
local now_ms   = tonumber(ARGV[3])
local ttl_ms   = tonumber(ARGV[4])

if capacity == nil or amount == nil or now_ms == nil or ttl_ms == nil then
    return redis.error_reply("token_bucket_refund: bad ARGV")
end
if amount < 0 then amount = 0 end

local exists = redis.call("EXISTS", KEYS[1])
if exists == 0 then
    return capacity
end

local state = redis.call("HMGET", KEYS[1], "tokens")
local tokens = tonumber(state[1])
if tokens == nil then
    tokens = capacity
end

tokens = math.min(capacity, tokens + amount)
redis.call("HMSET", KEYS[1], "tokens", tokens, "ts", now_ms)
redis.call("PEXPIRE", KEYS[1], ttl_ms)

return math.floor(tokens)
