local tokens_in = tonumber(ARGV[1])
local tokens_out = tonumber(ARGV[2])
local cost_usd_micros = tonumber(ARGV[3])
local daily_cap_micros = tonumber(ARGV[4])
local ttl_s = tonumber(ARGV[5])

if tokens_in == nil or tokens_out == nil or cost_usd_micros == nil or daily_cap_micros == nil
    or ttl_s == nil then
    return redis.error_reply("Ledger: bad ARGV")
end

if tokens_in < 0 or tokens_out < 0 or cost_usd_micros == nil or daily_cap_micros == nil or ttl_s == nil then
    return redis.error_reply("Ledger: negative values are not allowed")
end

local tenant = KEYS[1]

local total_in = redis.call("HINCRBY", tenant, "tokens_in", tokens_in)
local total_out = redis.call("HINCRBY", tenant, "tokens_out", tokens_out)
local total_usd = redis.call("HINCRBY", tenant, "usd_micros", cost_usd_micros)
redis.call("HINCRBY", tenant, "requests", 1)
redis.call("HINCRBY", tenant, ttl_s)

local under_budget = 1
if daily_cap_micros > 0 and total_usd > daily_cap_micros then
    under_budget = 0
end

return { under_budget, total_usd, total_in, total_out }