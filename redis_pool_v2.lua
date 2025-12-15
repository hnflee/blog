local redis = require "resty.redis"
local consistent_hash = require "consistent_hash"
local semaphore = require "ngx.semaphore"

local _M = { _VERSION = '0.2.0' }

-- 默认配置
local DEFAULT_TIMEOUT = 1000              -- 连接/读写超时 1秒
local DEFAULT_POOL_SIZE = 100             -- 每个pool的最大空闲连接数
local DEFAULT_REPLICAS = 10               -- 虚拟节点数
local DEFAULT_KEEPALIVE_TIMEOUT = 10000   -- 空闲连接保持时间 10秒
local DEFAULT_MAX_CONNECTIONS = 100       -- 每个分片最大并发连接数（新增）
local DEFAULT_WAIT_TIMEOUT = 5            -- 等待信号量超时时间（秒）

function _M.new(self, shards, options)
    options = options or {}

    if not shards or #shards == 0 then
        ngx.log(ngx.ERR, "错误：没有提供分片节点")
        return nil, "no shards provided"
    end

    ngx.log(ngx.INFO, "初始化分片，节点数: " .. #shards .. ", 节点列表: " .. table.concat(shards, ", "))

    local timeout = options.timeout or DEFAULT_TIMEOUT
    local pool_size = options.pool_size or DEFAULT_POOL_SIZE
    local replicas = options.replicas or DEFAULT_REPLICAS
    local password = options.password
    local keepalive_timeout = options.keepalive_timeout or DEFAULT_KEEPALIVE_TIMEOUT
    local max_connections = options.max_connections or DEFAULT_MAX_CONNECTIONS
    local wait_timeout = options.wait_timeout or DEFAULT_WAIT_TIMEOUT

    local ring = consistent_hash:new(shards, replicas)
    if not ring or not ring.nodes or #ring.nodes == 0 then
        ngx.log(ngx.ERR, "错误：一致性哈希环初始化失败")
        return nil, "failed to initialize consistent hash ring"
    end

    -- 为每个分片创建信号量，限制最大并发连接数
    local pools = {}
    for _, shard in ipairs(shards) do
        local host, port = shard:match("([^:]+):?(%d*)")
        port = tonumber(port) or 6379

        -- 创建信号量限制并发连接
        local sema, err = semaphore.new(max_connections)
        if not sema then
            ngx.log(ngx.ERR, "创建信号量失败 [" .. shard .. "]: " .. err)
            return nil, "failed to create semaphore: " .. err
        end

        ngx.log(ngx.INFO, "配置分片: host=" .. host .. ", port=" .. port .. ", max_connections=" .. max_connections)

        pools[shard] = {
            host = host,
            port = port,
            password = password,
            semaphore = sema,           -- 信号量控制并发
            active_count = 0            -- 当前活跃连接数（用于监控）
        }
    end

    local instance = {
        ring = ring,
        pools = pools,
        timeout = timeout,
        pool_size = pool_size,
        keepalive_timeout = keepalive_timeout,
        max_connections = max_connections,
        wait_timeout = wait_timeout,
        options = options
    }

    setmetatable(instance, { __index = _M })

    return instance
end

-- 获取连接（带并发控制）
function _M.get_connection(self, key)
    local shard = consistent_hash:get_node(self.ring, key)
    if not shard then
        return nil, "no available shard"
    end

    local pool = self.pools[shard]
    if not pool then
        return nil, "shard not found: " .. shard
    end

    -- 等待获取信号量（限制并发连接数）
    local ok, err = pool.semaphore:wait(self.wait_timeout)
    if not ok then
        ngx.log(ngx.WARN, "获取信号量超时 [" .. shard .. "], 当前活跃连接: " .. pool.active_count .. ", 错误: " .. err)
        return nil, "connection pool exhausted, please retry later"
    end

    -- 成功获取信号量，更新活跃连接计数
    pool.active_count = pool.active_count + 1

    local pool_name = pool.host .. ":" .. pool.port
    local red = redis:new()
    red:set_timeout(self.timeout)

    local connect_ok, connect_err = red:connect(pool.host, pool.port, {
        pool = pool_name,
        pool_size = self.pool_size
    })

    if not connect_ok then
        -- 连接失败，释放信号量
        pool.semaphore:post(1)
        pool.active_count = pool.active_count - 1
        ngx.log(ngx.ERR, "Redis连接失败 [" .. pool_name .. "]: " .. connect_err)
        return nil, "failed to connect: " .. connect_err
    end

    local reused_times = red:get_reused_times()

    -- 新连接需要认证
    if pool.password and reused_times == 0 then
        local auth_ok, auth_err = red:auth(pool.password)
        if not auth_ok then
            red:close()
            pool.semaphore:post(1)
            pool.active_count = pool.active_count - 1
            return nil, "failed to authenticate: " .. auth_err
        end
    end

    -- 返回连接和分片信息（用于归还时释放信号量）
    return red, nil, shard
end

-- 归还连接（带信号量释放）
function _M.return_connection(self, red, shard)
    if not red then
        return
    end

    local pool = self.pools[shard]

    local ok, err = red:set_keepalive(self.keepalive_timeout, self.pool_size)
    if not ok then
        ngx.log(ngx.WARN, "设置keepalive失败: " .. tostring(err))
        red:close()
    end

    -- 释放信号量
    if pool then
        pool.semaphore:post(1)
        pool.active_count = pool.active_count - 1
    end

    return true
end

-- 关闭连接（出错时）
function _M.close_connection(self, red, shard)
    if red then
        red:close()
    end

    -- 释放信号量
    local pool = self.pools[shard]
    if pool then
        pool.semaphore:post(1)
        pool.active_count = pool.active_count - 1
    end
end

-- 获取连接池状态（用于监控）
function _M.get_stats(self)
    local stats = {}
    for shard, pool in pairs(self.pools) do
        stats[shard] = {
            active_connections = pool.active_count,
            max_connections = self.max_connections,
            available = self.max_connections - pool.active_count
        }
    end
    return stats
end

-- ============ Redis 命令封装 ============

-- 通用命令执行器（减少重复代码）
local function execute_command(self, key, cmd, ...)
    local red, err, shard = self:get_connection(key)
    if not red then
        return nil, err
    end

    local method = red[cmd]
    if not method then
        self:close_connection(red, shard)
        return nil, "unknown command: " .. cmd
    end

    local result, cmd_err = method(red, ...)

    -- 判断是否为错误（不同命令返回值不同）
    if cmd_err then
        ngx.log(ngx.ERR, cmd .. "失败 key=[" .. tostring(key) .. "]: " .. cmd_err)
        self:close_connection(red, shard)
        return nil, cmd .. " failed: " .. cmd_err
    end

    self:return_connection(red, shard)
    return result
end

-- SET
function _M.set(self, key, value)
    local red, err, shard = self:get_connection(key)
    if not red then
        return nil, err
    end

    local ok, err = red:set(key, value)
    if not ok then
        ngx.log(ngx.ERR, "SET失败 key=[" .. key .. "]: " .. err)
        self:close_connection(red, shard)
        return nil, "failed to set: " .. err
    end

    self:return_connection(red, shard)
    return true
end

-- SETEX
function _M.setex(self, key, expire_time, value)
    local red, err, shard = self:get_connection(key)
    if not red then
        return nil, err
    end

    local ok, err = red:setex(key, tonumber(expire_time), value)
    if not ok then
        ngx.log(ngx.ERR, "SETEX失败 key=[" .. key .. "]: " .. err)
        self:close_connection(red, shard)
        return nil, "failed to setex: " .. err
    end

    self:return_connection(red, shard)
    return true
end

-- SETNX
function _M.setnx(self, key, value)
    local red, err, shard = self:get_connection(key)
    if not red then
        return nil, err
    end

    local result, err = red:setnx(key, value)
    if err then
        ngx.log(ngx.ERR, "SETNX失败 key=[" .. key .. "]: " .. err)
        self:close_connection(red, shard)
        return nil, "failed to setnx: " .. err
    end

    self:return_connection(red, shard)
    return result
end

-- GET
function _M.get(self, key)
    local red, err, shard = self:get_connection(key)
    if not red then
        return nil, err
    end

    local value, err = red:get(key)
    if err then
        ngx.log(ngx.ERR, "GET失败 key=[" .. key .. "]: " .. err)
        self:close_connection(red, shard)
        return nil, "failed to get: " .. err
    end

    self:return_connection(red, shard)
    return value
end

-- MGET (注意：需要所有key在同一分片)
function _M.mget(self, key, ...)
    local red, err, shard = self:get_connection(key)
    if not red then
        return nil, err
    end

    local values, err = red:mget(key, ...)
    if err then
        ngx.log(ngx.ERR, "MGET失败: " .. err)
        self:close_connection(red, shard)
        return nil, "failed to mget: " .. err
    end

    self:return_connection(red, shard)
    return values
end

-- INCR
function _M.incr(self, key)
    local red, err, shard = self:get_connection(key)
    if not red then
        return nil, err
    end

    local result, err = red:incr(key)
    if not result then
        ngx.log(ngx.ERR, "INCR失败 key=[" .. key .. "]: " .. err)
        self:close_connection(red, shard)
        return nil, "failed to incr: " .. err
    end

    self:return_connection(red, shard)
    return result
end

-- INCRBY
function _M.incrby(self, key, increment)
    local red, err, shard = self:get_connection(key)
    if not red then
        return nil, err
    end

    local result, err = red:incrby(key, tonumber(increment))
    if not result then
        ngx.log(ngx.ERR, "INCRBY失败 key=[" .. key .. "]: " .. err)
        self:close_connection(red, shard)
        return nil, "failed to incrby: " .. err
    end

    self:return_connection(red, shard)
    return result
end

-- DECR
function _M.decr(self, key)
    local red, err, shard = self:get_connection(key)
    if not red then
        return nil, err
    end

    local result, err = red:decr(key)
    if not result then
        ngx.log(ngx.ERR, "DECR失败 key=[" .. key .. "]: " .. err)
        self:close_connection(red, shard)
        return nil, "failed to decr: " .. err
    end

    self:return_connection(red, shard)
    return result
end

-- DEL
function _M.del(self, key)
    local red, err, shard = self:get_connection(key)
    if not red then
        return nil, err
    end

    local result, err = red:del(key)
    if not result then
        ngx.log(ngx.ERR, "DEL失败 key=[" .. key .. "]: " .. err)
        self:close_connection(red, shard)
        return nil, "failed to del: " .. err
    end

    self:return_connection(red, shard)
    return result
end

-- EXPIRE
function _M.expire(self, key, expire_time)
    local red, err, shard = self:get_connection(key)
    if not red then
        return nil, err
    end

    local ok, err = red:expire(key, tonumber(expire_time))
    if not ok then
        ngx.log(ngx.ERR, "EXPIRE失败 key=[" .. key .. "]: " .. err)
        self:close_connection(red, shard)
        return nil, "EXPIRE failed: " .. err
    end

    self:return_connection(red, shard)
    return true
end

-- EXISTS
function _M.exists(self, key)
    local red, err, shard = self:get_connection(key)
    if not red then
        return nil, err
    end

    local exists, err = red:exists(key)
    if err then
        ngx.log(ngx.ERR, "EXISTS失败 key=[" .. key .. "]: " .. err)
        self:close_connection(red, shard)
        return nil, "EXISTS failed: " .. err
    end

    self:return_connection(red, shard)
    return exists
end

-- TTL
function _M.ttl(self, key)
    local red, err, shard = self:get_connection(key)
    if not red then
        return nil, err
    end

    local ttl, err = red:ttl(key)
    if err then
        ngx.log(ngx.ERR, "TTL失败 key=[" .. key .. "]: " .. err)
        self:close_connection(red, shard)
        return nil, "TTL failed: " .. err
    end

    self:return_connection(red, shard)
    return ttl
end

-- ============ Hash 命令 ============

-- HSET
function _M.hset(self, key, field, value)
    local red, err, shard = self:get_connection(key)
    if not red then
        return nil, err
    end

    local result, err = red:hset(key, field, value)
    if not result then
        ngx.log(ngx.ERR, "HSET失败 key=[" .. key .. "]: " .. err)
        self:close_connection(red, shard)
        return nil, "HSET failed: " .. err
    end

    self:return_connection(red, shard)
    return result
end

-- HMSET
function _M.hmset(self, key, ...)
    local red, err, shard = self:get_connection(key)
    if not red then
        return nil, err
    end

    local result, err = red:hmset(key, ...)
    if not result then
        ngx.log(ngx.ERR, "HMSET失败 key=[" .. key .. "]: " .. err)
        self:close_connection(red, shard)
        return nil, "HMSET failed: " .. err
    end

    self:return_connection(red, shard)
    return result
end

-- HGET
function _M.hget(self, key, field)
    local red, err, shard = self:get_connection(key)
    if not red then
        return nil, err
    end

    local value, err = red:hget(key, field)
    if err then
        ngx.log(ngx.ERR, "HGET失败 key=[" .. key .. "]: " .. err)
        self:close_connection(red, shard)
        return nil, "HGET failed: " .. err
    end

    self:return_connection(red, shard)
    return value
end

-- HMGET
function _M.hmget(self, key, ...)
    local red, err, shard = self:get_connection(key)
    if not red then
        return nil, err
    end

    local values, err = red:hmget(key, ...)
    if err then
        ngx.log(ngx.ERR, "HMGET失败 key=[" .. key .. "]: " .. err)
        self:close_connection(red, shard)
        return nil, "HMGET failed: " .. err
    end

    self:return_connection(red, shard)
    return values
end

-- HGETALL
function _M.hgetall(self, key)
    local red, err, shard = self:get_connection(key)
    if not red then
        return nil, err
    end

    local result, err = red:hgetall(key)
    if err then
        ngx.log(ngx.ERR, "HGETALL失败 key=[" .. key .. "]: " .. err)
        self:close_connection(red, shard)
        return nil, "HGETALL failed: " .. err
    end

    self:return_connection(red, shard)
    return result
end

-- HDEL
function _M.hdel(self, key, ...)
    local red, err, shard = self:get_connection(key)
    if not red then
        return nil, err
    end

    local result, err = red:hdel(key, ...)
    if not result then
        ngx.log(ngx.ERR, "HDEL失败 key=[" .. key .. "]: " .. err)
        self:close_connection(red, shard)
        return nil, "HDEL failed: " .. err
    end

    self:return_connection(red, shard)
    return result
end

-- HEXISTS
function _M.hexists(self, key, field)
    local red, err, shard = self:get_connection(key)
    if not red then
        return nil, err
    end

    local result, err = red:hexists(key, field)
    if err then
        ngx.log(ngx.ERR, "HEXISTS失败 key=[" .. key .. "]: " .. err)
        self:close_connection(red, shard)
        return nil, "HEXISTS failed: " .. err
    end

    self:return_connection(red, shard)
    return result
end

-- HINCRBY
function _M.hincrby(self, key, field, increment)
    local red, err, shard = self:get_connection(key)
    if not red then
        return nil, err
    end

    local result, err = red:hincrby(key, field, tonumber(increment))
    if not result then
        ngx.log(ngx.ERR, "HINCRBY失败 key=[" .. key .. "]: " .. err)
        self:close_connection(red, shard)
        return nil, "HINCRBY failed: " .. err
    end

    self:return_connection(red, shard)
    return result
end

-- ============ Set 命令 ============

-- SADD
function _M.sadd(self, key, ...)
    local red, err, shard = self:get_connection(key)
    if not red then
        return nil, err
    end

    local result, err = red:sadd(key, ...)
    if not result then
        ngx.log(ngx.ERR, "SADD失败 key=[" .. key .. "]: " .. err)
        self:close_connection(red, shard)
        return nil, "SADD failed: " .. err
    end

    self:return_connection(red, shard)
    return result
end

-- SREM
function _M.srem(self, key, ...)
    local red, err, shard = self:get_connection(key)
    if not red then
        return nil, err
    end

    local result, err = red:srem(key, ...)
    if not result then
        ngx.log(ngx.ERR, "SREM失败 key=[" .. key .. "]: " .. err)
        self:close_connection(red, shard)
        return nil, "SREM failed: " .. err
    end

    self:return_connection(red, shard)
    return result
end

-- SISMEMBER
function _M.sismember(self, key, member)
    local red, err, shard = self:get_connection(key)
    if not red then
        return nil, err
    end

    local result, err = red:sismember(key, member)
    if err then
        ngx.log(ngx.ERR, "SISMEMBER失败 key=[" .. key .. "]: " .. err)
        self:close_connection(red, shard)
        return nil, "SISMEMBER failed: " .. err
    end

    self:return_connection(red, shard)
    return result
end

-- SMEMBERS
function _M.smembers(self, key)
    local red, err, shard = self:get_connection(key)
    if not red then
        return nil, err
    end

    local result, err = red:smembers(key)
    if err then
        ngx.log(ngx.ERR, "SMEMBERS失败 key=[" .. key .. "]: " .. err)
        self:close_connection(red, shard)
        return nil, "SMEMBERS failed: " .. err
    end

    self:return_connection(red, shard)
    return result
end

-- SCARD
function _M.scard(self, key)
    local red, err, shard = self:get_connection(key)
    if not red then
        return nil, err
    end

    local result, err = red:scard(key)
    if err then
        ngx.log(ngx.ERR, "SCARD失败 key=[" .. key .. "]: " .. err)
        self:close_connection(red, shard)
        return nil, "SCARD failed: " .. err
    end

    self:return_connection(red, shard)
    return result
end

-- ============ List 命令 ============

-- LPUSH
function _M.lpush(self, key, ...)
    local red, err, shard = self:get_connection(key)
    if not red then
        return nil, err
    end

    local result, err = red:lpush(key, ...)
    if not result then
        ngx.log(ngx.ERR, "LPUSH失败 key=[" .. key .. "]: " .. err)
        self:close_connection(red, shard)
        return nil, "LPUSH failed: " .. err
    end

    self:return_connection(red, shard)
    return result
end

-- RPUSH
function _M.rpush(self, key, ...)
    local red, err, shard = self:get_connection(key)
    if not red then
        return nil, err
    end

    local result, err = red:rpush(key, ...)
    if not result then
        ngx.log(ngx.ERR, "RPUSH失败 key=[" .. key .. "]: " .. err)
        self:close_connection(red, shard)
        return nil, "RPUSH failed: " .. err
    end

    self:return_connection(red, shard)
    return result
end

-- LPOP
function _M.lpop(self, key)
    local red, err, shard = self:get_connection(key)
    if not red then
        return nil, err
    end

    local result, err = red:lpop(key)
    if err then
        ngx.log(ngx.ERR, "LPOP失败 key=[" .. key .. "]: " .. err)
        self:close_connection(red, shard)
        return nil, "LPOP failed: " .. err
    end

    self:return_connection(red, shard)
    return result
end

-- RPOP
function _M.rpop(self, key)
    local red, err, shard = self:get_connection(key)
    if not red then
        return nil, err
    end

    local result, err = red:rpop(key)
    if err then
        ngx.log(ngx.ERR, "RPOP失败 key=[" .. key .. "]: " .. err)
        self:close_connection(red, shard)
        return nil, "RPOP failed: " .. err
    end

    self:return_connection(red, shard)
    return result
end

-- LRANGE
function _M.lrange(self, key, start, stop)
    local red, err, shard = self:get_connection(key)
    if not red then
        return nil, err
    end

    local result, err = red:lrange(key, start, stop)
    if err then
        ngx.log(ngx.ERR, "LRANGE失败 key=[" .. key .. "]: " .. err)
        self:close_connection(red, shard)
        return nil, "LRANGE failed: " .. err
    end

    self:return_connection(red, shard)
    return result
end

-- LLEN
function _M.llen(self, key)
    local red, err, shard = self:get_connection(key)
    if not red then
        return nil, err
    end

    local result, err = red:llen(key)
    if err then
        ngx.log(ngx.ERR, "LLEN失败 key=[" .. key .. "]: " .. err)
        self:close_connection(red, shard)
        return nil, "LLEN failed: " .. err
    end

    self:return_connection(red, shard)
    return result
end

-- ============ 通用命令 ============

-- 执行任意命令
function _M.execute(self, key, cmd, ...)
    local red, err, shard = self:get_connection(key)
    if not red then
        return nil, err
    end

    local method = red[cmd]
    if not method then
        self:close_connection(red, shard)
        return nil, "unknown command: " .. cmd
    end

    local result, err = method(red, ...)
    if err then
        ngx.log(ngx.ERR, cmd .. "失败: " .. err)
        self:close_connection(red, shard)
        return nil, cmd .. " failed: " .. err
    end

    self:return_connection(red, shard)
    return result
end

-- 执行 pipeline
function _M.pipeline(self, key, commands)
    local red, err, shard = self:get_connection(key)
    if not red then
        return nil, err
    end

    red:init_pipeline()

    for _, cmd in ipairs(commands) do
        local method = red[cmd[1]]
        if method then
            method(red, unpack(cmd, 2))
        end
    end

    local results, err = red:commit_pipeline()
    if not results then
        ngx.log(ngx.ERR, "Pipeline执行失败: " .. err)
        self:close_connection(red, shard)
        return nil, "pipeline failed: " .. err
    end

    self:return_connection(red, shard)
    return results
end

return _M
