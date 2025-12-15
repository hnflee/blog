local redis = require "resty.redis"
local consistent_hash = require "consistent_hash"

local _M = { _VERSION = '0.1.1' }

-- 默认配置
local DEFAULT_TIMEOUT = 1000           -- 1秒超时
local DEFAULT_POOL_SIZE = 100          -- 连接池大小
local DEFAULT_REPLICAS = 10            -- 虚拟节点数
local DEFAULT_KEEPALIVE_TIMEOUT = 10000 -- keepalive超时 10秒
local DEFAULT_BACKLOG = 100            -- 排队连接数

function _M.new(self, shards, options)
    options = options or {}

    -- 验证shards参数
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
    local backlog = options.backlog or DEFAULT_BACKLOG

    local ring = consistent_hash:new(shards, replicas)
    if not ring or not ring.nodes or #ring.nodes == 0 then
        ngx.log(ngx.ERR, "错误：一致性哈希环初始化失败")
        return nil, "failed to initialize consistent hash ring"
    end

    -- 只存储配置信息，不预创建 redis 实例
    local pools = {}
    for _, shard in ipairs(shards) do
        local host, port = shard:match("([^:]+):?(%d*)")
        -- 修复：确保 port 是数字，空字符串时使用默认值
        port = tonumber(port) or 6379

        ngx.log(ngx.INFO, "配置分片: host=" .. host .. ", port=" .. port)

        pools[shard] = {
            host = host,
            port = port,
            password = password
        }
    end

    local instance = {
        ring = ring,
        pools = pools,
        timeout = timeout,
        pool_size = pool_size,
        keepalive_timeout = keepalive_timeout,
        backlog = backlog,
        options = options
    }

    setmetatable(instance, { __index = _M })

    return instance
end

-- 获取 Redis 连接
function _M.get_connection(self, key)
    local shard = consistent_hash:get_node(self.ring, key)
    if not shard then
        return nil, "no available shard"
    end

    local pool = self.pools[shard]
    if not pool then
        return nil, "shard not found: " .. shard
    end

    local pool_name = pool.host .. ":" .. pool.port
    local red = redis:new()
    red:set_timeout(self.timeout)

    -- 连接 Redis，使用连接池
    local ok, err = red:connect(pool.host, pool.port, {
        pool = pool_name,
        pool_size = self.pool_size,
        backlog = self.backlog
    })

    if not ok then
        ngx.log(ngx.ERR, "Redis连接失败 [" .. pool_name .. "]: " .. err)
        return nil, "failed to connect: " .. err
    end

    local reused_times = red:get_reused_times()

    -- 只在新连接时进行认证 (reused_times == 0 表示新连接)
    if pool.password and reused_times == 0 then
        local res, err = red:auth(pool.password)
        if not res then
            red:close()
            return nil, "failed to authenticate: " .. err
        end
        ngx.log(ngx.DEBUG, "Redis认证成功 [" .. pool_name .. "]")
    end

    return red, nil, reused_times
end

-- 归还连接到连接池
function _M.return_connection(self, red)
    if not red then
        return
    end

    local ok, err = red:set_keepalive(self.keepalive_timeout, self.pool_size)
    if not ok then
        ngx.log(ngx.WARN, "设置keepalive失败: " .. tostring(err))
        red:close()
        return false, err
    end

    return true
end

-- 关闭连接（出错时使用）
function _M.close_connection(self, red)
    if red then
        red:close()
    end
end

-- SET 命令
function _M.set(self, key, value)
    local red, err = self:get_connection(key)
    if not red then
        return nil, err
    end

    local ok, err = red:set(key, value)
    if not ok then
        ngx.log(ngx.ERR, "SET失败 key=[" .. key .. "]: " .. err)
        self:close_connection(red)
        return nil, "failed to set: " .. err
    end

    ngx.log(ngx.INFO, "SET成功 key=[" .. key .. "] value=[" .. tostring(value) .. "]")
    self:return_connection(red)

    return true
end

-- SETEX 命令
function _M.setex(self, key, expire_time, value)
    local red, err = self:get_connection(key)
    if not red then
        ngx.log(ngx.ERR, "获取连接失败: " .. err)
        return nil, err
    end

    local ok, err = red:setex(key, tonumber(expire_time), value)
    if not ok then
        ngx.log(ngx.ERR, "SETEX失败 key=[" .. key .. "]: " .. err)
        self:close_connection(red)
        return nil, "failed to setex: " .. err
    end

    ngx.log(ngx.INFO, "SETEX成功 key=[" .. key .. "] expire=" .. expire_time)
    self:return_connection(red)

    return true
end

-- GET 命令
function _M.get(self, key)
    local red, err = self:get_connection(key)
    if not red then
        return nil, err
    end

    local value, err = red:get(key)
    if err then
        ngx.log(ngx.ERR, "GET失败 key=[" .. key .. "]: " .. err)
        self:close_connection(red)
        return nil, "failed to get: " .. err
    end

    ngx.log(ngx.INFO, "GET成功 key=[" .. key .. "] value=[" .. tostring(value) .. "]")
    self:return_connection(red)

    return value
end

-- INCR 命令
function _M.incr(self, key)
    local red, err = self:get_connection(key)
    if not red then
        return nil, err
    end

    local result, err = red:incr(key)
    if not result then
        ngx.log(ngx.ERR, "INCR失败 key=[" .. key .. "]: " .. err)
        self:close_connection(red)
        return nil, "failed to incr: " .. err
    end

    ngx.log(ngx.INFO, "INCR成功 key=[" .. key .. "] result=" .. result)
    self:return_connection(red)

    return result
end

-- DECR 命令
function _M.decr(self, key)
    local red, err = self:get_connection(key)
    if not red then
        return nil, err
    end

    local result, err = red:decr(key)
    if not result then
        ngx.log(ngx.ERR, "DECR失败 key=[" .. key .. "]: " .. err)
        self:close_connection(red)
        return nil, "failed to decr: " .. err
    end

    ngx.log(ngx.INFO, "DECR成功 key=[" .. key .. "] result=" .. result)
    self:return_connection(red)

    return result
end

-- EXPIRE 命令
function _M.expire(self, key, expire_time)
    local red, err = self:get_connection(key)
    if not red then
        ngx.log(ngx.ERR, "获取连接失败: " .. err)
        return nil, err
    end

    local ok, err = red:expire(key, tonumber(expire_time))
    if not ok then
        ngx.log(ngx.ERR, "EXPIRE失败 key=[" .. key .. "]: " .. err)
        self:close_connection(red)
        return nil, "EXPIRE failed: " .. err
    end

    ngx.log(ngx.INFO, "EXPIRE成功 key=[" .. key .. "] expire=" .. expire_time .. "秒")
    self:return_connection(red)

    return true
end

-- DEL 命令
function _M.del(self, key)
    local red, err = self:get_connection(key)
    if not red then
        return nil, err
    end

    local result, err = red:del(key)
    if not result then
        ngx.log(ngx.ERR, "DEL失败 key=[" .. key .. "]: " .. err)
        self:close_connection(red)
        return nil, "failed to del: " .. err
    end

    ngx.log(ngx.INFO, "DEL成功 key=[" .. key .. "]")
    self:return_connection(red)

    return result
end

-- EXISTS 命令
function _M.exists(self, key)
    local red, err = self:get_connection(key)
    if not red then
        ngx.log(ngx.ERR, "获取连接失败: " .. err)
        return nil, err
    end

    local exists, err = red:exists(key)
    if err then
        ngx.log(ngx.ERR, "EXISTS失败 key=[" .. key .. "]: " .. err)
        self:close_connection(red)
        return nil, "EXISTS failed: " .. err
    end

    ngx.log(ngx.INFO, "EXISTS成功 key=[" .. key .. "] result=" .. exists)
    self:return_connection(red)

    return exists
end

-- TTL 命令
function _M.ttl(self, key)
    local red, err = self:get_connection(key)
    if not red then
        return nil, err
    end

    local ttl, err = red:ttl(key)
    if err then
        ngx.log(ngx.ERR, "TTL失败 key=[" .. key .. "]: " .. err)
        self:close_connection(red)
        return nil, "TTL failed: " .. err
    end

    ngx.log(ngx.INFO, "TTL成功 key=[" .. key .. "] ttl=" .. ttl)
    self:return_connection(red)

    return ttl
end

-- SISMEMBER 命令
function _M.sismember(self, key, member)
    local red, err = self:get_connection(key)
    if not red then
        ngx.log(ngx.ERR, "获取连接失败: " .. err)
        return nil, err
    end

    local result, err = red:sismember(key, member)
    if err then
        ngx.log(ngx.ERR, "SISMEMBER失败 key=[" .. key .. "] member=[" .. tostring(member) .. "]: " .. err)
        self:close_connection(red)
        return nil, "SISMEMBER failed: " .. err
    end

    ngx.log(ngx.INFO, "SISMEMBER成功 key=[" .. key .. "] member=[" .. tostring(member) .. "] result=" .. result)
    self:return_connection(red)

    return result
end

-- SADD 命令
function _M.sadd(self, key, ...)
    local red, err = self:get_connection(key)
    if not red then
        return nil, err
    end

    local result, err = red:sadd(key, ...)
    if not result then
        ngx.log(ngx.ERR, "SADD失败 key=[" .. key .. "]: " .. err)
        self:close_connection(red)
        return nil, "SADD failed: " .. err
    end

    ngx.log(ngx.INFO, "SADD成功 key=[" .. key .. "] result=" .. result)
    self:return_connection(red)

    return result
end

-- SREM 命令
function _M.srem(self, key, ...)
    local red, err = self:get_connection(key)
    if not red then
        return nil, err
    end

    local result, err = red:srem(key, ...)
    if not result then
        ngx.log(ngx.ERR, "SREM失败 key=[" .. key .. "]: " .. err)
        self:close_connection(red)
        return nil, "SREM failed: " .. err
    end

    ngx.log(ngx.INFO, "SREM成功 key=[" .. key .. "] result=" .. result)
    self:return_connection(red)

    return result
end

-- SMEMBERS 命令
function _M.smembers(self, key)
    local red, err = self:get_connection(key)
    if not red then
        return nil, err
    end

    local result, err = red:smembers(key)
    if err then
        ngx.log(ngx.ERR, "SMEMBERS失败 key=[" .. key .. "]: " .. err)
        self:close_connection(red)
        return nil, "SMEMBERS failed: " .. err
    end

    ngx.log(ngx.INFO, "SMEMBERS成功 key=[" .. key .. "]")
    self:return_connection(red)

    return result
end

-- HSET 命令
function _M.hset(self, key, field, value)
    local red, err = self:get_connection(key)
    if not red then
        ngx.log(ngx.ERR, "获取连接失败: " .. err)
        return nil, err
    end

    local result, err = red:hset(key, field, value)
    if not result then
        ngx.log(ngx.ERR, "HSET失败 key=[" .. key .. "] field=[" .. tostring(field) .. "]: " .. err)
        self:close_connection(red)
        return nil, "HSET failed: " .. err
    end

    ngx.log(ngx.INFO, "HSET成功 key=[" .. key .. "] field=[" .. tostring(field) .. "]")
    self:return_connection(red)

    return result
end

-- HMSET 命令
function _M.hmset(self, key, ...)
    local red, err = self:get_connection(key)
    if not red then
        return nil, err
    end

    local result, err = red:hmset(key, ...)
    if not result then
        ngx.log(ngx.ERR, "HMSET失败 key=[" .. key .. "]: " .. err)
        self:close_connection(red)
        return nil, "HMSET failed: " .. err
    end

    ngx.log(ngx.INFO, "HMSET成功 key=[" .. key .. "]")
    self:return_connection(red)

    return result
end

-- HGET 命令
function _M.hget(self, key, field)
    local red, err = self:get_connection(key)
    if not red then
        ngx.log(ngx.ERR, "获取连接失败 key=[" .. key .. "]: " .. err)
        return nil, err
    end

    local value, err = red:hget(key, field)
    if err then
        ngx.log(ngx.ERR, "HGET失败 key=[" .. key .. "] field=[" .. tostring(field) .. "]: " .. err)
        self:close_connection(red)
        return nil, "HGET failed: " .. err
    end

    ngx.log(ngx.INFO, "HGET成功 key=[" .. key .. "] field=[" .. tostring(field) .. "] value=[" .. tostring(value) .. "]")
    self:return_connection(red)

    return value
end

-- HMGET 命令
function _M.hmget(self, key, ...)
    local red, err = self:get_connection(key)
    if not red then
        return nil, err
    end

    local values, err = red:hmget(key, ...)
    if err then
        ngx.log(ngx.ERR, "HMGET失败 key=[" .. key .. "]: " .. err)
        self:close_connection(red)
        return nil, "HMGET failed: " .. err
    end

    ngx.log(ngx.INFO, "HMGET成功 key=[" .. key .. "]")
    self:return_connection(red)

    return values
end

-- HGETALL 命令
function _M.hgetall(self, key)
    local red, err = self:get_connection(key)
    if not red then
        return nil, err
    end

    local result, err = red:hgetall(key)
    if err then
        ngx.log(ngx.ERR, "HGETALL失败 key=[" .. key .. "]: " .. err)
        self:close_connection(red)
        return nil, "HGETALL failed: " .. err
    end

    ngx.log(ngx.INFO, "HGETALL成功 key=[" .. key .. "]")
    self:return_connection(red)

    return result
end

-- HDEL 命令
function _M.hdel(self, key, ...)
    local red, err = self:get_connection(key)
    if not red then
        return nil, err
    end

    local result, err = red:hdel(key, ...)
    if not result then
        ngx.log(ngx.ERR, "HDEL失败 key=[" .. key .. "]: " .. err)
        self:close_connection(red)
        return nil, "HDEL failed: " .. err
    end

    ngx.log(ngx.INFO, "HDEL成功 key=[" .. key .. "]")
    self:return_connection(red)

    return result
end

-- HEXISTS 命令
function _M.hexists(self, key, field)
    local red, err = self:get_connection(key)
    if not red then
        return nil, err
    end

    local result, err = red:hexists(key, field)
    if err then
        ngx.log(ngx.ERR, "HEXISTS失败 key=[" .. key .. "] field=[" .. tostring(field) .. "]: " .. err)
        self:close_connection(red)
        return nil, "HEXISTS failed: " .. err
    end

    ngx.log(ngx.INFO, "HEXISTS成功 key=[" .. key .. "] field=[" .. tostring(field) .. "] result=" .. result)
    self:return_connection(red)

    return result
end

-- HINCRBY 命令
function _M.hincrby(self, key, field, increment)
    local red, err = self:get_connection(key)
    if not red then
        return nil, err
    end

    local result, err = red:hincrby(key, field, tonumber(increment))
    if not result then
        ngx.log(ngx.ERR, "HINCRBY失败 key=[" .. key .. "] field=[" .. tostring(field) .. "]: " .. err)
        self:close_connection(red)
        return nil, "HINCRBY failed: " .. err
    end

    ngx.log(ngx.INFO, "HINCRBY成功 key=[" .. key .. "] field=[" .. tostring(field) .. "] result=" .. result)
    self:return_connection(red)

    return result
end

-- LPUSH 命令
function _M.lpush(self, key, ...)
    local red, err = self:get_connection(key)
    if not red then
        return nil, err
    end

    local result, err = red:lpush(key, ...)
    if not result then
        ngx.log(ngx.ERR, "LPUSH失败 key=[" .. key .. "]: " .. err)
        self:close_connection(red)
        return nil, "LPUSH failed: " .. err
    end

    ngx.log(ngx.INFO, "LPUSH成功 key=[" .. key .. "] result=" .. result)
    self:return_connection(red)

    return result
end

-- RPUSH 命令
function _M.rpush(self, key, ...)
    local red, err = self:get_connection(key)
    if not red then
        return nil, err
    end

    local result, err = red:rpush(key, ...)
    if not result then
        ngx.log(ngx.ERR, "RPUSH失败 key=[" .. key .. "]: " .. err)
        self:close_connection(red)
        return nil, "RPUSH failed: " .. err
    end

    ngx.log(ngx.INFO, "RPUSH成功 key=[" .. key .. "] result=" .. result)
    self:return_connection(red)

    return result
end

-- LPOP 命令
function _M.lpop(self, key)
    local red, err = self:get_connection(key)
    if not red then
        return nil, err
    end

    local result, err = red:lpop(key)
    if err then
        ngx.log(ngx.ERR, "LPOP失败 key=[" .. key .. "]: " .. err)
        self:close_connection(red)
        return nil, "LPOP failed: " .. err
    end

    ngx.log(ngx.INFO, "LPOP成功 key=[" .. key .. "]")
    self:return_connection(red)

    return result
end

-- RPOP 命令
function _M.rpop(self, key)
    local red, err = self:get_connection(key)
    if not red then
        return nil, err
    end

    local result, err = red:rpop(key)
    if err then
        ngx.log(ngx.ERR, "RPOP失败 key=[" .. key .. "]: " .. err)
        self:close_connection(red)
        return nil, "RPOP failed: " .. err
    end

    ngx.log(ngx.INFO, "RPOP成功 key=[" .. key .. "]")
    self:return_connection(red)

    return result
end

-- LRANGE 命令
function _M.lrange(self, key, start, stop)
    local red, err = self:get_connection(key)
    if not red then
        return nil, err
    end

    local result, err = red:lrange(key, start, stop)
    if err then
        ngx.log(ngx.ERR, "LRANGE失败 key=[" .. key .. "]: " .. err)
        self:close_connection(red)
        return nil, "LRANGE failed: " .. err
    end

    ngx.log(ngx.INFO, "LRANGE成功 key=[" .. key .. "]")
    self:return_connection(red)

    return result
end

-- LLEN 命令
function _M.llen(self, key)
    local red, err = self:get_connection(key)
    if not red then
        return nil, err
    end

    local result, err = red:llen(key)
    if err then
        ngx.log(ngx.ERR, "LLEN失败 key=[" .. key .. "]: " .. err)
        self:close_connection(red)
        return nil, "LLEN failed: " .. err
    end

    ngx.log(ngx.INFO, "LLEN成功 key=[" .. key .. "] len=" .. result)
    self:return_connection(red)

    return result
end

-- 执行任意 Redis 命令（高级用法）
function _M.execute(self, key, cmd, ...)
    local red, err = self:get_connection(key)
    if not red then
        return nil, err
    end

    local method = red[cmd]
    if not method then
        self:close_connection(red)
        return nil, "unknown command: " .. cmd
    end

    local result, err = method(red, ...)
    if err then
        ngx.log(ngx.ERR, cmd .. "失败: " .. err)
        self:close_connection(red)
        return nil, cmd .. " failed: " .. err
    end

    self:return_connection(red)

    return result
end

return _M
