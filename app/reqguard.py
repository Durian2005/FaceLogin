"""请求来源校验：Host 白名单、Origin/Referer 校验、每进程随机页面令牌。

为什么本地服务也要防这个
------------------------
本地服务最现实的攻击面不是"别人连进来"，而是**你自己的浏览器**。你在上网时
随手打开的一个恶意页面，可以在你毫不知情的情况下向 127.0.0.1:5000 发请求。
两类具体手法：

1. **跨站请求伪造（CSRF）** —— 页面里放一个自动提交的 `<form method=POST>`。
   CORS 只挡"读取响应"，不挡"发出请求"，所以副作用照样发生。而
   `/api/shutdown`、`/api/bye` 这类接口根本不读 Cookie，`SameSite` 拦不住它们。
   后果：任意网页一个表单就能关掉你的服务。

2. **DNS rebinding** —— 恶意域名先解析到自己的服务器骗取信任，随后把 TTL 改短、
   重新解析到 127.0.0.1。在浏览器看来请求是"同源"的，同源策略因此失效，
   页面里的 JS 能直接读走 `/api/stream` 的摄像头画面。

对应三道闸门
------------
- **Host 头白名单**：掐断 DNS rebinding —— rebinding 请求的 Host 是攻击者域名。
- **Origin / Referer 校验**：掐断 CSRF —— 跨站请求一定带着别人的 Origin。
- **每进程随机 page token**：只有本站页面拿得到，用来保护 `/api/stream` 与
  `/api/bye` 这两个"无法要求登录态"的接口（页面在登录前就要显示摄像头画面）。

设计上的两个取舍
----------------
- Origin 与 Referer **都缺失时放行**。因为 curl / 测试脚本本来就没有这两个头，
  而 CSRF 必须借浏览器之手，浏览器发跨站 POST 时一定会带 Origin。
- `Origin: null`（sandbox iframe、file:// 页面）**一律拒绝** —— 那是真实的攻击面，
  不是正常客户端。
"""

import hmac
import os
import secrets
from urllib.parse import urlsplit

#: 允许的本机主机名。本服务只监听回环地址，所以白名单固定为这三种写法。
LOOPBACK_NAMES = ("127.0.0.1", "localhost", "::1")


def allowed_hosts(port):
    """构造 Host / Origin 白名单。

    同时收 "host:port" 与 "host:80" 两种形式：前者对应服务自己的地址，
    后者对应通过 80 端口访问的场景（正常不会有，但不必人为制造意外拒绝）。
    """
    hosts = set()
    for name in LOOPBACK_NAMES:
        hosts.add(name)
        hosts.add("%s:%d" % (name, port))
        hosts.add("%s:80" % name)
    hosts.add("[::1]")
    hosts.add("[::1]:%d" % port)
    hosts.add("[::1]:80")

    extra = os.environ.get("FACELOGIN_EXTRA_HOSTS", "")
    for item in extra.split(","):
        item = item.strip().lower()
        if item:
            hosts.add(item)
    return frozenset(hosts)


def host_allowed(host_header, allowed):
    """Host 头是否在白名单内。空 Host 一律拒绝（HTTP/1.0 客户端不带 Host）。"""
    if not host_header:
        return False
    return host_header.strip().lower() in allowed


def _split_origin(value):
    """把 Origin/Referer 拆成 (主机名, 端口)；解析不出来返回 (None, None)。"""
    try:
        parts = urlsplit(value)
    except ValueError:
        return None, None
    if not parts.scheme or not parts.hostname:
        return None, None
    try:
        port = parts.port
    except ValueError:
        return None, None
    if port is None:
        port = 443 if parts.scheme == "https" else 80
    return parts.hostname.lower(), port


def origin_allowed(origin, referer, allowed):
    """校验请求来源。

    规则：Origin 与 Referer 都缺 => 放行（非浏览器客户端）；
    只要其中一个存在，且指向的 host:port 不在白名单内 => 拒绝。
    """
    for raw in (origin, referer):
        if not raw:
            continue
        if raw.strip().lower() == "null":
            return False
        name, port = _split_origin(raw)
        if name is None:
            return False
        if "%s:%d" % (name, port) not in allowed:
            return False
    return True


def new_page_token():
    """每次进程启动重新生成 —— 重启即失效，不给令牌跨运行复用的机会。"""
    return secrets.token_urlsafe(24)


def token_matches(supplied, expected):
    """常数时间比较，避免用响应时间把令牌一位一位试出来。"""
    if not supplied or not expected:
        return False
    return hmac.compare_digest(str(supplied), str(expected))
