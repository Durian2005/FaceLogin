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

#: 允许的本机主机名。本服务只监听回环地址，所以白名单固定为这几种写法。
#: 注意 Origin 解析出来的主机名**不带方括号**（`urlsplit` 会剥掉），
#: 而原始 Host 头里是带方括号的 `[::1]` —— 两种形态都要能对上。
LOOPBACK_NAMES = ("127.0.0.1", "localhost", "::1", "[::1]")


def _extra_hosts():
    """`FACELOGIN_EXTRA_HOSTS` 里手工追加的条目（逗号分隔，精确匹配）。"""
    out = set()
    for item in os.environ.get("FACELOGIN_EXTRA_HOSTS", "").split(","):
        item = item.strip().lower()
        if item:
            out.add(item)
    return out


def allowed_hosts(port):
    """构造白名单。返回值同时供 Host 校验与 Origin 校验使用，但两者用法不同：

    - `host_allowed` **只看主机名**，不看端口（原因见那里）；
    - `origin_allowed` 要求 `host:port` **精确匹配**，端口在这里是有意义的。

    同时收 `host:port` 与 `host:80` 两种形式：前者对应服务自己的地址，
    后者对应通过 80 端口访问的场景（正常不会有，但不必人为制造意外拒绝）。
    """
    hosts = set()
    for name in LOOPBACK_NAMES:
        hosts.add(name)
        hosts.add("%s:%d" % (name, port))
        hosts.add("%s:80" % name)
    hosts |= _extra_hosts()
    return frozenset(hosts)


def _split_host_port(value):
    """把 Host 头拆成 (主机名, 端口)。拆不出来时返回 ("", None)。

    用 `urlsplit` 而不是手工按冒号切：手工切会被
    `127.0.0.1:8080.evil.com` 这种「主机名对、尾巴是垃圾」的写法骗过去，
    而 `urlsplit` 在端口不是纯数字时会直接报错。
    """
    if "@" in value:            # Host 里出现 userinfo 不是合法写法，直接否掉
        return "", None
    try:
        parts = urlsplit("//" + value)
        return (parts.hostname or "").lower(), parts.port
    except ValueError:
        return "", None


def host_allowed(host_header, allowed):
    """Host 头是否指向本机回环名字。

    ⚠️ **端口不参与判定**，这是刻意的。理由有两条：

    1. 端口在防 DNS 重绑定里**不是承重点**。重绑定之后浏览器发的 Host 是
       攻击者域名（`evil.com:5000` 这样），把它拦下靠的是「域名不是本机」，
       端口对不对无关紧要。
    2. 反过来，如果这里要求端口精确匹配，服务一旦绑到与预期不同的端口
       （改了 `FACELOGIN_PORT`、或在测试里绑随机端口），就会把自己**全部
       403 掉**。这个自伤故障是真机探测测出来的，不是推想出来的。

    "攻击者页面直连 127.0.0.1" 那条路本来就**不是** Host 校验在管，
    而是由 `origin_allowed` 管（它要求端口精确匹配）。

    空 Host 一律拒绝（HTTP/1.0 客户端不带 Host）。
    """
    if not host_header:
        return False
    h = host_header.strip().lower()
    if h in allowed:            # 兼容 FACELOGIN_EXTRA_HOSTS 的精确写法
        return True
    name, _port = _split_host_port(h)
    if not name:
        return False
    return name in LOOPBACK_NAMES or name in allowed


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
