"""校验审计日志的哈希链是否完好。**只读**，不会修改任何数据。

    .venv\\Scripts\\python.exe tools\\verify_audit_chain.py

退出码 0 = 链条完好；1 = 检测到篡改或断链。

它回答的问题是：「这张审计表有没有被人事后改过？」
每条记录都带着前一条的哈希，改动任何一条，从它往后的所有 entry_hash 都会对不上。
注意边界：哈希链只能证明"被改过"，不能阻止改动 —— 拿到库文件的人可以把整条链
重算一遍。要防住那种情况，得把链头锚定到库外（见 THREAT_MODEL.md）。
"""

import os
import sys

APP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "app")
sys.path.insert(0, os.path.abspath(APP_DIR))

import db  # noqa: E402

print("数据库:", db.DB_PATH)
stats = db.event_stats()
print("审计事件: %d 条（其中失败 %d 条）" % (stats["total"], stats["failed"]))
print()

ok, detail = db.verify_audit_chain()

if ok:
    print("链条状态: 完好")
    print("已校验  : %d 条" % detail["checked"])
    print("链头哈希: " + detail["head"])
    sys.exit(0)

print("链条状态: 已被破坏")
print("断点 id : %d" % detail["broken_at"])
print("原因    : " + detail["reason"])
print("已校验  : %d 条（断点之前的记录都还对得上）" % detail["checked"])
print("应为    : %s ..." % detail["expected"])
print("实为    : %s ..." % detail["found"])
sys.exit(1)
