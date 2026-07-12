#!/bin/bash
# 双击启动：本地静态服务 + 自动打开浏览器
# 纯前端工具，所有计算在浏览器里跑，这个脚本只负责把文件用 http:// 发出来
cd "$(dirname "$0")" || exit 1

PORT=8126

# 8126 可能被别的本地项目占用。只有确认页面确实是本工具才直接复用；
# 否则向后寻找空闲端口，避免打开一个完全无关的网站。
while lsof -nP -iTCP:$PORT -sTCP:LISTEN >/dev/null 2>&1; do
  if curl -fsS --max-time 1 "http://127.0.0.1:$PORT/" 2>/dev/null | grep -q "聖地巡礼"; then
    URL="http://localhost:$PORT/"
    echo "🎨 圣地巡礼调色工具"
    echo "服务已在运行，直接打开浏览器：$URL"
    open "$URL"
    exit 0
  fi
  PORT=$((PORT + 1))
  if [ "$PORT" -gt 8136 ]; then
    echo "未找到可用端口（8126–8136），请先关闭部分本地服务。"
    read -r -p "按回车关闭…"
    exit 1
  fi
done

URL="http://localhost:$PORT/"

# 找一个可用的 python3
PY="$(command -v python3 || echo /opt/homebrew/bin/python3)"

echo "🎨 圣地巡礼调色工具"
echo "目录: $(pwd)"
echo "地址: $URL"
echo "（关掉这个终端窗口即停止服务）"
echo ""

# 启动后稍等再开浏览器
( sleep 1; open "$URL" ) &
# 带 COOP/COEP 头的静态服务：开启 cross-origin isolation 后
# onnxruntime-web 才能用多线程 WASM（AI 抠图约快 3-4 倍）
exec "$PY" - "$PORT" <<'PYEOF'
import sys
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler

class Handler(SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header('Cross-Origin-Opener-Policy', 'same-origin')
        self.send_header('Cross-Origin-Embedder-Policy', 'require-corp')
        self.send_header('Cross-Origin-Resource-Policy', 'cross-origin')
        super().end_headers()

ThreadingHTTPServer(('127.0.0.1', int(sys.argv[1])), Handler).serve_forever()
PYEOF
