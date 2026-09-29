# -*- coding: utf-8 -*-
"""最小 OpenAI 兼容 mock 服务：把每行原文回显为「【译】原文」，用于验证翻译链路
（不改任何东西，只用来测 onsei2lrc.py 的批处理/解析/LRC 输出是否正确）

用法：
    python tests/mock_openai_server.py 8123
然后在另一个终端：
    python onsei2lrc.py test_long.wav --retranslate --base-url http://127.0.0.1:8123/v1 \
        --model-name mock --lrc-mode both
"""
import json
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer


class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        req = json.loads(self.rfile.read(n) or b"{}")
        msgs = req.get("messages", [])
        user = next((m["content"] for m in reversed(msgs) if m.get("role") == "user"), "")
        system = next((m["content"] for m in msgs if m.get("role") == "system"), "")

        if "JSON" in system or "json" in user[:80]:          # json 协议
            body = user.split("：\n", 1)[-1]
            try:
                items = json.loads(body)
                out = json.dumps([{"i": it["i"], "zh": "【译】" + it["ja"]} for it in items],
                                 ensure_ascii=False)
            except Exception:
                out = "[]"
        else:                                                 # sakura 行对齐协议
            lines = user.split("：", 1)[-1].strip("\n").splitlines()
            out = "\n".join("【译】" + l for l in lines if l.strip())

        resp = {"id": "mock", "object": "chat.completion", "model": req.get("model", "mock"),
                "choices": [{"index": 0, "finish_reason": "stop",
                             "message": {"role": "assistant", "content": out}}],
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}}
        data = json.dumps(resp).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *a):
        pass


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8123
    print(f"mock OpenAI server on http://127.0.0.1:{port}/v1", flush=True)
    HTTPServer(("127.0.0.1", port), Handler).serve_forever()
