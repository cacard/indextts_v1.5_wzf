import json
import os
import traceback
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import parse_qs, urlparse

from demo import gen_single

HOST = "0.0.0.0"
PORT = 8015


class TTSRequestHandler(BaseHTTPRequestHandler):
    def _send_json(self, status_code, payload):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path != "/tts":
            self._send_json(404, {"error": "Not Found", "message": "Use /tts endpoint."})
            return

        params = parse_qs(parsed.query)
        prompt_wav = params.get("prompt_wav", [None])[0]
        text = params.get("text", [None])[0]
        save_path = params.get("save_path", [None])[0]

        if not prompt_wav:
            self._send_json(400, {"error": "missing_parameter", "message": "prompt_wav is required."})
            return

        if not text:
            self._send_json(400, {"error": "missing_parameter", "message": "text is required."})
            return

        prompt_wav = os.path.expanduser(prompt_wav)
        if not os.path.isabs(prompt_wav):
            prompt_wav = os.path.abspath(prompt_wav)

        if not os.path.isfile(prompt_wav):
            self._send_json(400, {
                "error": "invalid_parameter",
                "message": f"prompt_wav file does not exist: {prompt_wav}",
            })
            return

        if save_path:
            save_path = os.path.expanduser(save_path)
            if not os.path.isabs(save_path):
                save_path = os.path.abspath(save_path)

        try:
            output_path = gen_single(prompt_wav, text, save_path=save_path)
            self._send_json(200, {
                "status": "success",
                "output_path": output_path,
            })
        except Exception as exc:
            self._send_json(500, {
                "error": "internal_error",
                "message": str(exc),
                "traceback": traceback.format_exc(),
            })

    def log_message(self, format, *args):
        return


def run_server(host=HOST, port=PORT):
    server = HTTPServer((host, port), TTSRequestHandler)
    print(f"Starting HTTP server on http://{host}:{port}/tts")
    print("Use query parameters: prompt_wav, text, save_path(optional)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("Shutting down server...")
    finally:
        server.server_close()


if __name__ == "__main__":
    run_server()
