import json
import os
import sys
import time
from urllib.parse import urlencode
from urllib.request import urlopen, Request

HOST = "127.0.0.1"
PORT = 7860


def main():
    prompt_wav = os.path.abspath(os.path.join("ref_audio", "1.wav"))
    if not os.path.isfile(prompt_wav):
        print(f"prompt_wav file not found: {prompt_wav}")
        return 1

    text = "你好，欢迎使用IndexTTS，这是 HTTP 测试。"
    save_path = os.path.abspath(os.path.join("ref_output", f"http_test_{int(time.time())}.wav"))
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    params = {
        "prompt_wav": prompt_wav,
        "text": text,
        "save_path": save_path,
    }
    query = urlencode(params)
    url = f"http://{HOST}:{PORT}/tts?{query}"

    print(f"Sending request to {url}")
    req = Request(url, method="GET")
    try:
        with urlopen(req, timeout=600) as resp:
            body = resp.read().decode("utf-8")
            data = json.loads(body)
            print("Response:")
            print(json.dumps(data, ensure_ascii=False, indent=2))

            if resp.status != 200:
                print(f"HTTP server error status: {resp.status}")
                return 2

            output_path = data.get("output_path")
            if not output_path:
                print("Server response did not return output_path.")
                return 3

            if not os.path.isfile(output_path):
                print(f"Expected output file not found: {output_path}")
                return 4

            print(f"Test passed. Generated audio saved to: {output_path}")
            return 0
    except Exception as exc:
        print(f"Request failed: {exc}")
        return 5


if __name__ == "__main__":
    sys.exit(main())
