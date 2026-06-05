import json
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError


def test_unload(host: str = "127.0.0.1", port: int = 8015) -> None:
    url = f"http://{host}:{port}/unload"
    req = Request(url, method="GET")
    try:
        with urlopen(req, timeout=10) as resp:
            body = resp.read().decode("utf-8")
            print(f"HTTP {resp.status} {resp.reason}")
            try:
                data = json.loads(body)
                print(json.dumps(data, indent=2, ensure_ascii=False))
            except json.JSONDecodeError:
                print(body)
    except HTTPError as exc:
        print(f"HTTP error: {exc.code} {exc.reason}")
        print(exc.read().decode("utf-8", errors="ignore"))
    except URLError as exc:
        print(f"URL error: {exc}")


if __name__ == "__main__":
    test_unload()
