import hashlib
import json
import threading
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from baselines.adaptok import download_checkpoint as downloader


@pytest.fixture
def hub_server(monkeypatch):
    content = b"transport test object, never loaded as a neural checkpoint"
    state = {"wrong_range": False}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            if "/api/" in self.path:
                payload = json.dumps({"sha": "test-revision", "siblings": [{"rfilename": "model.safetensors", "size": len(content),
                                      "lfs": {"sha256": hashlib.sha256(content).hexdigest()}}]}).encode()
                self.send_response(200)
            elif "config.json" in self.path:
                payload = b"{}"
                self.send_response(200)
            else:
                start, end = map(int, self.headers["Range"].removeprefix("bytes=").split("-"))
                payload = content[start:end + 1]
                self.send_response(206)
                reported_start = start + 1 if state["wrong_range"] else start
                self.send_header("Content-Range", f"bytes {reported_start}-{end}/{len(content)}")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    opener = urllib.request.urlopen

    def redirect(request, timeout=30):
        url = request.full_url if isinstance(request, urllib.request.Request) else request
        parsed = urllib.parse.urlsplit(url)
        headers = dict(request.header_items()) if isinstance(request, urllib.request.Request) else {}
        local_url = f"http://127.0.0.1:{server.server_port}{parsed.path}?{parsed.query}"
        return opener(urllib.request.Request(local_url, headers=headers), timeout=timeout)

    monkeypatch.setattr(downloader.urllib.request, "urlopen", redirect)
    monkeypatch.setattr(downloader.time, "sleep", lambda _: None)
    yield content, state
    server.shutdown()
    worker.join()
    server.server_close()


def test_download_assembles_verifies_and_reuses_complete_object(hub_server, tmp_path, monkeypatch):
    content, _ = hub_server
    monkeypatch.setattr("sys.argv", ["download_checkpoint", "--output", str(tmp_path)])
    downloader.main()
    assert (tmp_path / "model.safetensors").read_bytes() == content
    assert not (tmp_path / ".download_parts").exists()
    downloader.main()
    assert (tmp_path / "model.safetensors").read_bytes() == content


def test_wrong_ranges_are_rejected_and_cached_parts_are_reused(hub_server, tmp_path):
    content, state = hub_server
    state["wrong_range"] = True
    with pytest.raises(ValueError, match="exact byte range"):
        downloader.download_part(tmp_path, "https://example/model.safetensors", 0, 9)
    assert not list(tmp_path.iterdir())
    target = tmp_path / "part_000000000000"
    target.write_bytes(content[:10])
    assert downloader.download_part(tmp_path, "https://example/model.safetensors", 0, 9) == target
