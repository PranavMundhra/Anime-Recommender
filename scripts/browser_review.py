"""Capture the real local UI using installed Edge and its debugging protocol.

Run only after starting Streamlit and approving a browser review. No personal
browser profile is used, and no AI or catalogue search is submitted.
"""
import base64
from contextlib import ExitStack
import json
from pathlib import Path
import subprocess
import time
from urllib.request import urlopen

from websockets.sync.client import connect

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "docs" / "screenshots"
EDGE = Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe")


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    profile = ROOT / ".cache" / "browser-review"
    profile.mkdir(parents=True, exist_ok=True)
    process = subprocess.Popen([
        str(EDGE), "--headless=new", "--disable-gpu", "--no-first-run",
        "--no-default-browser-check", "--remote-debugging-port=9222",
        "--remote-debugging-address=127.0.0.1", f"--user-data-dir={profile}",
        "--window-size=1440,1000", "about:blank",
    ], creationflags=subprocess.CREATE_NO_WINDOW)
    socket = None
    connections = ExitStack()
    counter = 0
    events = []
    try:
        deadline = time.monotonic() + 15
        while True:
            try:
                with urlopen("http://127.0.0.1:9222/json", timeout=2) as response:
                    pages = json.load(response)
                page = next(item for item in pages if item["type"] == "page")
                break
            except Exception:
                if time.monotonic() > deadline:
                    raise RuntimeError("The isolated browser did not start.")
                time.sleep(.25)
        socket = connections.enter_context(connect(page["webSocketDebuggerUrl"], open_timeout=5))

        def command(method, params=None):
            nonlocal counter
            counter += 1
            current = counter
            socket.send(json.dumps({"id": current, "method": method, "params": params or {}}))
            while True:
                result = json.loads(socket.recv(timeout=20))
                if result.get("id") == current:
                    if "error" in result:
                        raise RuntimeError(result["error"])
                    return result.get("result", {})
                if result.get("method") in ("Runtime.exceptionThrown", "Log.entryAdded"):
                    events.append(result)

        def evaluate(expression):
            result = command("Runtime.evaluate", {"expression": expression, "returnByValue": True})
            if result.get("exceptionDetails"):
                raise RuntimeError("Browser evaluation failed.")
            return result.get("result", {}).get("value")

        def wait_for(expression):
            deadline = time.monotonic() + 25
            while not evaluate(expression):
                if time.monotonic() > deadline:
                    raise RuntimeError("The expected app view did not render.")
                time.sleep(.25)

        def screenshot(name):
            # Visible viewport, captured from the actual browser renderer.
            result = command("Page.captureScreenshot", {"format": "png", "captureBeyondViewport": False})
            (OUTPUT / name).write_bytes(base64.b64decode(result["data"]))

        def scroll_top():
            evaluate("document.querySelectorAll('*').forEach(e => { if(e.scrollTop) e.scrollTop = 0; }); window.scrollTo(0,0)")
            time.sleep(.25)

        command("Page.enable")
        command("Runtime.enable")
        command("Log.enable")
        command("Emulation.setDeviceMetricsOverride", {"width": 1440, "height": 1000, "deviceScaleFactor": 1, "mobile": False})
        command("Page.navigate", {"url": "http://127.0.0.1:8501"})
        wait_for("document.body.innerText.includes('find a story for how you feel')")
        wait_for("Array.from(document.querySelectorAll('button')).some(b => b.textContent.includes('Choose my own filters'))")
        screenshot("desktop-startup.png")
        startup = evaluate("document.body.innerText")
        evaluate("Array.from(document.querySelectorAll('button')).find(b => b.textContent.includes('Choose my own filters')).click()")
        wait_for("document.body.innerText.includes('Pick your own filters') && document.querySelector('textarea') !== null")
        command("Emulation.setDeviceMetricsOverride", {"width": 1440, "height": 1400, "deviceScaleFactor": 1, "mobile": False})
        scroll_top()
        screenshot("desktop-profile.png")
        profile_text = evaluate("document.body.innerText")
        command("Emulation.setDeviceMetricsOverride", {"width": 390, "height": 844, "deviceScaleFactor": 1, "mobile": True})
        time.sleep(.5)
        scroll_top()
        screenshot("mobile-profile.png")
        overflow = evaluate("document.documentElement.scrollWidth > window.innerWidth + 2")
        review = {"url": "http://127.0.0.1:8501", "desktop_startup": {"width": 1440, "height": 1000},
                  "desktop_profile": {"width": 1440, "height": 1400},
                  "mobile": {"width": 390, "height": 844, "horizontal_overflow": overflow},
                  "startup_rendered": "find a story for how you feel" in startup,
                  "profile_rendered": "Pick your own filters" in profile_text,
                  "app_exception_rendered": "Traceback" in profile_text,
                  "browser_runtime_exception_count": sum(e.get("method") == "Runtime.exceptionThrown" for e in events),
                  "screenshots": ["desktop-startup.png", "desktop-profile.png", "mobile-profile.png"],
                  "scope": "Real local browser; no AI conversation or catalogue query submitted."}
        (OUTPUT / "browser-review.json").write_text(json.dumps(review, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(review, indent=2))
    finally:
        connections.close()
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()


if __name__ == "__main__":
    main()
