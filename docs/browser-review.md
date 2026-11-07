# Local browser review

Reviewed 14 October 2026 using the installed Microsoft Edge headless browser and the running Streamlit server at `http://127.0.0.1:8501`.

The actual startup screen and manual profile editor rendered successfully. There were no JavaScript runtime exceptions or rendered application tracebacks during these interactions. The 390-pixel mobile view had no horizontal page overflow. Desktop profile fields and the submit button were visually inspected in the captured image.

| Capture | Viewport | File |
| --- | --- | --- |
| Desktop startup | 1440 × 1000 | [Screenshot](screenshots/desktop-startup.png) |
| Desktop profile editor | 1440 × 1400 | [Screenshot](screenshots/desktop-profile.png) |
| Mobile profile editor | 390 × 844 | [Screenshot](screenshots/mobile-profile.png) |

The [machine-readable browser report](screenshots/browser-review.json) records the checks. `scripts/browser_review.py` reproduces them after a local server has been started and browser execution approved. It uses a separate ignored browser profile, closes that browser after capture, and leaves Streamlit running for the owner.

These are real browser screenshots. No AI message or Jikan search was submitted in this review. They do not establish provider latency, successful catalogue connectivity, streamed-message timing, or live recommendation quality. Conversation/recommendation behavior remains covered by the documented mocked tests; authenticated and public-network end-to-end checks remain outstanding.
