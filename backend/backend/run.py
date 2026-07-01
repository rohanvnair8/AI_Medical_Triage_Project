"""
run.py — start the triage app and open the browser at the same time.

Usage:
    python run.py
"""

import threading
import time
import webbrowser
import sys
import os

HOST = "0.0.0.0"
PORT = int(os.getenv("PORT", "5001"))
URL  = f"http://{HOST}:{PORT}"


def open_browser():
    """Wait for Flask to be ready, then open the browser."""
    # Poll until the server is accepting connections
    import socket
    for _ in range(30):          # try for up to 6 seconds
        try:
            s = socket.create_connection((HOST, PORT), timeout=0.2)
            s.close()
            break
        except OSError:
            time.sleep(0.2)

    webbrowser.open(URL)
    print(f"\n✅  Browser opened at {URL}\n")


def main():
    # Make sure we can find app.py next to this file
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    # Import the Flask app (this also calls initialize_database)
    from app import app, load_patients

    load_patients()

    # Launch browser in a background thread so it doesn't block Flask
    threading.Thread(target=open_browser, daemon=True).start()

    print(f"🚀  Starting triage server on {URL} ...")
    print("    Press Ctrl+C to stop.\n")

    # Run Flask (use_reloader=False keeps the single-process model clean)
    app.run(
        host=HOST,
        port=PORT,
        debug=False,        # set True if you want auto-reload during dev
        use_reloader=False,
    )


if __name__ == "__main__":
    main()