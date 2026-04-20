"""
helix_server.py — Flask bridge for the Helix GUI
Run with:  python helix_server.py
The GUI connects to http://localhost:5000/chat
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import logging

# ── Import your newly fixed Helix agent ──────────────────────────
# Make sure this file sits in the same folder as helix_agent.py
from helix_agent import helix_agent, memory_collection, _conversation_history, unload_all

# Suppress verbose Flask logging so your terminal stays clean for the Agent's thoughts
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

app = Flask(__name__)
CORS(app)  # allow the browser GUI to call localhost

# Add send_file to your imports
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import logging
import os

# ... (existing setup code) ...

# ADD THIS NEW ROUTE:
@app.route("/image", methods=["GET"])
def get_image():
    """Serves the generated image file to the frontend."""
    image_path = "helix_creation.png"
    if os.path.exists(image_path):
        return send_file(image_path, mimetype='image/png')
    return "Image not found", 404

# ... (existing /chat route) ...

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(force=True)
    message = (data.get("message") or "").strip()

    if not message:
        return jsonify({"response": "Empty message."}), 400

    try:
        # Helix cleanly returns the final string natively!
        final_response = helix_agent(message)

        return jsonify({
            "response": final_response,
            "mem_count": memory_collection.count() if memory_collection else 0,
        })
    except Exception as e:
        return jsonify({"response": f"[System Error]: {str(e)}"}), 500


@app.route("/status", methods=["GET"])
def status():
    return jsonify({
        "status": "online",
        "mem_count": memory_collection.count() if memory_collection else 0,
        "history_turns": len(_conversation_history)
    })


if __name__ == "__main__":
    print("\n[Helix]: Starting standalone Web Server on http://localhost:5000")

    # Optional: ensure VRAM is totally clear before we start taking requests
    try:
        unload_all()
    except Exception:
        pass

    app.run(host="0.0.0.0", port=5000, debug=False, threaded=False)