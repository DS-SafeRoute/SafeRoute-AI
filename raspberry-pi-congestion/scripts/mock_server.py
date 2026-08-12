"""Development-only receiver. It is not evidence that the final Spring API exists."""
from flask import Flask, jsonify, request

app = Flask(__name__)


@app.post("/api/v1/device/congestion-observations")
def receive_observation():
    print("[OBSERVATION]", request.get_json())
    return jsonify({"accepted": True}), 202


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8080)
