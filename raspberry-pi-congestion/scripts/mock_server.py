"""Development-only receiver for the final Pi/BE contract."""
import time
from uuid import uuid4

from flask import Flask, jsonify, request

app = Flask(__name__)
SESSION_ID = "550e8400-e29b-41d4-a716-446655440000"


@app.get("/api/v1/device/congestion-config")
def config():
    code = request.args.get("cctvCode", "CCTV_001")
    return jsonify({
        "trainingActive": True, "trainingSessionId": SESSION_ID, "cctvCode": code,
        "monitoredAreaM2": 2.0, "configVersion": 1, "snapshotIntervalSec": 5,
        "targetInferenceFps": 5,
        "congestionThresholds": {"CAUTION_FROM": 2.0, "CROWDED_FROM": 3.0, "VERY_CROWDED_FROM": 5.0},
        "eventDetection": {"requiredConsecutiveFrames": 3, "recoveryConsecutiveFrames": 5, "cooldownSec": 30},
    })


@app.post("/api/v1/device/congestion-images/presigned-url")
def presigned():
    body = request.get_json()
    key = f"training/{body['trainingSessionId']}/{body['imageType'].lower()}/{body['cctvCode']}/{body['referenceId']}.jpg"
    return jsonify({"objectKey": key, "uploadUrl": f"http://127.0.0.1:8080/upload/{uuid4()}", "expiresAt": int(time.time() * 1000) + 60000})


@app.put("/upload/<upload_id>")
def upload(upload_id):
    return "", 200


@app.post("/api/v1/device/congestion-observations")
def observation():
    print("[OBSERVATION]", request.get_json())
    return jsonify({"accepted": True}), 201


@app.post("/api/v1/device/congestion-events")
def event():
    print("[EVENT]", request.get_json())
    return jsonify({"accepted": True}), 201


@app.patch("/api/v1/device/congestion-events/<event_id>/image")
def event_image(event_id):
    print("[EVENT IMAGE]", event_id, request.get_json())
    return jsonify({"accepted": True}), 200


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8080)
