import json
import os

ASSESSMENT_FILE = "assessments.json"


def load_assessments():
    """Load assessments from JSON file."""
    if not os.path.exists(ASSESSMENT_FILE):
        with open(ASSESSMENT_FILE, "w") as f:
            json.dump({}, f)
        return {}

    with open(ASSESSMENT_FILE, "r") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return {}  # corrupted file fallback


def save_assessments(data):
    """Save entire assessment dictionary back to file."""
    with open(ASSESSMENT_FILE, "w") as f:
        json.dump(data, f, indent=4)
