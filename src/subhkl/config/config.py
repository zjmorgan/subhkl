import json
import os

THIS_DIR = os.path.dirname(os.path.abspath(__file__))

# Beamline / instrument definitions
BEAMLINES_JSON = "beamlines.json"
with open(os.path.join(THIS_DIR, BEAMLINES_JSON), 'r') as f:
    beamlines = json.load(f)

# Reduction settigns / DAS mappigs
SETTINGS_JSON = "reduction_settings.json"
with open(os.path.join(THIS_DIR, SETTINGS_JSON), 'r') as f:
    reduction_settings = json.load(f)