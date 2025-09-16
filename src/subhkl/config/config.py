import json
import os


THIS_DIR = os.path.dirname(os.path.absolute(__file__))
CONFIG_JSON = "config.json"

beamlines = json.loads(os.path.join(THIS_DIR, CONFIG_JSON))