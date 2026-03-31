import json
from importlib import resources

resource_path = resources.files("subhkl.resources")

# Beamline / instrument definitions
with (resource_path / "beamlines.json").open() as f:
    beamlines = json.load(f)

# Reduction settigns / DAS mappings
with (resource_path / "reduction_settings.json").open() as f:
    reduction_settings = json.load(f)
