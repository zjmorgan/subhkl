from enum import IntEnum, auto


# Rank symmetries: Lower number = Lower Symmetry (More free params)
class LatticeSystem(IntEnum):
    TRICLINIC = 0
    MONOCLINIC = auto()
    ORTHORHOMBIC = auto()
    RHOMBOHEDRAL = auto()
    TETRAGONAL = auto()
    HEXAGONAL = auto()
    CUBIC = auto()


class _Params(IntEnum):
    # Indices for readability
    A = 0
    B = auto()
    C = auto()
    ALPHA = auto()
    BETA = auto()
    GAMMA = auto()


SG_SYSTEM_MAP = {
    "cubic": LatticeSystem.CUBIC,
    "hexagonal": LatticeSystem.HEXAGONAL,
    "tetragonal": LatticeSystem.TETRAGONAL,
    "orthorhombic": LatticeSystem.ORTHORHOMBIC,
    "monoclinic": LatticeSystem.MONOCLINIC,
    "triclinic": LatticeSystem.TRICLINIC,
}


# Data-Oriented Physics Constraints
LATTICE_CONSTRAINTS = {
    LatticeSystem.CUBIC: {
        "equal_lengths": [(_Params.A, _Params.B), (_Params.B, _Params.C)],
        "fixed_angles": {_Params.ALPHA: 90.0, _Params.BETA: 90.0, _Params.GAMMA: 90.0},
    },
    LatticeSystem.HEXAGONAL: {
        "equal_lengths": [(_Params.A, _Params.B)],
        "fixed_angles": {_Params.ALPHA: 90.0, _Params.BETA: 90.0, _Params.GAMMA: 120.0},
    },
    LatticeSystem.TETRAGONAL: {
        "equal_lengths": [(_Params.A, _Params.B)],
        "fixed_angles": {_Params.ALPHA: 90.0, _Params.BETA: 90.0, _Params.GAMMA: 90.0},
    },
    LatticeSystem.RHOMBOHEDRAL: {
        "equal_lengths": [(_Params.A, _Params.B), (_Params.B, _Params.C)],
        "equal_angles": [(_Params.ALPHA, _Params.BETA), (_Params.BETA, _Params.GAMMA)],
    },
    LatticeSystem.ORTHORHOMBIC: {
        "fixed_angles": {_Params.ALPHA: 90.0, _Params.BETA: 90.0, _Params.GAMMA: 90.0}
    },
    LatticeSystem.MONOCLINIC: {
        # Standard b-unique setting: alpha=gamma=90
        "fixed_angles": {_Params.ALPHA: 90.0, _Params.GAMMA: 90.0}
    },
    LatticeSystem.TRICLINIC: {},
}


def _get_col(p, idx):
    return p[..., idx]


LATTICE_CONFIG = {
    LatticeSystem.CUBIC: {
        "name": "Cubic",
        "num_params": 1,
        "active_indices": [0],
        "reconstruct": lambda p: (
            _get_col(p, 0),
            _get_col(p, 0),
            _get_col(p, 0),
            90.0,
            90.0,
            90.0,
        ),
    },
    LatticeSystem.HEXAGONAL: {
        "name": "Hexagonal",
        "num_params": 2,
        "active_indices": [0, 2],
        "reconstruct": lambda p: (
            _get_col(p, 0),
            _get_col(p, 0),
            _get_col(p, 1),
            90.0,
            90.0,
            120.0,
        ),
    },
    LatticeSystem.TETRAGONAL: {
        "name": "Tetragonal",
        "num_params": 2,
        "active_indices": [0, 2],
        "reconstruct": lambda p: (
            _get_col(p, 0),
            _get_col(p, 0),
            _get_col(p, 1),
            90.0,
            90.0,
            90.0,
        ),
    },
    LatticeSystem.RHOMBOHEDRAL: {
        "name": "Rhombohedral",
        "num_params": 2,
        "active_indices": [0, 3],  # Assuming params are [a, alpha]
        "reconstruct": lambda p: (
            _get_col(p, 0),
            _get_col(p, 0),
            _get_col(p, 0),
            _get_col(p, 1),
            _get_col(p, 1),
            _get_col(p, 1),
        ),
    },
    LatticeSystem.ORTHORHOMBIC: {
        "name": "Orthorhombic",
        "num_params": 3,
        "active_indices": [0, 1, 2],
        "reconstruct": lambda p: (
            _get_col(p, 0),
            _get_col(p, 1),
            _get_col(p, 2),
            90.0,
            90.0,
            90.0,
        ),
    },
    LatticeSystem.MONOCLINIC: {
        "name": "Monoclinic",
        "num_params": 4,
        "active_indices": [0, 1, 2, 4],  # a, b, c, beta
        "reconstruct": lambda p: (
            _get_col(p, 0),
            _get_col(p, 1),
            _get_col(p, 2),
            90.0,
            _get_col(p, 3),
            90.0,
        ),
    },
    LatticeSystem.TRICLINIC: {
        "name": "Triclinic",
        "num_params": 6,
        "active_indices": [0, 1, 2, 3, 4, 5],
        "reconstruct": lambda p: tuple(_get_col(p, i) for i in range(6)),
    },
}
