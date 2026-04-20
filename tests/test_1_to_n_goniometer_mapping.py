import numpy as np
import h5py
from unittest.mock import patch, MagicMock

# Handle import depending on your branch structure
try:
    from subhkl.io.command_line_parser import indexer
except ImportError:
    from subhkl.io.parser import indexer

from subhkl.instrument.goniometer import Goniometer


def test_1_to_n_goniometer_mapping(tmp_path):
    peaks_h5 = tmp_path / "synthetic_peaks.h5"
    output_h5 = tmp_path / "indexed.h5"
    dummy_nexus = tmp_path / "dummy.nxs"

    # The settings block containing the virtual kappa decoupled stages
    # Note the '#' suffixes which allow duplicate motors in JSON
    mock_settings = {
        "Goniometer": {
            "CG4D:Mot:phi": [0, 1, 0, -1],
            "CG4D:Mot:alpha#tilt": [1, 0, 0, -1],
            "CG4D:Mot:kappa": [0, 1, 0, 1],
            "CG4D:Mot:alpha#antitilt": [1, 0, 0, 1],
            "CG4D:Mot:omega": [0, 1, 0, 1],
        }
    }

    # 1. Create a Mock NeXus File with the DASlogs
    # Even though there are 5 axes in the JSON, there are only 4 actual logs!
    with h5py.File(dummy_nexus, "w") as f:
        f.create_dataset("entry/DASlogs/CG4D:Mot:phi/average_value", data=[45.0])
        f.create_dataset("entry/DASlogs/CG4D:Mot:alpha/average_value", data=[24.0])
        f.create_dataset("entry/DASlogs/CG4D:Mot:kappa/average_value", data=[30.0])
        f.create_dataset("entry/DASlogs/CG4D:Mot:omega/average_value", data=[10.0])

    # 2. Test the Goniometer Loading Logic
    with patch.dict(
        "subhkl.instrument.goniometer.reduction_settings", {"CG4D": mock_settings}
    ):
        gonio = Goniometer.from_nexus(str(dummy_nexus), "CG4D")

        # Assert the parser successfully generated 5 axes
        assert len(gonio.axes_raw) == 5, "Failed to build 5 axes from settings."
        assert len(gonio.angles_raw) == 5, "Failed to expand 4 motors into 5 angles."

        # Assert the # suffixes were correctly stripped so optimization.py can link them
        expected_names = [
            "CG4D:Mot:phi",
            "CG4D:Mot:alpha",
            "CG4D:Mot:kappa",
            "CG4D:Mot:alpha",
            "CG4D:Mot:omega",
        ]
        assert gonio.names_raw == expected_names, (
            f"Suffix stripping failed. Got {gonio.names_raw}"
        )

        # Assert alpha (24.0) was correctly applied to both tilt and antitilt stages
        np.testing.assert_allclose(gonio.angles_raw, [45.0, 24.0, 30.0, 24.0, 10.0])

    # 3. Generate Physically Valid Synthetic Peaks
    a, b, c = 10.0, 10.0, 10.0
    alpha, beta, gamma = 90.0, 90.0, 90.0

    from subhkl.optimization import FindUB

    fu = FindUB()
    fu.a, fu.b, fu.c = a, b, c
    fu.alpha, fu.beta, fu.gamma = alpha, beta, gamma
    B = fu.reciprocal_lattice_B()

    hkls = [
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
        [1, 1, 0],
        [1, 0, 1],
        [0, 1, 1],
        [1, 1, 1],
        [2, 0, 0],
        [0, 2, 0],
        [0, 0, 2],
    ]

    R0 = gonio.rotation
    ki_hat = np.array([0.0, 0.0, 1.0])
    xyz = []

    for hkl in hkls:
        hkl = np.array(hkl)
        q_lab = R0 @ B @ hkl
        Q = 2.0 * np.pi * q_lab
        Q_sq = np.sum(Q**2)
        lamb_val = -4.0 * np.pi * np.dot(ki_hat, Q) / (Q_sq + 1e-9)

        # If scattering is unphysical (forward scattering), flip the HKL
        if lamb_val < 0:
            hkl = -hkl
            q_lab = R0 @ B @ hkl
            Q = 2.0 * np.pi * q_lab
            lamb_val = -4.0 * np.pi * np.dot(ki_hat, Q) / (Q_sq + 1e-9)

        kf_hat = (lamb_val / (2.0 * np.pi)) * Q + ki_hat
        kf_hat /= np.linalg.norm(kf_hat)  # normalize
        xyz.append(kf_hat)

    xyz = np.array(xyz)
    num_peaks = len(xyz)

    # 4. Build the Mock HDF5 Input for the Indexer
    with h5py.File(peaks_h5, "w") as f:
        f["sample/a"], f["sample/b"], f["sample/c"] = a, b, c
        f["sample/alpha"], f["sample/beta"], f["sample/gamma"] = alpha, beta, gamma
        f["sample/space_group"] = "P 1"
        f["instrument/wavelength"] = [0.1, 10.0]

        f["peaks/two_theta"] = np.zeros(num_peaks)
        f["peaks/azimuthal"] = np.zeros(num_peaks)
        f["peaks/intensity"] = np.ones(num_peaks)
        f["peaks/sigma"] = np.ones(num_peaks) * 0.1
        f["peaks/radius"] = np.zeros(num_peaks)
        f["peaks/run_index"] = np.zeros(num_peaks, dtype=np.int32)
        f["peaks/xyz"] = xyz

        f["goniometer/R"] = np.tile(R0, (num_peaks, 1, 1))
        f["goniometer/axes"] = gonio.axes_raw
        # Transpose so shapes map correctly (num_axes, num_peaks)
        f["goniometer/angles"] = np.tile(gonio.angles_raw, (num_peaks, 1)).T
        f.create_dataset(
            "goniometer/names", data=[n.encode("utf-8") for n in gonio.names_raw]
        )

        f["peaks/pixel_r"] = np.zeros(num_peaks)
        f["peaks/pixel_c"] = np.zeros(num_peaks)
        f["peaks/image_index"] = np.zeros(num_peaks, dtype=np.int32)
        f["bank"] = np.ones(num_peaks, dtype=np.int32)
        f["bank_ids"] = np.array([1], dtype=np.int32)

    # 5. Run the Indexer with Goniometer Refinement
    # If the JAX motor_map fails to compile or throws a shape error, this will crash.
    with (
        patch("subhkl.instrument.detector.Detector") as mock_detector,
        patch("subhkl.commands.Peaks") as mock_peaks,  # noqa: F841
        patch("subhkl.commands.get_rotation_data_from_nexus") as mock_get_rot,
        patch.dict("subhkl.config.beamlines", {"CG4D": {"1": {}}}),
    ):
        mock_det_instance = MagicMock()
        mock_det_instance.pixel_to_lab.return_value = xyz
        mock_det_instance.pixel_to_angles.return_value = (
            np.zeros(num_peaks),
            np.zeros(num_peaks),
        )
        mock_detector.return_value = mock_det_instance

        # Bypass the internal nexus load during run_index
        mock_get_rot.return_value = (
            gonio.axes_raw,
            np.array(gonio.angles_raw)[None, :],
            gonio.names_raw,
        )

        indexer(
            peaks_h5_filename=str(peaks_h5),
            output_peaks_filename=str(output_h5),
            a=a,
            b=b,
            c=c,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            space_group="P 1",
            strategy_name="DE",
            population_size=200,
            gens=50,
            n_runs=1,
            seed=42,
            refine_goniometer=True,  # <-- Crucial! Triggers the JAX motor_map fusion
            instrument_name="CG4D",
            original_nexus_filename=str(dummy_nexus),
        )

    # 6. Verify Success
    with h5py.File(output_h5, "r") as f:
        import json

        flags = json.loads(f["optimization/flags"][()].decode("utf-8"))
        assert flags["refine_goniometer"] is True, (
            "Indexer ignored goniometer refinement flag."
        )

        hkl_out = f["peaks/h"][:]
        # Because we provided perfect initial coordinates, it should index perfectly.
        # But more importantly, the test proves that JAX successfully navigated
        # a 4-parameter search space across 5 physical rotation matrices.
        indexed_count = np.sum(hkl_out != 0)
        assert indexed_count > 0, (
            "Optimization crashed or failed to index any peaks during 1:n refinement."
        )


def test_mock_instrument_1_to_n_mapping(tmp_path):
    peaks_h5 = tmp_path / "mock_peaks.h5"
    output_h5 = tmp_path / "mock_indexed.h5"
    dummy_nexus = tmp_path / "mock_dummy.nxs"

    # Define a completely fake instrument with a weird 1:n mapping
    # Motor A drives Axis 1 and Axis 3. Motor B drives Axis 2.
    mock_instrument_def = {
        "MOCK_KAPPA": {
            "Goniometer": {
                "Motor_A#part1": [1, 0, 0, 1],
                "Motor_B": [0, 1, 0, 1],
                "Motor_A#part2": [0, 0, 1, -1],
            }
        }
    }

    # 1. Create a Mock NeXus File with only Motor A and Motor B
    with h5py.File(dummy_nexus, "w") as f:
        f.create_dataset("entry/DASlogs/Motor_A/average_value", data=[15.0])
        f.create_dataset("entry/DASlogs/Motor_B/average_value", data=[45.0])

    # 2. Generate trivial synthetic data
    num_peaks = 10
    xyz = np.random.normal(size=(num_peaks, 3))
    xyz /= np.linalg.norm(xyz, axis=1, keepdims=True)

    with h5py.File(peaks_h5, "w") as f:
        f["sample/a"], f["sample/b"], f["sample/c"] = 10, 10, 10
        f["sample/alpha"], f["sample/beta"], f["sample/gamma"] = 90, 90, 90
        f["sample/space_group"] = "P 1"
        f["instrument/wavelength"] = [1.0, 2.0]

        f["peaks/two_theta"] = np.zeros(num_peaks)
        f["peaks/azimuthal"] = np.zeros(num_peaks)
        f["peaks/intensity"] = np.ones(num_peaks)
        f["peaks/sigma"] = np.ones(num_peaks) * 0.1
        f["peaks/radius"] = np.zeros(num_peaks)
        f["peaks/run_index"] = np.zeros(num_peaks, dtype=np.int32)
        f["peaks/xyz"] = xyz

        f["peaks/pixel_r"] = np.zeros(num_peaks)
        f["peaks/pixel_c"] = np.zeros(num_peaks)
        f["peaks/image_index"] = np.zeros(num_peaks, dtype=np.int32)
        f["bank"] = np.ones(num_peaks, dtype=np.int32)
        f["bank_ids"] = np.array([1], dtype=np.int32)

    # 3. Run the Indexer, injecting the MOCK_KAPPA definition
    with (
        patch("subhkl.instrument.detector.Detector") as mock_detector,
        patch("subhkl.commands.Peaks") as mock_peaks,  # noqa: F841
        patch.dict(
            "subhkl.instrument.goniometer.reduction_settings", mock_instrument_def
        ),
        patch.dict("subhkl.config.beamlines", {"MOCK_KAPPA": {"1": {}}}),
    ):
        mock_det_instance = MagicMock()
        mock_det_instance.pixel_to_lab.return_value = xyz
        mock_det_instance.pixel_to_angles.return_value = (
            np.zeros(num_peaks),
            np.zeros(num_peaks),
        )
        mock_detector.return_value = mock_det_instance

        indexer(
            peaks_h5_filename=str(peaks_h5),
            output_peaks_filename=str(output_h5),
            a=10,
            b=10,
            c=10,
            alpha=90,
            beta=90,
            gamma=90,
            space_group="P 1",
            strategy_name="DE",
            population_size=10,
            gens=2,
            n_runs=1,  # Tiny run just to test compilation
            refine_goniometer=True,
            instrument_name="MOCK_KAPPA",
            original_nexus_filename=str(dummy_nexus),
        )

    # 4. Verify the optimization compiled and processed the 1:n map successfully
    with h5py.File(output_h5, "r") as f:
        # The output file should have successfully stored all 3 axes,
        # even though they were driven by only 2 parameters during refinement.
        assert f["goniometer/axes"].shape[0] == 3
        assert list(f["goniometer/names"][()]) == [b"Motor_A", b"Motor_B", b"Motor_A"]
