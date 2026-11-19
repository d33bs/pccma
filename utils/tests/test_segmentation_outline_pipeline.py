"""Run the real notebook scripts on explicitly synthetic crops-index and outline-TIFF inputs.

Styled like test_crop_pipeline.py: a synthetic "atlas" root (its own .git, so get_root_dir finds
it) holds a tiny crops_index.parquet plus matching outline TIFFs, and the real nbconverted scripts
are run as subprocesses against it, never against the real atlas.
"""

import os
import pathlib
import shutil
import subprocess
import sys

import duckdb
import nbformat
import numpy as np
import pandas as pd
import pytest
import tifffile
from nbconvert import PythonExporter
from skimage.draw import circle_perimeter

REPO = pathlib.Path(__file__).resolve().parents[2]
MODULE = "5b.segmentation_outlines"
NOTEBOOKS = ("0.generate_segmentation_outlines", "1.build_segmentation_index")
FINAL_KEY = [
    "Metadata_Plate", "Metadata_Well", "Metadata_Site", "Metadata_ImageNumber",
    "Metadata_Nuclei_Number_Object_Number", "Metadata_Cells_Number_Object_Number",
]
GEOMETRY_COLUMNS = [
    "Metadata_Nucleocentric_Crop_Minimum_Y", "Metadata_Nucleocentric_Crop_Maximum_Y",
    "Metadata_Nucleocentric_Crop_Minimum_X", "Metadata_Nucleocentric_Crop_Maximum_X",
]


def _outline_image(size, circles):
    """A green-on-black (0, 32896, 0) 16-bit outline TIFF, the real OverlayOutlines convention."""
    image = np.zeros((size, size, 3), dtype=np.uint16)
    for center, radius in circles:
        rows, cols = circle_perimeter(*center, radius, shape=(size, size))
        image[rows, cols] = (0, 32896, 0)
    return image


@pytest.fixture
def atlas(tmp_path):
    """Three synthetic cells: one closed outline, one open at the field's own edge, one with no
    source TIFF at all (a whole missing field)."""
    root = tmp_path / "synthetic atlas"
    root.mkdir()
    (root / ".git").mkdir()
    script_dir = root / MODULE / "nbconverted"
    script_dir.mkdir(parents=True)
    for name in NOTEBOOKS:
        shutil.copyfile(REPO / MODULE / f"{name}.ipynb", script_dir.parent / f"{name}.ipynb")
        notebook = nbformat.read(REPO / MODULE / f"{name}.ipynb", as_version=4)
        script, _ = PythonExporter().from_notebook_node(notebook)
        (script_dir / f"{name}.py").write_text(script)

    from nucleocentric_crops import get_nucleocentric_crop_box

    field_size = 256
    outlines_dir = root / "outlines"
    field_dir = outlines_dir / "BR00150698_A673"
    field_dir.mkdir(parents=True)
    # Cell 1's nucleus/cell outline is a closed circle well inside its own 224x224 crop -> "ok".
    # Cell 2's outline is centered near the field's own corner; its own crop extends past the field
    # edge, so the part of the ring inside the crop never closes -> "not_closed_at_center".
    nuclei_mask = _outline_image(field_size, [((128, 128), 50), ((8, 8), 60)])
    cells_mask = _outline_image(field_size, [((128, 128), 55), ((8, 8), 65)])
    tifffile.imwrite(field_dir / "NucleiOutlines_BR00150698_C05_2.tiff", nuclei_mask)
    tifffile.imwrite(field_dir / "CellsOutlines_BR00150698_C05_2.tiff", cells_mask)
    # Cell 3 (well C05, site 3) has no outline TIFFs anywhere -> "missing_source" for both
    # compartments, not a crash and not a silently dropped row.

    rows = []
    cells = [
        ("C05", 2, 7, (128, 128)),  # obj 1: closed outline
        ("C05", 2, 7, (8, 8)),      # obj 2: outline open at the field/crop edge
        ("C05", 3, 9, (128, 128)),  # obj 3: whole field has no outline TIFFs
    ]
    for obj, (well, site, image_number, center) in enumerate(cells, start=1):
        box = get_nucleocentric_crop_box(center[0], center[1])
        row = {
            "Metadata_Plate": "BR00150698", "Metadata_Well": well, "Metadata_Site": site,
            "Metadata_ImageNumber": image_number,
            "Metadata_Nuclei_Number_Object_Number": obj, "Metadata_Cells_Number_Object_Number": obj,
            "Metadata_cell_line": "A673", "Metadata_time_point": 24,
        }
        row.update(dict(zip(GEOMETRY_COLUMNS, box)))
        rows.append(row)
    crops_dir = root / "crops"
    crops_dir.mkdir()
    crops_index_path = crops_dir / "crops_index.parquet"
    pd.DataFrame(rows).to_parquet(crops_index_path, index=False)
    return root, outlines_dir, crops_index_path


def run_stage(root, outlines_dir, crops_index_path, number, *args, ok=True):
    env = dict(
        os.environ, PYTHONPATH=str(REPO / "utils"),
        PCCMA_OUTLINES_DIR=str(outlines_dir), PCCMA_CROPS_INDEX=str(crops_index_path),
    )
    result = subprocess.run(
        [sys.executable, str(root / MODULE / "nbconverted" / f"{NOTEBOOKS[number]}.py"), *args],
        cwd=root, env=env, capture_output=True, text=True, timeout=60,
    )
    output = result.stdout + result.stderr
    if ok:
        assert result.returncode == 0, output
    else:
        assert result.returncode != 0, output
    return output


def generate(root, outlines_dir, crops_index_path, *args, ok=True):
    return run_stage(root, outlines_dir, crops_index_path, 0, "--plate", "BR00150698", *args, ok=ok)


def build_index(root, outlines_dir, crops_index_path, *args, ok=True):
    return run_stage(root, outlines_dir, crops_index_path, 1, *args, ok=ok)


def warehouse(root):
    return root / MODULE / "data/segmentation_outlines"


def test_end_to_end_statuses_and_duckdb_readback(atlas):
    root, outlines_dir, crops_index_path = atlas
    generate(root, outlines_dir, crops_index_path)
    output = build_index(root, outlines_dir, crops_index_path)

    # Read the published index back with DuckDB only, not pandas/pyarrow directly: this pytest
    # session also imports geoarrow.pyarrow (test_segmentation_outlines.py), and once that has
    # happened, reading a file with one of its extension columns via pyarrow/pandas in the same
    # process can hang forever at interpreter shutdown (see segmentation_outlines.py's CAUTION).
    index_path = warehouse(root) / "segmentation_index.parquet"
    connection = duckdb.connect()
    row_count = connection.execute(f"SELECT count(*) FROM read_parquet('{index_path.as_posix()}')").fetchone()[0]
    assert row_count == 3
    rows = connection.execute(f"SELECT * FROM read_parquet('{index_path.as_posix()}')").fetchall()
    columns = [description[0] for description in connection.description]
    by_obj = {row[columns.index("Metadata_Nuclei_Number_Object_Number")]: dict(zip(columns, row)) for row in rows}
    assert len(by_obj) == 3

    # Cell 1: a genuinely closed outline gives an "ok" row with real X/Y/Geometry.
    assert by_obj[1]["Nuclei_Segmentation_Boundary_Status"] == "ok"
    assert by_obj[1]["Nuclei_Segmentation_Boundary_X"] is not None
    assert by_obj[1]["Nuclei_Segmentation_Boundary_Y"] is not None
    assert by_obj[1]["Nuclei_Segmentation_Boundary_Geometry"] is not None
    assert by_obj[1]["Nuclei_Segmentation_Boundary_VertexCount"] >= 4
    assert by_obj[1]["Cells_Segmentation_Boundary_Status"] == "ok"

    # Cell 2: an outline that cannot close at the crop's own center is reported, not dropped.
    assert by_obj[2]["Nuclei_Segmentation_Boundary_Status"] == "not_closed_at_center"
    assert by_obj[2]["Nuclei_Segmentation_Boundary_X"] is None
    assert by_obj[2]["Nuclei_Segmentation_Boundary_Geometry"] is None
    assert by_obj[2]["Nuclei_Segmentation_Boundary_VertexCount"] is None

    # Cell 3: a whole missing field is "missing_source" for every cell, for both compartments.
    assert by_obj[3]["Nuclei_Segmentation_Boundary_Status"] == "missing_source"
    assert by_obj[3]["Cells_Segmentation_Boundary_Status"] == "missing_source"
    assert by_obj[3]["Nuclei_Segmentation_Boundary_Source_TIFF"] is None
    assert by_obj[3]["Nuclei_Segmentation_Boundary_Geometry"] is None

    assert "DuckDB read-back" in output
    assert "spot check passed" in output


def test_index_build_rejects_key_set_mismatch(atlas):
    root, outlines_dir, crops_index_path = atlas
    generate(root, outlines_dir, crops_index_path)
    crops_index = pd.read_parquet(crops_index_path)
    extra_row = crops_index.iloc[[0]].copy()
    extra_row["Metadata_Nuclei_Number_Object_Number"] = 99
    extra_row["Metadata_Cells_Number_Object_Number"] = 99
    pd.concat([crops_index, extra_row], ignore_index=True).to_parquet(crops_index_path, index=False)
    output = build_index(root, outlines_dir, crops_index_path, ok=False)
    assert "does not exactly match" in output
    assert not (warehouse(root) / "segmentation_index.parquet").is_file()
