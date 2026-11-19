"""Extract per-object segmentation boundaries from CellProfiler's outline overlays.

CellProfiler's OverlayOutlines module already writes one outline image per field of view
(`<Compartment>Outlines_<plate>_<well>_<site>.tiff`, used for visual QC by
`4.preprocess_features/1.sc_quality_control.ipynb`), but it burns every object's outline into the
same image without per-object identity. This module re-derives one closed polygon per cell from
that raster: it crops the outline image with the same nucleocentric crop box as the JPG warehouse
(`crop_warehouse`/`nucleocentric_crops`), fills the interior of the outline that encloses the
crop's own center pixel, and traces its boundary, so the polygon lands in the same crop-relative
pixel space as the composite JPGs.

We do not depend on https://github.com/WayScience/ome-arrow (an early-stage project without a
release) to decide the export shape. Instead we use the same scikit-image contour tracing already
used elsewhere for object boundaries, and encode each polygon two ways in Parquet/DuckDB, neither
of which needs a database extension to read:
  - plain `list<double>` x/y coordinate columns (crop-relative pixels, origin top-left), readable
    by pandas/pyarrow/DuckDB with no spatial dependency at all;
  - a native GeoArrow polygon column (nested `list<rings: list<vertices: struct<x, y>>>`, tagged
    with the `geoarrow.polygon` Arrow extension type), built with `geoarrow-pyarrow`
    (https://github.com/geoarrow/geoarrow-python, a stable, released, non-domain-specific
    implementation of the GeoArrow spec) rather than hand-rolled WKB blobs or ome-arrow itself.
    `polygon_to_wkb` still exists and is used internally to validate each ring before handing it to
    geoarrow-pyarrow, and remains available directly for anything that specifically wants WKB.
A per-cell status column records why a polygon is missing (open outline, too few vertices, etc.)
instead of silently dropping the cell.

CAUTION (found and reproduced locally, geoarrow-pyarrow 0.3.0 / geoarrow-c 0.4.0, macOS/CPython
3.11): once `geoarrow.pyarrow` has been imported in a process (directly, or transitively through
`polygons_to_geoarrow`), later reading a Parquet file that contains one of its extension columns
with `pyarrow.parquet.read_table` or `pandas.read_parquet` *in that same process* can hang forever
at interpreter shutdown, even though the read itself returns the correct data immediately.
Writing such a column is unaffected, and DuckDB's own native Parquet reader is unaffected (it
never goes through pyarrow's registered-extension-type deserialization). Read a published
GeoArrow-bearing Parquet file back with DuckDB (`duckdb.connect().execute(...).fetchall()` or
`.df()`), not with `pyarrow`/`pandas` directly, in any process that also calls
`polygons_to_geoarrow` or otherwise imports `geoarrow.pyarrow`.
"""

import numpy as np
import tifffile
from scipy import ndimage
from skimage.measure import approximate_polygon, find_contours, label

from nucleocentric_crops import crop_with_padding

# the two compartments CellProfiler's OverlayOutlines writes for this pipeline (IdentifyPrimaryObjects
# for Nuclei, IdentifySecondaryObjects for Cells); matches Metadata_Nuclei/Cells_Number_Object_Number
COMPARTMENTS = ["Nuclei", "Cells"]

# why extract_object_polygon did not return a polygon
STATUS_OK = "ok"
STATUS_NOT_CLOSED = "not_closed_at_center"
STATUS_TOO_FEW_VERTICES = "too_few_vertices"
STATUS_CONTOUR_FAILED = "contour_extraction_failed"
STATUS_MISSING_SOURCE = "missing_source"
STATUS_OPEN_RING = "open_ring_at_crop_edge"


def outline_filename(compartment: str, plate: str, well: str, site: int) -> str:
    """Build the file name of an OverlayOutlines TIFF, the same pattern as 1.sc_quality_control.

    Parameters
    ----------
    compartment : str
        "Nuclei" or "Cells".
    plate, well, site : str, str, int
        Plate barcode, well position, and field-of-view number (Image_Metadata_Site).

    Returns
    -------
    str
        For example "NucleiOutlines_BR00150695_D04_1.tiff".
    """
    return f"{compartment}Outlines_{plate}_{well}_{site}.tiff"


def load_outline_mask(path) -> np.ndarray:
    """Load an OverlayOutlines TIFF and collapse it to a 2D boolean outline-pixel mask.

    OverlayOutlines writes a solid color (observed: pure green, (0, 32896, 0) in 16-bit) on a
    black background with no per-object identity; any nonzero pixel in any channel is an outline
    pixel of some object.

    Parameters
    ----------
    path : str | pathlib.Path
        Path to the outline TIFF.

    Returns
    -------
    np.ndarray
        2D boolean array, True where any channel of the source image is nonzero.
    """
    image = tifffile.imread(path)
    if image.ndim == 2:
        return image > 0
    if image.ndim == 3:
        return np.any(image > 0, axis=-1)
    raise ValueError(f"Unexpected outline image shape {image.shape} in {path}")


def polygon_area_px2(x: np.ndarray, y: np.ndarray) -> float:
    """Shoelace-formula area of a closed polygon, in pixels^2 (no shapely needed).

    Parameters
    ----------
    x, y : np.ndarray
        Vertex coordinates (crop-relative pixels), first vertex repeated as the last.

    Returns
    -------
    float
        Unsigned area of the polygon.
    """
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    return abs(np.sum(x[:-1] * y[1:] - x[1:] * y[:-1])) / 2.0


def extract_object_polygon(
    mask: np.ndarray,
    crop_box: tuple[int, int, int, int],
    tolerance: float = 1.0,
) -> dict:
    """Trace the boundary of the single object whose outline encloses the crop's own center.

    The outline mask is cropped with the exact crop box of the warehouse JPGs (so the polygon is
    already in the same crop-relative pixel space as the composite image), then the interior of
    every closed outline is filled (`scipy.ndimage.binary_fill_holes`). An outline open at the crop
    edge (the object was bigger than the crop, or merged with a neighbor) usually never fills,
    because a filled "hole" cannot touch the border by definition; we only accept the one filled
    region that covers the crop's own center pixel, which rejects a neighboring cell's outline even
    if its polygon happens to close within the same crop. But the object's own *outline pixels*
    (not the hole they enclose) can still reach the crop's outermost row/column without the hole
    itself being border-connected -- `find_contours` traces an open polyline (not a closed ring,
    first vertex != last) in that case, so the ring closure is checked explicitly and reported as
    STATUS_OPEN_RING rather than accepted as STATUS_OK.

    Parameters
    ----------
    mask : np.ndarray
        2D boolean outline-pixel mask of one field of view (load_outline_mask).
    crop_box : tuple[int, int, int, int]
        (y_min, y_max, x_min, x_max) from `nucleocentric_crops.get_nucleocentric_crop_box`.
    tolerance : float
        Douglas-Peucker tolerance (pixels) for `skimage.measure.approximate_polygon`.

    Returns
    -------
    dict
        status : str (STATUS_* above)
        x, y : list[float] | None (crop-relative pixel coordinates, closed ring, or None)
        vertex_count : int | None
        area_px2 : float | None
    """
    empty = {"status": None, "x": None, "y": None, "vertex_count": None, "area_px2": None}
    y_min, y_max, x_min, x_max = crop_box
    height, width = y_max - y_min, x_max - x_min
    local_mask, _ = crop_with_padding(mask, crop_box, pad_value=False)
    filled = ndimage.binary_fill_holes(local_mask)
    labeled = label(filled, connectivity=1)
    center_row, center_col = height // 2, width // 2
    target = labeled[center_row, center_col]
    if target == 0:
        return {**empty, "status": STATUS_NOT_CLOSED}
    contours = find_contours((labeled == target).astype(np.uint8), level=0.5)
    if not contours:
        return {**empty, "status": STATUS_CONTOUR_FAILED}
    contour = max(contours, key=len)
    simplified = approximate_polygon(contour, tolerance=tolerance)
    if len(simplified) < 4:
        return {**empty, "status": STATUS_TOO_FEW_VERTICES}
    if not np.allclose(simplified[0], simplified[-1]):
        return {**empty, "status": STATUS_OPEN_RING}
    y, x = simplified[:, 0].tolist(), simplified[:, 1].tolist()
    return {
        "status": STATUS_OK,
        "x": x,
        "y": y,
        "vertex_count": len(x),
        "area_px2": polygon_area_px2(np.asarray(x), np.asarray(y)),
    }


def polygon_to_wkb(x, y) -> bytes | None:
    """Encode a crop-relative polygon as OGC Well-Known Binary, or None if it is not a simple polygon.

    Uses shapely, not https://github.com/WayScience/ome-arrow directly: WKB is the stable,
    decades-old interchange encoding that GeoArrow/GeoParquet-style tools (ome-arrow included) are
    built on, without taking a dependency on an early-stage package.

    Parameters
    ----------
    x, y : list[float] | np.ndarray
        Vertex coordinates of a closed ring (extract_object_polygon's "x"/"y").

    Returns
    -------
    bytes | None
        WKB bytes of a valid, non-empty Polygon, or None when the ring cannot form one (for example
        a self-intersecting simplification).
    """
    from shapely.geometry import Polygon
    from shapely.wkb import dumps

    try:
        polygon = Polygon(zip(x, y))
        if not polygon.is_valid:
            polygon = polygon.buffer(0)
        if polygon.is_empty or not polygon.is_valid or polygon.geom_type != "Polygon":
            return None
        return dumps(polygon)
    except Exception:
        return None


def polygons_to_geoarrow(x_lists, y_lists):
    """Encode a whole column of crop-relative polygons as a native GeoArrow polygon array.

    Builds a `geoarrow.pyarrow` `geoarrow.polygon` extension array (storage
    `list<rings: list<vertices: struct<x, y>>>`) for a batch of cells at once, rather than one
    polygon at a time, because GeoArrow's own conversion machinery (`geoarrow.pyarrow.as_geoarrow`)
    operates on a whole array. See the CAUTION in this module's docstring before reading back a
    Parquet file that stores this column's output: use DuckDB, not `pyarrow`/`pandas` directly, in
    any process that calls this function.

    Parameters
    ----------
    x_lists, y_lists : Sequence[list[float] | None]
        One entry per cell, each either a closed ring's x/y coordinates (extract_object_polygon's
        "x"/"y") or None. Both sequences must be the same length and pair up index-by-index.

    Returns
    -------
    geoarrow.pyarrow.PolygonType array
        Same length as the inputs; a None pair becomes a null entry, not a zero-length polygon.
    """
    import geoarrow.pyarrow as ga
    import pyarrow as pa

    if len(x_lists) == 0:
        return ga.polygon().wrap_array(pa.array([], type=ga.polygon().storage_type))
    wkb = [
        polygon_to_wkb(x, y) if x is not None and y is not None else None
        for x, y in zip(x_lists, y_lists)
    ]
    wkb_array = ga.array(pa.array(wkb, type=pa.binary()), type_=ga.wkb())
    return ga.as_geoarrow(wkb_array)
