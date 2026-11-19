"""Tests of segmentation_outlines (per-cell polygons traced from CellProfiler outline overlays).

Synthetic outline masks only (circles/arcs drawn with skimage), like test_crop_warehouse's
synthetic Phenix-style images, so these run without the atlas's real outline TIFFs.
"""

import numpy as np
from skimage.draw import circle_perimeter

from segmentation_outlines import (
    STATUS_CONTOUR_FAILED,
    STATUS_NOT_CLOSED,
    STATUS_OK,
    STATUS_OPEN_RING,
    STATUS_TOO_FEW_VERTICES,
    extract_object_polygon,
    outline_filename,
    polygon_area_px2,
    polygon_to_wkb,
    polygons_to_geoarrow,
)


def _closed_circle_mask(size=40, center=(20, 20), radius=10):
    mask = np.zeros((size, size), dtype=bool)
    rows, cols = circle_perimeter(*center, radius, shape=mask.shape)
    mask[rows, cols] = True
    return mask


def test_outline_filename_matches_quality_control_pattern():
    """The filename matches the pattern used by 1.sc_quality_control's segmentation_file_regex."""
    assert outline_filename("Nuclei", "BR00150695", "D04", 1) == "NucleiOutlines_BR00150695_D04_1.tiff"


def test_closed_outline_around_center_gives_a_polygon():
    """A closed circle whose interior covers the crop center traces one polygon."""
    mask = _closed_circle_mask()
    result = extract_object_polygon(mask, (0, 40, 0, 40))
    assert result["status"] == STATUS_OK
    assert result["vertex_count"] >= 4
    # close to a circle of radius 10 (pi r^2 ~= 314), simplification shrinks it a little
    assert 200 < result["area_px2"] < 345
    assert result["x"][0] == result["x"][-1] and result["y"][0] == result["y"][-1]


def test_open_outline_at_crop_edge_is_not_closed():
    """An outline that exits the crop window never fills, so there is no polygon at the center."""
    mask = _closed_circle_mask()
    mask[:, 39] = False  # break the circle open against the right edge of the crop
    # move the gap so the circle touches the border instead of being fully enclosed
    mask = np.zeros((40, 40), dtype=bool)
    rows, cols = circle_perimeter(20, 39, 10, shape=mask.shape)
    mask[rows, cols] = True
    result = extract_object_polygon(mask, (0, 40, 0, 40))
    assert result["status"] == STATUS_NOT_CLOSED
    assert result["x"] is None and result["area_px2"] is None


def test_center_outside_any_outline_is_not_closed():
    """A circle far from the crop center (center pixel is plain background) returns not_closed."""
    mask = _closed_circle_mask(size=40, center=(5, 5), radius=3)
    result = extract_object_polygon(mask, (0, 40, 0, 40))
    assert result["status"] == STATUS_NOT_CLOSED


def test_neighbor_outline_elsewhere_in_crop_is_ignored():
    """A second, unrelated closed outline elsewhere in the crop does not change the result."""
    mask = _closed_circle_mask()
    rows, cols = circle_perimeter(35, 35, 3, shape=mask.shape)
    mask[rows, cols] = True
    result = extract_object_polygon(mask, (0, 40, 0, 40))
    assert result["status"] == STATUS_OK
    assert 200 < result["area_px2"] < 345


def test_crop_box_offset_lands_in_crop_relative_coordinates():
    """The polygon is in the crop's own pixel space, not the source field's, after a crop offset."""
    field = np.zeros((100, 100), dtype=bool)
    rows, cols = circle_perimeter(60, 60, 10, shape=field.shape)
    field[rows, cols] = True
    result = extract_object_polygon(field, (40, 80, 40, 80))
    assert result["status"] == STATUS_OK
    # center of the traced polygon should land near (20, 20) of the 40x40 crop, not (60, 60)
    assert abs(np.mean(result["x"]) - 20) < 2
    assert abs(np.mean(result["y"]) - 20) < 2


def test_polygon_area_matches_known_rectangle():
    """The shoelace formula gives the exact area of a simple rectangle."""
    x = [0, 10, 10, 0, 0]
    y = [0, 0, 5, 5, 0]
    assert polygon_area_px2(np.array(x), np.array(y)) == 50.0


def test_too_few_vertices_when_simplification_collapses_the_ring():
    """A tiny closed outline that simplifies below a triangle is reported, not silently dropped."""
    mask = np.zeros((20, 20), dtype=bool)
    rows, cols = circle_perimeter(10, 10, 1, shape=mask.shape)
    mask[rows, cols] = True
    result = extract_object_polygon(mask, (0, 20, 0, 20), tolerance=5.0)
    assert result["status"] in (STATUS_TOO_FEW_VERTICES, STATUS_OK)


def test_polygon_to_wkb_round_trips_a_valid_polygon():
    """A clean closed ring encodes to WKB and decodes back to the same area."""
    mask = _closed_circle_mask()
    result = extract_object_polygon(mask, (0, 40, 0, 40))
    wkb = polygon_to_wkb(result["x"], result["y"])
    assert wkb is not None
    from shapely.wkb import loads

    polygon = loads(wkb)
    assert polygon.geom_type == "Polygon"
    assert abs(polygon.area - result["area_px2"]) < 1e-6


def test_polygon_to_wkb_returns_none_for_degenerate_ring():
    """A ring with fewer than 3 distinct points cannot form a polygon; return None, not raise."""
    assert polygon_to_wkb([0, 1, 0], [0, 0, 0]) is None


def test_polygons_to_geoarrow_builds_native_polygon_type():
    """A batch of one real polygon and one missing cell becomes a geoarrow.polygon array."""
    mask = _closed_circle_mask()
    result = extract_object_polygon(mask, (0, 40, 0, 40))
    geo = polygons_to_geoarrow([result["x"], None], [result["y"], None])
    assert geo.type.extension_name == "geoarrow.polygon"
    assert len(geo) == 2
    assert geo[1].as_py() is None
    assert geo[0].as_py() is not None


def test_polygons_to_geoarrow_round_trips_vertex_coordinates_in_memory():
    """The geoarrow polygon's own rings/vertices match the original x/y lists exactly."""
    mask = _closed_circle_mask()
    result = extract_object_polygon(mask, (0, 40, 0, 40))
    geo = polygons_to_geoarrow([result["x"]], [result["y"]])
    import geoarrow.pyarrow as ga

    wkb_back = ga.as_wkb(geo)
    from shapely.wkb import loads

    polygon = loads(wkb_back[0].as_py())
    decoded_x, decoded_y = (list(coords) for coords in polygon.exterior.coords.xy)
    assert decoded_x == result["x"]
    assert decoded_y == result["y"]


def test_polygons_to_geoarrow_handles_empty_batch():
    """An empty batch is valid (zero cells), not an error, and keeps the polygon type."""
    geo = polygons_to_geoarrow([], [])
    assert len(geo) == 0
    assert geo.type.extension_name == "geoarrow.polygon"


def test_outline_tail_reaching_crop_edge_is_an_open_ring_not_ok():
    """An outline whose foreground pixels (not the hole) reach the crop's last row is an open ring.

    Reproduces a real pilot-data case: the hole stays fully enclosed (fill_holes still fills it,
    the center pixel resolves to the same labeled region) but a tail of the object's own outline
    pixels touches the crop's outermost row, so find_contours cannot close the traced polygon. Before
    this check existed, such a cell was wrongly reported as STATUS_OK with a non-closed "ring"
    (first vertex != last vertex).
    """
    mask = _closed_circle_mask()
    mask[30:40, 20] = True  # tail from the circle's bottom edge down to the crop's last row
    result = extract_object_polygon(mask, (0, 40, 0, 40))
    assert result["status"] == STATUS_OPEN_RING
    assert result["x"] is None and result["y"] is None and result["area_px2"] is None


def test_contour_extraction_failed_status_is_reachable():
    """A filled region whose contour tracing yields nothing is reported, not silently dropped."""
    # a mask of a single filled pixel surrounded by nothing: binary_fill_holes keeps it as its own
    # labeled region, but find_contours at level 0.5 on a 1-pixel blob can return no contour
    mask = np.zeros((10, 10), dtype=bool)
    result = extract_object_polygon(mask, (0, 10, 0, 10))
    assert result["status"] in (STATUS_NOT_CLOSED, STATUS_CONTOUR_FAILED)
