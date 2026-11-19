# 5b. Segmentation outlines

## Why

The dashboard (`7.dashboard`) wants to draw a segmentation outline over a crop, but
`5a.object_image_crops`'s JPG warehouse has no per-cell boundary geometry: the crop box is a
rectangle, not the cell's actual shape. CellProfiler's own `OverlayOutlines` module already drew
every object's outline during `3.cp_analysis` (one `<Compartment>Outlines_<plate>_<well>_<site>.tiff`
per field of view, used today only for visual QC in
`4.preprocess_features/1.sc_quality_control.ipynb`), but that raster has no per-object identity —
every cell's outline in a field is the same flat color on the same image.

This module re-derives one closed polygon per cell from that raster, so a later step (or the
dashboard itself) can draw it directly over the matching JPG crop without re-running segmentation.

## Approach

For each cell already published in `crops_index.parquet` (by `5a.object_image_crops`):

1. Load the `Nuclei`/`Cells` outline TIFF of that cell's own field of view (loaded once per field,
   reused for every cell in it).
2. Crop the outline raster with the **exact same nucleocentric crop box** already used for that
   cell's JPG crop, so the polygon ends up in the same crop-relative pixel space (origin top-left)
   as the composite JPG — no extra alignment step is needed later.
3. Fill the interior of every closed outline in the crop (`scipy.ndimage.binary_fill_holes`) and
   keep only the one whose filled interior covers the crop's own center pixel. An outline that
   exits the crop (object bigger than the crop, or merged with a neighbor) never fills, so it is
   correctly rejected rather than mistaken for the cell.
4. Trace that region's boundary (`skimage.measure.find_contours` + `approximate_polygon`) and
   encode it two ways: a plain `list<double>` x/y coordinate pair, and a native GeoArrow polygon
   (nested `list<rings: list<vertices: struct<x, y>>>`, tagged with the `geoarrow.polygon` Arrow
   extension type) via `geoarrow-pyarrow`.

All of this logic already exists, already tested, in `utils/segmentation_outlines.py`
(`utils/tests/test_segmentation_outlines.py`, 14 tests) — this module only imports and orchestrates
it over `crops_index.parquet` and the real outline TIFFs; it does not change that file.

This references the shape and spirit of
[WayScience/ome-arrow](https://github.com/WayScience/ome-arrow) (per-object geometry alongside
tabular profiles), but does **not** depend on it — it has no release. Instead this module uses
[`geoarrow-pyarrow`](https://github.com/geoarrow/geoarrow-python), a stable, released,
non-domain-specific implementation of the GeoArrow spec, plus plain Parquet `list<double>` columns:
both get the same geometry/columnar interoperability goal without taking a dependency on an
early-stage, domain-specific package. `shapely`'s OGC WKB encoding is still used internally
(`polygon_to_wkb`) to validate each ring before handing it to `geoarrow-pyarrow`, and remains
available directly for anything that specifically wants WKB.

CAUTION, found and reproduced locally (`geoarrow-pyarrow` 0.3.0 / `geoarrow-c` 0.4.0,
macOS/CPython 3.11): once `geoarrow.pyarrow` has been imported in a process — directly, or
transitively through `polygons_to_geoarrow` — later reading a Parquet file that contains one of
its extension columns with `pyarrow.parquet.read_table` or `pandas.read_parquet` *in that same
process* can hang forever at interpreter shutdown, even though the read itself returns the correct
data immediately. Writing such a column is unaffected, and DuckDB's own native Parquet reader is
unaffected (it never goes through pyarrow's registered-extension-type deserialization). Read a
published GeoArrow-bearing Parquet file back with DuckDB
(`duckdb.connect().execute(...).fetchall()` or `.df()`), not with `pyarrow`/`pandas` directly, in
any process that also calls `polygons_to_geoarrow`. `1.build_segmentation_index.py` and this
module's pipeline tests both follow that rule.

This module reads only `crops_index.parquet` and the outline TIFFs. It does not touch converted
profiles, QC annotations, or illumination-correction functions — segmentation geometry is fully
determined by geometry that already exists.

## Schema

One row per cell (same grain as `crops_index.parquet`), columns:

| Column(s) | Meaning |
|---|---|
| `crop_warehouse.FINAL_PROFILE_KEY` (`Metadata_Plate`, `Metadata_Well`, `Metadata_Site`, `Metadata_ImageNumber`, `Metadata_Nuclei_Number_Object_Number`, `Metadata_Cells_Number_Object_Number`) | Unique cell key, same as the crop index. |
| `Metadata_cell_line`, `Metadata_time_point` | Carried through from `crops_index.parquet`. |
| `<Compartment>_Segmentation_Boundary_Status` | One of the status codes below, for `<Compartment>` in (`Nuclei`, `Cells`). |
| `<Compartment>_Segmentation_Boundary_X`, `_Y` | `list<double>` crop-relative pixel coordinates (closed ring, first vertex repeated last), or `null`. Published by the per-plate job manifest (`0.generate_segmentation_outlines.py`). |
| `<Compartment>_Segmentation_Boundary_Geometry` | Native GeoArrow polygon (`geoarrow.polygon` extension type; storage `list<rings: list<vertices: struct<x, y>>>`), or `null`. Attached only by the index build (`1.build_segmentation_index.py`), from the same `X`/`Y` lists — the per-plate job manifests do not carry this column. |
| `<Compartment>_Segmentation_Boundary_VertexCount` | Nullable integer, number of ring vertices. |
| `<Compartment>_Segmentation_Boundary_AreaPx2` | Nullable double, polygon area in squared pixels. |
| `<Compartment>_Segmentation_Boundary_Source_TIFF` | Outline TIFF path, relative to `PCCMA_OUTLINES_DIR`, that this cell's polygon was traced from, or `null`. |
| `<Compartment>_Segmentation_Boundary_Source_SHA256` | SHA-256 of that TIFF's raw bytes (hashed once per field, shared by every cell in it), or `null`. |

Status codes (from `utils/segmentation_outlines.py`):

- `ok` — a closed outline was found covering the crop's own center pixel; `X`/`Y` are populated.
  `Geometry` is usually also populated, but can still be `null`: building it applies `shapely`'s
  `buffer(0)` cleanup to a self-intersecting simplification, and drops the rare ring that still
  isn't a simple polygon afterwards, even though `X`/`Y` exist. In the verified full-plate run below
  every `ok` row of both compartments had a non-null `Geometry`.
- `open_ring_at_crop_edge` — a filled region covers the crop's own center pixel, but the traced
  boundary ring reaches the crop's outermost row/column without closing on itself (the object, or a
  hole in it, is cut open by the crop edge rather than being fully inside the crop). `X`/`Y`/
  `Geometry`/`VertexCount`/`AreaPx2` are `null`. In the verified full-plate run below this was the
  majority status for `Cells` (56,649 of 80,493) and rare for `Nuclei` (1,435 of 80,493) — the
  `Cells` boundary is far more likely to exceed the 224px nucleocentric crop, which is exactly why
  the dashboard (`7.dashboard/build_report.py`) only draws the `Nuclei` outline.
- `not_closed_at_center` — the crop's own center pixel is not inside any filled outline (outline
  absent entirely, or present but not covering the center). `X`/`Y`/`Geometry`/`VertexCount`/
  `AreaPx2` are `null`.
- `too_few_vertices` — the traced ring simplified down to fewer than 4 points. `null` fields as above.
- `contour_extraction_failed` — a filled region was found but contour tracing returned nothing.
  `null` fields as above.
- `missing_source` — this compartment's outline TIFF for the cell's field of view does not exist
  under `PCCMA_OUTLINES_DIR`. Every cell of that field still gets a row (never dropped); `X`/`Y`/
  `Geometry`/`VertexCount`/`AreaPx2`/`Source_TIFF`/`Source_SHA256` are all `null`.

## Environment variables

| Variable | Default | Meaning |
|---|---|---|
| `PCCMA_CROPS_INDEX` | `5a.object_image_crops/data/object_image_crops/crops_index.parquet` | Input: the published crop index to trace polygons for. |
| `PCCMA_OUTLINES_DIR` | `~/mnt/bandicoot/PCCMA_data/Atlas_v2_profiles/sqlite_outputs` | Root of CellProfiler's `OverlayOutlines` output, one `<plate>_<cell_line>` subfolder per job, holding `<Compartment>Outlines_<plate>_<well>_<site>.tiff`. Same convention as `sqlite_dir` in `4.preprocess_features/1.sc_quality_control.ipynb`. |
| `PCCMA_SEGMENTATION_DIR` | `5b.segmentation_outlines/data/segmentation_outlines` | Output warehouse root (job manifests and the published index), mirroring `PCCMA_CROPS_DIR` in `5a.object_image_crops`. |
| `PCCMA_PLATE` | unset (no scope restriction) | Same plate-scope guard as `5a.object_image_crops` (`crop_warehouse.requested_plate`/`guard_plate_scope`); when set, every job and the index build are restricted to that one plate. |

## How to run

```bash
cd /Users/buntend/Documents/work/pccma

# one job per plate (not per plate+cell_line: crops_index.parquet already carries Metadata_cell_line,
# and there is no per-cell-line QC step here)
PYTHONPATH=utils env -u PYTHONPATH environments/preprocessing/.venv/bin/python \
  5b.segmentation_outlines/nbconverted/0.generate_segmentation_outlines.py --plate BR00150695

# concatenate every plate's manifest, validate against crops_index.parquet, publish, and do a
# DuckDB read-back check
PYTHONPATH=utils env -u PYTHONPATH environments/preprocessing/.venv/bin/python \
  5b.segmentation_outlines/nbconverted/1.build_segmentation_index.py
```

Both scripts read `PCCMA_CROPS_INDEX`/`PCCMA_OUTLINES_DIR`/`PCCMA_SEGMENTATION_DIR`/`PCCMA_PLATE`
from the environment, defaulting as in the table above.

## Verified full-plate run

This is a real, end-to-end run against the complete, genuine CellProfiler outline output for one
whole plate — **not a synthetic or partial pilot**. `PCCMA_OUTLINES_DIR` was pointed at a local copy
of every `Nuclei`/`Cells` outline TIFF for plate `BR00150695` (all 13 `<plate>_<cell_line>`
subfolders, 72 TIFFs each, shaped exactly like the real `PCCMA_OUTLINES_DIR` convention), and
`PCCMA_CROPS_INDEX` was pointed at the real
`5a.object_image_crops/data/pilot_runs/BR00150695-20261113T155150Z-g790ecb4/crops_index.parquet`
(80,493 cells, the whole plate, across 13 cell lines).

`0.generate_segmentation_outlines.py --plate BR00150695`:

```
BR00150695: 80493 crop-index cells across 13 cell lines to trace
Saved .../segmentation_outlines/manifests/BR00150695_manifest.parquet: 80493 cells in 425.1s
(Nuclei: ok=79058, open_ring_at_crop_edge=1435,
 Cells: not_closed_at_center=676, ok=23168, open_ring_at_crop_edge=56649)
```

Every one of the 80,493 cells of the plate got a real traced result for both compartments (no
`missing_source` at all, since every field's outline TIFF was present this time). `Nuclei` resolved
`ok` for 98.2% of cells; `Cells` resolved `ok` for only 28.8%, with the remaining 70.4% cut open at
the crop edge (`open_ring_at_crop_edge`) — real, direct evidence for the README's "Why" section
claim that `Cells` boundaries are not reliable at the current 224px crop size, and for why the
dashboard only draws the `Nuclei` outline.

`1.build_segmentation_index.py` against that same output:

```
Saved .../segmentation_outlines/segmentation_index.parquet: 80493 cells, 1 plate jobs,
status={'Nuclei': {'ok': 79058, 'open_ring_at_crop_edge': 1435},
        'Cells': {'open_ring_at_crop_edge': 56649, 'ok': 23168, 'not_closed_at_center': 676}}
DuckDB read-back: 80493 rows; Nuclei Geometry/X-Y spot check passed on {'Metadata_Plate': 'BR00150695',
'Metadata_Well': 'C03', 'Metadata_Site': 1, 'Metadata_ImageNumber': 1,
'Metadata_Nuclei_Number_Object_Number': 3, 'Metadata_Cells_Number_Object_Number': 2}
DuckDB read-back: 80493 rows; Cells Geometry/X-Y spot check passed on {'Metadata_Plate': 'BR00150695',
'Metadata_Well': 'C03', 'Metadata_Site': 1, 'Metadata_ImageNumber': 1,
'Metadata_Nuclei_Number_Object_Number': 31, 'Metadata_Cells_Number_Object_Number': 22}
```

The segmentation index's key set was confirmed to exactly match `crops_index.parquet`'s key set for
plate `BR00150695` (80,493 cells both sides), DuckDB's own row count matched the published Parquet
file (80,493), and for both the `Nuclei` and `Cells` compartments a sampled `ok` row's GeoArrow
`Geometry` decoded to the exact same vertex coordinates as its stored `X`/`Y` lists. A direct
DuckDB `DESCRIBE` of the published file confirms the native nested type with no extension needed to
read it:

```
Nuclei_Segmentation_Boundary_Geometry  STRUCT(x DOUBLE, y DOUBLE)[][]
Cells_Segmentation_Boundary_Geometry   STRUCT(x DOUBLE, y DOUBLE)[][]
```

and a separate `count(...)` over that same file, run afterward in a fresh process (so any
interpreter-shutdown hang from the GeoArrow caution above would show up), confirmed all 79,058 `ok`
`Nuclei` rows and all 23,168 `ok` `Cells` rows have a non-null `Geometry` — the process exited
cleanly in under a second.
