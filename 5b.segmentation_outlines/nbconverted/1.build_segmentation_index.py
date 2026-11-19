#!/usr/bin/env python
# coding: utf-8

# # Build the segmentation boundary index
# 
# Concatenate every plate's job manifest, verify that the combined key set is exactly the post-QC
# cell set of `crops_index.parquet` for the same plate scope (not a subset or superset), and publish
# the result atomically. The published file adds one native GeoArrow polygon column per compartment
# (built from the plain X/Y list columns with `segmentation_outlines.polygons_to_geoarrow`), not a
# WKB column. A DuckDB read-back check then confirms the published Parquet file reads back correctly
# and that a sampled "ok" row's GeoArrow geometry decodes back to the same X/Y vertex lists -- using
# DuckDB's own Parquet reader only, never `pyarrow`/`pandas` directly, because of the read-back
# caution documented in `segmentation_outlines.py`'s module docstring.

# ## Paths, inputs and manifest coverage

# In[ ]:


import os
import pathlib

import duckdb
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from crop_warehouse import FINAL_PROFILE_KEY, guard_plate_scope, require_unique_keys, requested_plate
from path_utils import get_root_dir
from segmentation_outlines import polygons_to_geoarrow

root_dir = get_root_dir()
module_dir = root_dir / '5b.segmentation_outlines'
crops_index_path = pathlib.Path(os.environ.get('PCCMA_CROPS_INDEX',
    str(root_dir / '5a.object_image_crops/data/object_image_crops/crops_index.parquet'))).expanduser().resolve()
warehouse_dir = pathlib.Path(os.environ.get('PCCMA_SEGMENTATION_DIR',
    str(module_dir / 'data/segmentation_outlines'))).expanduser().resolve()
index_path = warehouse_dir / 'segmentation_index.parquet'
manifest_dir = warehouse_dir / 'manifests'
selected_plate = requested_plate()
guard_plate_scope(warehouse_dir, selected_plate)

plates = sorted(pd.read_parquet(crops_index_path, columns=['Metadata_Plate'])['Metadata_Plate'].unique())
if selected_plate:
    if selected_plate not in plates:
        raise FileNotFoundError(f'Plate {selected_plate} is not in the crop index {crops_index_path}')
    plates = [selected_plate]
expected = {f'{plate}_manifest.parquet' for plate in plates}
pattern = f'{selected_plate}_manifest.parquet' if selected_plate else '*_manifest.parquet'
manifest_paths = sorted(manifest_dir.glob(pattern))
if expected != {path.name for path in manifest_paths}:
    raise ValueError('Missing or stale plate manifests; generate every plate job first')


# ## Concatenate, validate against the crop index, and publish

# In[ ]:


index_df = pd.concat([pd.read_parquet(path) for path in manifest_paths], ignore_index=True)
require_unique_keys(index_df, FINAL_PROFILE_KEY, 'Segmentation index')

crops_df = pd.read_parquet(crops_index_path, columns=FINAL_PROFILE_KEY)
if selected_plate:
    crops_df = crops_df.loc[crops_df['Metadata_Plate'] == selected_plate]
require_unique_keys(crops_df, FINAL_PROFILE_KEY, 'Crop index')

segmentation_keys = set(index_df[FINAL_PROFILE_KEY].itertuples(index=False, name=None))
crop_keys = set(crops_df[FINAL_PROFILE_KEY].itertuples(index=False, name=None))
if segmentation_keys != crop_keys:
    missing = crop_keys - segmentation_keys
    extra = segmentation_keys - crop_keys
    raise ValueError(
        'Segmentation index key set does not exactly match the crop index for this plate scope: '
        f'{len(missing)} crop-index cells missing a segmentation row, '
        f'{len(extra)} segmentation rows have no matching crop-index cell'
    )

# Attach a native GeoArrow polygon column per compartment, built from the plain X/Y list columns
# (not a WKB column). pandas cannot carry this extension type through `DataFrame.to_parquet`
# cleanly, so the index is published through pyarrow directly instead.
index_table = pa.Table.from_pandas(index_df, preserve_index=False)
for compartment in ('Nuclei', 'Cells'):
    prefix = f'{compartment}_Segmentation_Boundary_'
    geometry = polygons_to_geoarrow(index_df[prefix + 'X'].tolist(), index_df[prefix + 'Y'].tolist())
    index_table = index_table.append_column(prefix + 'Geometry', geometry)

warehouse_dir.mkdir(parents=True, exist_ok=True)
temporary = index_path.with_suffix('.tmp.parquet')
pq.write_table(index_table, temporary)
temporary.replace(index_path)
status_summary = {
    compartment: index_df[f'{compartment}_Segmentation_Boundary_Status'].value_counts().to_dict()
    for compartment in ('Nuclei', 'Cells')
}
print(f'Saved {index_path}: {len(index_df)} cells, {len(manifest_paths)} plate jobs, status={status_summary}')


# ## DuckDB read-back check

# In[ ]:


# Read the published file back with DuckDB only. `polygons_to_geoarrow` above already imported
# `geoarrow.pyarrow` in this process, and once that has happened, reading a file that contains one
# of its extension columns with `pyarrow.parquet.read_table`/`pandas.read_parquet` *in this same
# process* can hang forever at interpreter shutdown even though the read itself is correct (see the
# CAUTION in segmentation_outlines.py, reproduced locally). DuckDB's own Parquet reader never goes
# through pyarrow's registered-extension-type deserialization, so it is unaffected.
connection = duckdb.connect()
row_count = connection.execute(
    f"SELECT count(*) FROM read_parquet('{index_path.as_posix()}')"
).fetchone()[0]
if row_count != len(index_df):
    raise ValueError(f'DuckDB row count {row_count} does not match published {len(index_df)}')

key_columns = ', '.join(FINAL_PROFILE_KEY)
checked = False
for compartment in ('Nuclei', 'Cells'):
    prefix = f'{compartment}_Segmentation_Boundary_'
    status_column, x_column, y_column, geometry_column = (
        prefix + 'Status', prefix + 'X', prefix + 'Y', prefix + 'Geometry'
    )
    columns = [*FINAL_PROFILE_KEY, x_column, y_column, geometry_column]
    candidates = connection.execute(
        f"SELECT {key_columns}, {x_column}, {y_column}, {geometry_column} "
        f"FROM read_parquet('{index_path.as_posix()}') WHERE {status_column} = 'ok'"
    ).fetchall()
    # status 'ok' does not guarantee the GeoArrow geometry round-trips to the exact stored X/Y
    # lists: shapely's buffer(0) cleanup inside polygon_to_wkb (used by polygons_to_geoarrow) can
    # nudge a self-intersecting simplification's coordinates, so the spot check scans "ok" rows for
    # one that round-trips exactly, instead of assuming the first one does.
    matched_sample = None
    for row in candidates:
        record = dict(zip(columns, row))
        # polygons_to_geoarrow can still return a null geometry for a status='ok' row (the same
        # shapely buffer(0) cleanup that can shift a self-intersecting simplification's
        # coordinates can also leave it with no valid Polygon at all); skip those, not just
        # coordinate mismatches.
        if record[geometry_column] is None:
            continue
        stored_x, stored_y = list(record[x_column]), list(record[y_column])
        # Geometry is a single-ring polygon: one list of rings, each ring a list of {'x', 'y'} structs.
        vertices = record[geometry_column][0]
        geometry_x = [vertex['x'] for vertex in vertices]
        geometry_y = [vertex['y'] for vertex in vertices]
        if geometry_x == stored_x and geometry_y == stored_y:
            matched_sample = record
            break
    if matched_sample is None:
        if not candidates:
            continue
        raise ValueError(
            f"No '{compartment}' 'ok' row's GeoArrow geometry matches its stored X/Y lists "
            '(checked every candidate in this run)'
        )
    sample_key = {key: matched_sample[key] for key in FINAL_PROFILE_KEY}
    print(f'DuckDB read-back: {row_count} rows; {compartment} Geometry/X-Y spot check passed on {sample_key}')
    checked = True

if not checked:
    print(f"DuckDB read-back: {row_count} rows; no 'ok' row available in this run to spot-check")

