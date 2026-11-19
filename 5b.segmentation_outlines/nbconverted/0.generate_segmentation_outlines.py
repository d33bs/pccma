#!/usr/bin/env python
# coding: utf-8

# # Trace per-cell segmentation boundary polygons
# 
# Re-derive one closed polygon per cell from CellProfiler's own per-field outline overlays (already
# written by `3.cp_analysis`'s OverlayOutlines module during segmentation QC), using the exact
# nucleocentric crop box of the JPG warehouse (`5a.object_image_crops`) so each polygon lands in the
# same crop-relative pixel space as the composite JPGs. Geometry comes entirely from
# `crops_index.parquet` and the outline TIFFs; this module reads neither converted profiles, QC
# annotations, nor illumination functions. One job processes a plate (`crops_index.parquet` from one
# plate can mix cell lines; grouping is per field of view, not per cell line, since there is no
# per-cell-line QC step here, unlike `5a.object_image_crops`).

# ## Imports and job settings

# In[ ]:


import argparse
import hashlib
import os
import pathlib
import time

import pandas as pd
from crop_warehouse import FINAL_PROFILE_KEY, GEOMETRY_COLUMNS, guard_plate_scope, require_unique_keys, requested_plate
from path_utils import get_root_dir
from segmentation_outlines import (
    COMPARTMENTS,
    STATUS_MISSING_SOURCE,
    extract_object_polygon,
    load_outline_mask,
    outline_filename,
)

try:
    get_ipython()
    in_notebook = True
except NameError:
    in_notebook = False

root_dir = get_root_dir()
module_dir = root_dir / '5b.segmentation_outlines'
crops_index_path = pathlib.Path(os.environ.get('PCCMA_CROPS_INDEX',
    str(root_dir / '5a.object_image_crops/data/object_image_crops/crops_index.parquet'))).expanduser().resolve()
outlines_dir = pathlib.Path(os.environ.get('PCCMA_OUTLINES_DIR',
    '~/mnt/bandicoot/PCCMA_data/Atlas_v2_profiles/sqlite_outputs')).expanduser().resolve()
warehouse_dir = pathlib.Path(os.environ.get('PCCMA_SEGMENTATION_DIR',
    str(module_dir / 'data/segmentation_outlines'))).expanduser().resolve()

if not in_notebook:
    parser = argparse.ArgumentParser(description='Trace per-cell segmentation boundary polygons for one plate')
    parser.add_argument('--plate', required=True)
    args = parser.parse_args()
    plate = args.plate
else:
    plate = 'BR00150695'

# Identifiers are path components, never paths supplied by a job.
if pathlib.Path(plate).name != plate or plate in ('.', '..'):
    raise ValueError(f'Invalid job identifier: {plate}')

selected_plate = requested_plate()
if selected_plate and plate != selected_plate:
    raise ValueError(f'Job plate {plate} is outside requested scope {selected_plate}')
guard_plate_scope(warehouse_dir, selected_plate)

manifest_path = warehouse_dir / 'manifests' / f'{plate}_manifest.parquet'


# ## Load the crop index for this plate

# In[ ]:


crops = pd.read_parquet(crops_index_path)
crops = crops.loc[crops['Metadata_Plate'] == plate]
if crops.empty:
    raise FileNotFoundError(f'No crops_index rows for plate {plate} in {crops_index_path}')
require_unique_keys(crops, FINAL_PROFILE_KEY, 'Crops index')
for well in crops['Metadata_Well'].unique():
    if pathlib.Path(well).name != well or well in ('.', '..'):
        raise ValueError(f'Invalid well path component: {well}')
print(f'{plate}: {len(crops)} crop-index cells across {crops["Metadata_cell_line"].nunique()} cell lines to trace')


# ## Trace one polygon per cell per compartment, then publish the job manifest

# In[ ]:


start = time.monotonic()
manifest_path.parent.mkdir(parents=True, exist_ok=True)
rows = []
status_counts = {compartment: {} for compartment in COMPARTMENTS}
for cell_line, cell_line_df in crops.groupby('Metadata_cell_line', sort=True):
    field_dir = outlines_dir / f'{plate}_{cell_line}'
    for (well, site), field_df in cell_line_df.groupby(['Metadata_Well', 'Metadata_Site'], sort=True):
        field_cache = {}
        for compartment in COMPARTMENTS:
            tiff_path = field_dir / outline_filename(compartment, plate, well, int(site))
            if tiff_path.is_file():
                field_cache[compartment] = {
                    'mask': load_outline_mask(tiff_path),
                    'source_tiff': tiff_path.relative_to(outlines_dir).as_posix(),
                    'source_sha256': hashlib.sha256(tiff_path.read_bytes()).hexdigest(),
                }
            else:
                field_cache[compartment] = None
        for record in field_df.to_dict('records'):
            row = {key: record[key] for key in FINAL_PROFILE_KEY}
            row['Metadata_cell_line'] = record['Metadata_cell_line']
            row['Metadata_time_point'] = record['Metadata_time_point']
            crop_box = tuple(int(record[column]) for column in GEOMETRY_COLUMNS)
            for compartment in COMPARTMENTS:
                prefix = f'{compartment}_Segmentation_Boundary_'
                cached = field_cache[compartment]
                if cached is None:
                    status = STATUS_MISSING_SOURCE
                    polygon = {'x': None, 'y': None, 'vertex_count': None, 'area_px2': None}
                    source_tiff, source_sha256 = None, None
                else:
                    polygon = extract_object_polygon(cached['mask'], crop_box)
                    status = polygon['status']
                    source_tiff, source_sha256 = cached['source_tiff'], cached['source_sha256']
                row[prefix + 'Status'] = status
                row[prefix + 'X'] = polygon['x']
                row[prefix + 'Y'] = polygon['y']
                row[prefix + 'VertexCount'] = polygon['vertex_count']
                row[prefix + 'AreaPx2'] = polygon['area_px2']
                row[prefix + 'Source_TIFF'] = source_tiff
                row[prefix + 'Source_SHA256'] = source_sha256
                status_counts[compartment][status] = status_counts[compartment].get(status, 0) + 1
            rows.append(row)

manifest_df = pd.DataFrame(rows)
for compartment in COMPARTMENTS:
    manifest_df[f'{compartment}_Segmentation_Boundary_VertexCount'] = manifest_df[
        f'{compartment}_Segmentation_Boundary_VertexCount'].astype('Int64')
    manifest_df[f'{compartment}_Segmentation_Boundary_AreaPx2'] = manifest_df[
        f'{compartment}_Segmentation_Boundary_AreaPx2'].astype('Float64')
require_unique_keys(manifest_df, FINAL_PROFILE_KEY, 'Job manifest')
temporary = manifest_path.with_suffix('.tmp.parquet')
manifest_df.to_parquet(temporary, index=False)
temporary.replace(manifest_path)
summary = ', '.join(
    f'{compartment}: ' + ', '.join(f'{status}={count}' for status, count in sorted(status_counts[compartment].items()))
    for compartment in COMPARTMENTS
)
print(f'Saved {manifest_path}: {len(manifest_df)} cells in {time.monotonic() - start:.1f}s ({summary})')

