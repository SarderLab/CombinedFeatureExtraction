# CombinedFeatureExtraction

A collection of pipelines for extracting and quantifying morphometric and intensity-based features from segmented kidney whole slide images (WSIs). Developed by [CMI Lab](https://cmilab.nephrology.medicine.ufl.edu/) at the University of Florida.

The plugin operates on annotated WSIs produced by the [Multi-Compartment Segmentation model](https://github.com/SarderLab/Multi-Compartment-Segmentation), which segments kidney functional tissue units (FTUs) — glomeruli, tubules, and arteries/arterioles — from PAS-stained histology slides.

---

## Pipelines

### 1. PathomicsFE (Expanded Granular Features)

The primary pipeline. For each segmented FTU it performs **sub-compartment segmentation** — splitting the structure into three regions:

| Sub-compartment | Description |
|---|---|
| **Nuclei** | Detected using the inverse value channel of HSV space + watershed |
| **Eosinophilic (PAS)** | PAS-positive cytoplasmic region, detected via saturation channel |
| **Luminal Space** | Remaining pixels within the FTU boundary |

After sub-segmentation, it extracts **72 features per FTU** across four categories:

| Category | Features extracted |
|---|---|
| **Morphological** | Area, perimeter, aspect ratio, nuclei count, mean nuclear area, axis lengths |
| **Color** | Mean and standard deviation of R, G, B channels per sub-compartment |
| **Texture** | GLCM-based contrast, homogeneity, correlation, energy per sub-compartment |
| **Distance Transform** | Sum, mean, and max distance transforms (absolute and normalized by area) |

Aggregated slide-level statistics (sum, mean, std, median, min, max) are also computed across all FTUs for each annotation layer.

### 2. ClassicalFeatures

Coarser feature extraction directly from FTU contours without sub-compartmentalization:

- **Pathomic**: Area, mesangial area and fraction per glomerulus; average TBM thickness, cell thickness, and luminal fraction per tubule; arterial area per artery
- **Extended Clinical**: Area and radius per FTU

---

## Running Modes

The plugin supports two modes depending on your environment.

### Mode 1: DSA / HistomicsUI (Girder-coupled)

Runs as a plugin inside a [Digital Slide Archive](https://digitalslidearchive.github.io/digital_slide_archive/) instance. Inputs are passed as Girder file IDs, annotations are read from and written back to the DSA item, and output Excel files are uploaded directly to the item.

**Build:**
```bash
docker build -t dsrithad/fusion1_decoupled:feature-extraction .
```

The container exposes the CLI through `slicer_cli_web` and is registered with DSA via the `entry_path` label. The plugin appears in HistomicsUI under the **HistomicsTK** category.

---

### Mode 2: Local / Notebook (Girder-free)

Runs without any DSA or Girder dependency. Takes local file paths for the WSI and annotation JSON files. All outputs are written to a local directory. Suitable for running on a laptop, compute cluster, or JupyterHub.

**Build:**
```bash
docker build -f Dockerfile.notebook -t dsrithad/fusion1_decoupled:feature-extraction-notebook .
```

**Run:**
```bash
docker run --rm --platform linux/amd64 \
  -v /path/to/CombinedFeatureExtraction:/opt/FExtract \
  -v /path/to/input:/data/input \
  -v /path/to/output:/data/output \
  dsrithad/fusion1_decoupled:feature-extraction-notebook \
  python /opt/FExtract/fextract/cli/PathomicsFE/PathomicsFELocal.py \
    --input_image "/data/input/sample.tif" \
    --annotations_dir /data/input \
    --output_dir /data/output
```

**Without Docker (directly with Python):**
```bash
pip install -e .

python fextract/cli/PathomicsFE/PathomicsFELocal.py \
  --input_image input/sample.tif \
  --annotations_dir input/ \
  --output_dir output/
```

---

## Inputs

### WSI (`--input_image`)
Any whole slide image format supported by `tiffslide`: `.svs`, `.tif`, `.tiff`, `.ndpi`, `.scn`, `.mrxs`, `.vms`, `.vmu`.

### Annotations (`--annotations_dir`)
A folder containing one `.json` file per annotation layer, exported from DSA/HistomicsUI.

**How to export from HistomicsUI:**
1. Open your slide
2. In the Annotations panel, select the annotation layers you want
3. Click the download icon → save as JSON (one file per layer)

Each file should follow this structure (standard DSA annotation export format):
```json
{
  "_id": "...",
  "annotation": {
    "name": "tubules",
    "elements": [
      {
        "type": "polyline",
        "points": [[x1, y1, 0], [x2, y2, 0], "..."]
      }
    ]
  }
}
```

Supported annotation layer names (others are ignored):

| Layer name | Processed |
|---|---|
| `non_globally_sclerotic_glomeruli` | Yes |
| `globally_sclerotic_glomeruli` | Yes |
| `tubules` | Yes |
| `arteries/arterioles` | Yes |
| `gloms` | Yes |
| `cortical_interstitium` | Skipped |
| `medullary_interstitium` | Skipped |

The WSI and annotation JSONs can live in the same folder — the loader only picks up `.json` files, the slide file is ignored.

### Segmentation Parameters (optional)

| Flag | Default | Description |
|---|---|---|
| `--threshold_nuclei` | `200` | Pixel intensity cutoff for nuclei detection |
| `--minsize_nuclei` | `20` | Minimum nuclei object size (pixels) |
| `--threshold_PAS` | `50` | Pixel intensity cutoff for PAS/eosinophilic compartment |
| `--minsize_PAS` | `20` | Minimum PAS object size (pixels) |
| `--threshold_LS` | `0` | Pixel intensity cutoff for luminal space |
| `--minsize_LS` | `0` | Minimum luminal space object size (pixels) |

---

## Outputs

All outputs are written to `--output_dir`:

```
output/
  tubules_Features.xlsx                          # Per-element features, one sheet per category
  non_globally_sclerotic_glomeruliFeatures.xlsx
  arteries_arterioles_Features.xlsx
  ...
  metadata.json                                  # Aggregated slide-level statistics
  sub_compartment_params.json                    # Segmentation parameters used
  annotations_updated.json                       # Original annotations enriched with per-element feature values
```

Each Excel file has one sheet per feature category (`Distance Transform Features`, `Color Features`, `Texture Features`, `Morphological Features`, `Bounding Boxes`), with one row per FTU element.

---

## Repository Structure

```
CombinedFeatureExtraction/
  fextract/
    cli/
      PathomicsFE/
        PathomicsFE.py          # DSA/Girder entry point (ctk_cli)
        PathomicsFE.xml         # Slicer CLI descriptor for HistomicsUI
        PathomicsFELocal.py     # Local entry point (argparse, no Girder)
      docker-entrypoint.sh      # DSA container entrypoint (slicer_cli_web)
    extractioncodes/
      FeatureExtractor.py       # Feature extraction class (Girder-coupled)
      LocalFeatureExtractor.py  # Feature extraction class (Girder-free)
      run_pathomic_fe.py        # Classical pathomic feature runner
    extraction_utils/           # Shared image processing utilities
  Dockerfile                    # DSA/HistomicsUI image
  Dockerfile.notebook           # Local/notebook image (no Girder)
  setup.py
```

---

## References

1. Lucarelli N. et al. *Correlating Deep Learning-Based Automated Reference Kidney Histomorphometry with Patient Demographics and Creatinine.* Kidney360 4(12):1726–1737, December 2023. DOI: [10.34067/KID.0000000000000299](https://doi.org/10.34067/KID.0000000000000299)
