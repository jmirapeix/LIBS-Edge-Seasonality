# Edge-referenced analysis of 2D LIBS Mg/Ca maps of limpet shells

Code and data accompanying the paper:

> Mirapeix, J., Arniz-Mateos, R., García-Escárzaga, A., Gutiérrez-Zugasti, I., López-Higuera, J.M., Cobo, A.
> *Edge-referenced analysis of 2D LIBS Mg/Ca maps for automated seasonality estimation in limpet shells* (under review).

The script `mtp_v51.py` estimates the season of collection of a limpet shell
(*Patella vulgata*) automatically from a two-dimensional Mg/Ca map obtained by
Calibration-Free LIBS. It does not need any manually defined sampling trajectory:

1. **Final growth edge.** The final growth edge is delimited from the morphology of the shell section.
2. **Edge-referenced map.** The map is reorganised according to the distance to that edge, `I(s, d)`.
3. **Calibration interval.** The most recent seasonal cycle next to the edge is detected and used to obtain sample-specific low/high Mg/Ca references (P15/P85).
4. **Season estimate.** The season is assigned from the Mg/Ca level at the edge and its trend towards the edge:
   - winter: level ≤ 0.25;
   - summer: level ≥ 0.75;
   - otherwise spring (increasing towards the edge) or autumn (decreasing).
5. **Uncertainty.** It is assessed with a block bootstrap over the rows of the edge-referenced map, the sensitivity to the first edge pixels and an edge-quality index.

The code is commented, and each section refers to the corresponding section of the paper.

## Contents

| Path | Description |
|---|---|
| `mtp_v51.py` | Complete processing code (single file) |
| `data/` | 2D Mg/Ca maps (`.mat`) of the 30 specimens analysed in the paper: 24 modern specimens from Langre Beach (Cantabria, Spain) and 6 archaeological specimens from El Toral III cave (Asturias, Spain) |
| `requirements.txt` | Python dependencies |

## Installation

Python ≥ 3.9 is required:

```bash
pip install -r requirements.txt
```

## Reproducing the results of the paper

```bash
python mtp_v51.py --batch
```

This command processes the 30 specimens with the parameters of the paper and writes to `outputs/`:

- **`results.csv`**: one row per specimen. It contains:
  - the season estimate;
  - the edge value, the references and the relative edge level;
  - the trend towards the edge;
  - the estimates obtained excluding 1 and 2 edge pixels;
  - the bootstrap probabilities of the four seasons;
  - the edge-quality index.

  These are the values reported in Tables 4 and 5.
- **`metrics.json`**: the performance metrics of the modern specimens, each with its chance level, plus all the parameters used.
- **`profiles/`**: the median Mg/Ca(d) profile of each specimen.
- **`figures/`**: the per-specimen figures (Figs. 7–14 and S1–S30), the method figure (Fig. 5), the bootstrap summaries (Figs. 6 and 11) and the sensitivity figure (Fig. 15), as PDF, PNG and SVG files.

The full run takes about 3–5 minutes on a standard computer, because each
bootstrap repeats the complete classification 200 times per specimen. Specimens
are processed in parallel. Useful options:

```bash
python mtp_v51.py --batch --boot 50            # quicker test run (bootstrap probabilities will differ slightly)
python mtp_v51.py --batch --only LAN230 LAN237 # subset of specimens
```

The bootstrap uses a fixed random seed, so the results are exactly reproducible.

## Analysing a new map

```bash
python mtp_v51.py path/to/map.mat --edge-left    # collection edge on the left of the map
python mtp_v51.py path/to/map.mat --edge-right   # collection edge on the right (the map is flipped)
python mtp_v51.py                                # file dialog and questions
```

The estimate, the bootstrap probabilities, the sensitivity to the edge pixels and
the edge-quality index are printed, and the figure is saved to `outputs/figures/samples/`.

### Input format

Each `.mat` file must contain:

- `MgCa_map`: 2D array with the Mg/Ca ratio (mmol/mol) and `NaN` outside the shell section. In the paper the pixel size is 50 µm.
- `mask` (optional): boolean array of valid shell pixels. If it is missing, all non-`NaN` pixels are used.

Rows of the array are the y coordinate and columns the x coordinate. Maps are
displayed with the origin at the bottom-left corner. The orientation of the
specimens of the paper (collection edge on the left or on the right of the stored
map) is fixed in the `SAMPLES` table of the script. Do not use the
`apex_right_edge_left` metadata stored in some of the `.mat` files: it is not reliable.

## License

- Code: MIT (see `LICENSE`).
- Data (`data/`): CC BY 4.0.

If you use them, please cite the paper above.
