# NV magnetic-sensing benchmarks

Code for comparing CNN and compact MLP regressors on the Weng NV-array dataset,
evaluating standardisation on the Meneses elevator dataset, and measuring CPU
inference latency and process memory with PyTorch, NumPy and ONNX Runtime.

This repository accompanies *Beyond Parameter Counts: Standardisation, Nested Model
Selection, and Inference Backends in Neural Regression for NV Magnetic Localisation*.
Version v1.0.0 identifies the code and result records distributed with this study.
The software citation in CITATION.cff is separate from the manuscript author list.

## Quick start

Clone the repository, then retrieve the two large prediction tables:

~~~shell
git clone https://github.com/zzzzw1123/nv-magnetic-localisation.git
cd nv-magnetic-localisation
python -B scripts/fetch_release_data.py
python -B scripts/verify_results.py
~~~

Both commands use only the Python standard library. The download command checks
the archive and table SHA-256 checksums before restoring the original relative
paths. It leaves any existing, different file unchanged. The verification command
reads the stored experiments; it performs no training or resource measurements.
Install the dependencies below to run inference examples, analyses or plots.

## Layout

```text
src/qml_microrobot/                       Models and data loaders
scripts/                                 Compact-candidate and linear-reference runners
paper/figures/scientific_controls_20260912/ Experiment, analysis and plotting scripts
execution_sources/                       Source versions used for the recorded experiments
figure_data/                             Supplementary figure source tables
release_manifest.json                    Release file sizes, checksums and source categories
requirements.txt                         Recorded package versions
LICENSE                                  MIT licence for project code
```

The `qml_microrobot` package contains the models and data loaders used here.
The included NV regressors are classical neural and linear models.

## Training and evaluation

The experiment scripts are in `paper/figures/scientific_controls_20260912/`.

| Scripts | Purpose |
| --- | --- |
| `recipe_factorial.py` | Pilot fits for the input/target standardisation comparison |
| `recipe_full.py`, `recipe_additional.py` | Three initial seeds and seven additional seeds, with 1,500- and 3,000-epoch checkpoints |
| `nested_loao.py`, `nested_outer_additional.py` | Inner model selection and outer acquisition-level evaluation |
| `meneses_external.py` | Task-specific CNN and MLP fits on the independent Meneses dataset |
| `meneses_label_audit.py` | Evaluation-target checks and the boundary-target sensitivity analysis |

Weng inputs have 24 log-transformed spectral features and two position targets in
micrometres. Standardisation is fitted on training data. The main comparison
averages ten seeds within each acquisition before averaging the eight acquisitions.
The two epoch checkpoints come from the same training trajectory.

Training runners write results beside their scripts and check source fingerprints
before resuming. For new fits, use a separate working copy without the released
prediction CSVs or nested-selection output in its experiment directory. The
recipe sequence is `recipe_factorial.py run`, `recipe_full.py run`, then
`recipe_additional.py run`. New fits record the source fingerprint of that
working copy; they must not be appended to the historical CSVs. These are training
commands, not installation checks.

The Meneses analysis uses 40-sample, three-channel windows and position targets
in metres. Its three-seed summaries use two held-out segments and the released
floor-valued target subset.

## Result analysis and figures

`recipe_analysis.py` calculates acquisition-level summaries and paired bootstrap
intervals. `summarize_backend_records.py` aggregates recorded process measurements.
Both write output files when run as scripts.

| Scripts | Figures |
| --- | --- |
| `plot_recipe_factorial.py` | Standardisation-dependent architecture gaps |
| `plot_nested_external.py` | Nested selection and the Meneses comparison |
| `plot_overview_resource.py` | Inference latency versus process RSS |
| `plot_supplement_reference.py` | Supplementary diagnostics |

Plotting scripts use `--preview` for PNG previews and `--approved-export` for
figure exports. The first three export figures when run without arguments;
`plot_supplement_reference.py` defaults to PNG previews. Outputs go to
`outputs/previews/` or `outputs/figures/` under the code root. Repeated runs
overwrite figures with the same filename.

The plotting scripts accept these environment variables for data stored
outside the code directory:

| Variable | Location | Default |
| --- | --- | --- |
| `NV_PROJECT_DIR` | Project root containing the Sparse16 `results/` directory | Code root |
| `NV_RESULTS_DIR` | Recipe, nested, Meneses and ONNX result records | Experiment script directory |
| `NV_FIGURE_SOURCE_DIR` | Supplementary figure CSVs | `figure_data/` under the code root |
| `NV_OUTPUT_DIR` | Parent directory for `figures/` and `previews/` | `outputs/` under the code root |

For example, set `NV_RESULTS_DIR` to the directory containing the saved
experiment records, then run `plot_recipe_factorial.py --preview` from the
experiment script directory. These variables configure figure inputs and
outputs; training and benchmark scripts retain their recorded project layout.
The current figure previews require Arial.

Earlier three-seed analyses are retained in `matched_controls.py`, the
`coverage_*.py` scripts and `intervention_cnn_fixed_inputs.py`. Their companion
validation scripts check the corresponding saved records. These analyses are
separate from the ten-seed standardisation experiment.

## Inference benchmarks

`portable_inference.py` implements NumPy inference from saved weights and scaling
parameters. `portable_export_validate.py` checks coordinate predictions against
PyTorch. `onnx_benchmark.py` measures all three backends in isolated CPU processes
at batches 1 and 256. Latency is measured per API call; RSS includes runtime
dependencies and measurement tools.

The original measurements used Windows, an Intel Core i9-14900HX, one numerical
thread and CPU affinity `[12]`. Resource measurement scripts target Windows.
Benchmark runs should be separate from training
and other CPU-intensive work. The `memory_*.py` and `portable_benchmark*.py`
scripts contain the earlier memory and two-backend measurements.

The three-backend benchmark checks Python 3.9.13 and the recorded package
versions, including the `2.8.0+cpu` PyTorch build.

The resource benchmark uses saved CNN and MLP8 models fitted on acquisitions
A1–A5, separate from the ten-seed accuracy experiments.

## Environment and inputs

The experiment environment used Python 3.9.13. `requirements.txt` lists the
installed distribution versions; the PyTorch runtime was the CPU build
`2.8.0+cpu`.

From the code root, create a Python 3.9 environment and install the CPU build of
PyTorch before the other dependencies. These commands use Windows PowerShell:

```powershell
py -3.9 -m venv .venv
.\.venv\Scripts\python.exe -m pip install pip==25.3
.\.venv\Scripts\python.exe -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu --no-deps
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip check
```

Use this environment's Python interpreter for the examples below. On Linux,
the corresponding interpreter is `.venv/bin/python`.

Setup was tested on Windows with Python 3.9.13 and pip 25.3. The installation
check used the Tsinghua PyPI mirror for the requirements command:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt --index-url https://pypi.tuna.tsinghua.edu.cn/simple
```

The repository and its versioned data attachment provide the inputs, fitted states
and saved predictions needed for result checks and plots. Their relative locations are:

- `study_l/` and `results/`: Weng source inventories, acquisition mappings and
  analysis records.
- `checkpoints/`: saved CNN and MLP8 reference states.
- The experiment directory: recipe CSVs, nested-selection records, Meneses
  segments, benchmark inputs and exported NumPy/ONNX models.
- Configuration files: the compact-selection protocol under `study_l/`, and
  `memory_protocol.json` and `portable_protocol.json` in the experiment directory.

`release_manifest.json` lists the included files and their SHA-256 checksums.
The source inventories also describe archive members outside this benchmark;
only the input files listed in the release manifest are included.

Weng measurements originate from the
[published data release](https://doi.org/10.5281/zenodo.19855773).
The Meneses data and reference code use
[source commit 7e8a78a](https://github.com/Fertmeneses/ML_QDM_Meneses_et_al/tree/7e8a78a33078a1479fe66b9c73bf3c6d7e14ca02).
Source and licence details are in [THIRD_PARTY_NOTICES](THIRD_PARTY_NOTICES).
The approximately 113 MiB Meneses prediction table and 39 MiB additional-seed
recipe table are distributed in the nv-results-v1.0.0.zip asset of
[release v1.0.0](https://github.com/zzzzw1123/nv-magnetic-localisation/releases/tag/v1.0.0).
Run the quick-start download command once after cloning, or download the asset
separately and pass its path with --archive. All other manifest files are tracked
in Git. Original datasets retain the source licences; generated predictions use
CC BY 4.0. A Zenodo DOI has not yet been assigned.

## NumPy inference example

`portable_mlp8.npz` and `memory_input_features.npy` are included in the experiment
directory. Run the following commands from the code root. Downloading the source
dataset alone does not provide these fitted-model outputs.

```shell
cd paper/figures/scientific_controls_20260912
python portable_inference.py --help
python portable_inference.py --model mlp8 --weights portable_mlp8.npz --input memory_input_features.npy --output predictions.npy --batch-size 256
```

The example writes two files:

- `predictions.npy`: 256 rows of X/Y predictions in micrometres, with shape
  `(256, 2)` and dtype `float64`.
- `predictions.json`: model, input-row and batch counts, NumPy version,
  numerical checks and a checksum of the prediction array.

This example has been checked in a clean Python 3.9.13 environment. It runs
inference only; it does not train a model or measure latency or memory.
Load only trusted model files whose checksums match the release manifest;
some training-state files use Python pickle serialization.

## Result compatibility

The scripts under `src/`, `scripts/` and `paper/` are the reading and working
copies. `execution_sources/` retains the exact source bytes used to produce the
historical records, including their original comments and metadata. Changes to
comments or docstrings change a file's checksum even when its computation is
unchanged. Historical hashes have not been replaced with hashes of edited copies.

From the code root, run:

```shell
python -B scripts/verify_results.py
```

This standard-library command checks the manifest, matches recipe, nested,
Meneses and resource records to their archived source versions, checks prediction
coverage, recomputes Euclidean or absolute errors and verifies the main summary
values. It averages seeds within acquisitions or segments before the corresponding
group means. Resource summaries are reconstructed from the existing process
records. The separate earlier 54-fit matched-training records are also checked
against their archived sources, predictions and eight historical summary rows;
they are not merged with the ten-seed experiment. Other supplementary record
files receive manifest checksum checks. It writes no files and runs no training, inference or performance
measurements. The stored bootstrap intervals pass file-checksum checks; the
command does not regenerate their random resamples.

For file checksums alone, use `python -B scripts/verify_results.py --files-only`.
The earlier experiment-specific validators retain their exact-version checks;
running them against the edited working copies and historical outputs will reject
the source mismatch. Use the release verifier for the distributed records.

## Licence

Project code is licensed under the [MIT License](LICENSE), copyright
2026 Haowen Zhu. Author-generated predictions, summaries, figure-source tables,
features and fitted-model outputs are licensed under CC BY 4.0 as specified in
[THIRD_PARTY_NOTICES](THIRD_PARTY_NOTICES). Third-party datasets and other
materials retain their original licences and attribution requirements.
