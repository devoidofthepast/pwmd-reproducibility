# PWMD reproducibility package

Code, trained models, and per-case results for the paper

> **PWMD: Structured World-Model Planning with Learned Nomination for Dynamic Multi-Objective Flexible Job Shop Scheduling with AGVs**

The package contains:
- the SU-MODFJSP-HA-PC instance generator and shop simulator;
- the PWMD proposer and the scenario rollout certificate;
- the training scripts and the trained models;
- the compared methods: dispatching rules, DANIEL, DRL-S, HGT, DQN, GA, NSGA-II, NSGA-III, and MOEA/D;
- the CP-SAT validation model;
- the raw per-case records behind every table and result figure;
- the scripts that rebuild the tables and figures from these records.

## 1. Layout

| Folder | Content |
|---|---|
| `rerun_twoleg_early/src/su_modfjsp_ha_pc/` | Generator, simulator, objectives, base rule and transport rule, proposer data and model, training entry points |
| `rerun_twoleg_early/baselines/` | Scenario rollout certificate (`scenario_cert3.py`), dispatching rules, DRL adapters and training, DQN pipeline, evolutionary rescheduling, strict online execution |
| `rerun_twoleg_early/test_v3/` | Test-set runs (`test_v3.py`, `review_r2.py`, `test_v3_rnd.py`), weight-vector fronts (`front_v3.py`), timing (`timing_v3.py`); results in `runs/`, `front/`, `timing/` |
| `rerun_twoleg_early/dev_v3/` | Development runs: selection of *M*, *K*, and *k*; sensitivity analysis; evolutionary-algorithm budget; results in `runs_*` |
| `rerun_twoleg_early/strict_online/` | Strict online execution (Section 5.4); results in `runs/` |
| `rerun_twoleg_early/models_*`, `scales_*` | Trained proposer ensembles (20 and 5 training seeds), ablation proposers, objective scales |
| `rerun_twoleg_early/drl_v3/` | Selected DANIEL, DRL-S, HGT, and DQN models with training logs |
| `twoleg/` | CP-SAT validation (`cpsat_validate_early.py`) and its results (`cpsat_spfix_main/`) |
| `manuscript/` | Table and figure scripts; `published_tables/` holds the tables as printed in the paper |
| `external_baselines/` | Empty; see Section 3 |

## 2. Names used in the code

| Code | Paper |
|---|---|
| `pgnd` | PWMD |
| `rule-lookahead` | G-ECT (base rule) |
| `pgnd-nocert` | w/o certificate |
| `nocert-gated` | Gated w/o certificate |
| `rnd`, `heu-ect`, `heu-cost` | Random, ECT, and cost proposers |
| `abl-ridge`, `abl-context-only`, `abl-no-route-event` | Ridge, context-only, and w/o route-event proposers |
| `rule-*` | The other dispatching rules |
| `ref-*-known` | Reference runs with event outcomes known to the simulator (Table 12) |
| `pgnd-mis-*` | Mismatched scenario samplers (Fig. 9c) |
| `*-eq23` | DRL-based methods with the machine chosen by the transport rule |
| `L4` / `L5` | Moderate / severe event setting |
| `raw` / `realized` | With resource events only / with all events |
| `S`, `C`, `L` (blocks) | Synthetic test set (seeds 800–819); Mk1 and Edata7 (820–829); larger instances (830–839) |
| `syn`, `mk1`, `edata7` | Synthetic, Mk1, and Edata7 instance sets |

## 3. Environment

- Python 3.12.6
- PyTorch 2.5.1
- OR-Tools 9.15
- Other packages: see `requirements.txt`.

All timings in the paper were measured on one thread of an Intel Core i9-13900H CPU.

```
pip install -r requirements.txt
```

The DRL baselines and the Mk1 and Edata7 production cores use the original public repositories. Clone them into `external_baselines/` at the listed commits:

| Folder name | Repository | Commit |
|---|---|---|
| `FJSP-DRL-DANIEL` | https://github.com/wrqccc/fjsp-drl | `2cf81b13f5044451e78cf780f8fb3e7eeac054c1` |
| `fjsp-drl-song` | https://github.com/songwenas12/fjsp-drl | `d7637b70029be6cb8fe78e450dcb07767876daea` |
| `MDFJSSP-HGT` | https://github.com/LiaoMengqi/MDFJSSP-HGT | `a681d7f86d6f3f18637fb3cc0a51c2d322742daa` |

The Brandimarte Mk1 and Hurink Edata7 files are read from `FJSP-DRL-DANIEL/data/BenchData/`.

The evolutionary algorithms use pymoo. If pymoo is installed in a separate folder, put that folder on the path as `external_baselines/_vendor_py`.

All scripts are run from the folder that contains them.

## 4. Rebuilding the tables and figures from the stored results

These steps need only the stored records and take a few minutes.

```
cd manuscript
python make_tables_v9.py          # Tables 7, 8, 9, 11, 12 (before the final caption edits)
python make_ea_pref_table_v9.py   # per-preference block of Table 8
mkdir figures
python make_figs_r2.py            # Figs. 4, 5, 6, 8, 10
python make_figs_wm.py dist sens  # Figs. 7, 9
```

The generated tables contain the same numbers as `published_tables/`. The captions, column labels, and setting names were edited by hand afterwards.

The following rows and tables come from other analysis scripts:
- the Gated, ECT-proposer, and cost-proposer rows of Table 11 and the random-proposer row on the larger instances: `test_v3/analyze_r2_gate_mis.py`, `test_v3/analyze_r2_heu.py`, and `test_v3/analyze_r2_heuC_rndL.py`;
- Table 10 (fronts): `test_v3/analyze_front_v3.py`;
- Table 13 (computing times): from `test_v3/timing/`;
- Table 2 (CP-SAT): from `twoleg/cpsat_spfix_main/` (summary in `twoleg/cpsat_spfix_main_summary.txt`).

Figs. 1–3 are schematic diagrams and are not generated from results.

## 5. Re-running the experiments

Every run script resumes from its output file and skips records that already exist. To re-run a block from scratch, move its stored records away first.

| Step | Command (from the folder named) | Output |
|---|---|---|
| Objective scales and training labels | `rerun_twoleg_early`: `python run_labels_v3.py WORKERS` (seeds 600–619, random events through `v3_random_events.py`) | `labels_train_600_619_v3/` (not included, about 130 MB) |
| Proposer training | `rerun_twoleg_early/src`: `python -m su_modfjsp_ha_pc.run_pgnd_v4_train --context-dir ../labels_train_600_619_v3 --output-dir ../models_physics_train_600_619_v3 --start-seed 600 --end-seed 619 --objective-scales ../scales_train_600_619_v3/objective_scales.json` | 5 members, random states 320–324 |
| Ablation proposers | `run_pgnd_v5_ablation_train` and `run_v3_no_route_event_train` with the same arguments, output `models_ablation_train_600_619_v3` | ridge, context-only, w/o route-event |
| DRL baselines | `rerun_twoleg_early/baselines`: `python train_daniel_v3.py`, `train_song_v3.py`, `train_hgt_v3.py`, `cheng_pipeline_v3.py` | `drl_v3/` |
| Development runs | `rerun_twoleg_early/dev_v3`: `python dev_v3.py SET SEED ARMS`, `run_Kk.py`, `run_Kk_fresh.py`, `run_sens.py` | `runs_M*`, `runs_Kk*`, `runs_sens` |
| Test runs | `rerun_twoleg_early/test_v3`: `python test_v3.py BLOCK SEED GROUP` with GROUP in `rules`, `pgnd`, `drl`, `ablation`, `reference`, `ea`; `python review_r2.py BLOCK SEED GROUP` for `drl23`, `gate`, `heu`, `heuL`, `rndL`, `mis`; `python test_v3_rnd.py` for the random proposer | `runs/S`, `runs/C`, `runs/L` |
| Fronts | `rerun_twoleg_early/test_v3`: `python front_v3.py SEED det` and `python front_v3.py SEED ea --method M --run R` | `front/` |
| Strict online execution | `rerun_twoleg_early/strict_online`: `python run_strict.py SEED` | `runs/` |
| Timing | `rerun_twoleg_early/test_v3`: `python timing_v3.py BLOCK SEED GROUP` on an idle machine, one thread | `timing/` |
| CP-SAT validation | `twoleg`: `python cpsat_validate_early.py --seeds 400-409 --jobs 4 --levels L4,L5 --limit 300 --workers 8 --out OUT --cpsat-only` | one JSON file per case |

The PWMD settings used in all test runs are in `rerun_twoleg_early/v3_final_config.json`: *M* = 0.02, *K* = 8, and *k* = 2.

**Quick check.** Re-run the first cases of seed 800 and compare them with the stored records. For example, PWMD and G-ECT on the first synthetic case with the balanced preference give *V* = 0.382121 and 0.403055. These are the values in `test_v3/runs/S/pgnd_800.jsonl` and `rules_800.jsonl`, and the case shown in Fig. 5.

## 6. Notes

- Absolute folder names from the original machine appear in some log and manifest fields. They are provenance only and are not read by the code. The objective-scale file is checked by its SHA-256 hash, so it is kept unchanged.
- Some docstrings refer to internal protocol documents by file name. These documents are not part of the package. The settings they fixed are stated in the paper and in `v3_final_config.json`.
- `cpsat_validate_early.py` can also decode two dispatchers with an earlier proposer for comparison. That comparison is not reported in the paper, and `--cpsat-only` skips it. Table 2 uses only the CP-SAT status, bound, objective, and simulator check of each case.
- The `drl_v3/cheng/` folder contains the full grid-search runs used to select the DQN settings.

## 7. Citation

If you use this package, please cite the paper above. The citation details will be added after publication.
