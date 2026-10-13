# Empatia_Uncertainty

**Graph-based probabilistic affective assessment with contrastive learning and uncertainty-aware ensembles.**

Empatia_Uncertainty contains research code for predicting continuous affective scores from wearable data and assessing the reliability of those predictions. The pipeline combines unified physiological graphs, self-supervised contrastive encoding, Gaussian regression, and an ensemble of three independently trained assessment models, referred to as **virtual doctors**.

This repository corresponds to **Chapter 6 — Beyond Hard Emotion Recognition: Fine-Grained and Uncertainty-Aware Affective Assessment** of Junjiao Sun's doctoral thesis, *Toward Practical and Uncertainty-Aware Affective Computing with Wearable Physiological Signals*.

The chapter extends the representation-learning work in [Empatia_FLAGS](https://github.com/des-cei/Empatia_FLAGS) from categorical emotion recognition to continuous assessment with uncertainty estimates. The supplied scripts primarily implement the **MMASH** workflow. The chapter's WESAD experiments are not provided as a complete dataset-specific pipeline here.

> **Implementation status:** This repository contains development-time research scripts. Raw data, generated JSON files, encoder checkpoints, and experiment logs are not included. Several imports refer to directories from the original project layout, and the contrastive-loss implementation is missing. The preparation and compatibility notes below are required before end-to-end execution. `main.py` is a CUDA sparse-solver demonstration, not the entry point for the assessment pipeline.

## Method overview

The framework addresses two questions: **What affective score does the model predict?** and **How reliable is that prediction?**

1. Align Actigraph, RR-interval, and activity recordings and summarize them into minute-level descriptors.
2. Build a unified graph for each time window, connecting descriptor nodes through aggregate nodes.
3. Learn graph representations from original and augmented views using a contrastive objective.
4. Freeze the encoder and train assessment-specific probabilistic decoders.
5. Combine virtual-doctor predictions to estimate data uncertainty and model disagreement.
6. Compare those uncertainty scores with calibration-derived thresholds to assign a reliability category.

The regression targets are positive and negative affect scores from PANAS questionnaires. The current MMASH experiment uses five assessment slots: `10`, `14`, `18`, `22`, and `33`, where `33` represents 09:00 on the following day. Two affect dimensions and five slots give **ten decoders per doctor per held-out-user fold**.

“Virtual doctor” is the name of an ensemble member. The code's diagnosis and severity labels describe experimental reliability rules, not validated clinical diagnoses.

## Graph representation and contrastive encoder

### Data representation

The extraction functions reconstruct second-level streams, adjust timestamps across midnight, and fill missing intervals according to their individual implementations. `Merge_data.py` restricts the streams to their common time range. `Reduce_dim_min.py` trims to whole-hour boundaries and calculates minute-level summaries.

The full minute representation contains 21 input values: 18 Actigraph-derived descriptors, mean RR interval, activity code, and a timestamp. `graph_gen` adds an Actigraph aggregate node and a global aggregate node, producing **23 nodes**, each with a scalar feature. Edges point from aggregate nodes to their children; reverse edges are not explicitly added.

A reduced variant removes steps and four inclinometer descriptors. It retains 16 input values and adds two aggregate nodes, producing **18 nodes** through `graph_gen_less`. Keep the full and reduced graph caches separate; they are distinct experiment configurations.

The timestamp is included as a graph input in the checked-in construction. Record this choice when comparing physiological-only models or transferring between recording schedules.

### Self-supervised learning

`Graph_encoder_loso.py` defines a graph encoder with:

| Stage | Implementation |
| --- | --- |
| Graph convolution | `GCNConv(input_dim, 64)` → ReLU → `GCNConv(64, 64)`. |
| Graph pooling | Global mean pooling over nodes. |
| Projection | Linear `64 → 128` → ReLU → linear `128 → 64`. |
| Output | One 64-dimensional representation per graph. |

The training code groups four aligned views: original, Gaussian-noise, jitter, and white-noise graphs. Original-to-augmentation pairs are designated as positives. It calls `nt_bxent_loss`, imported from the absent `Map_process.Map_encoder` module. Restore the intended loss, including its temperature and positive-pair semantics, before running pretraining; the exact objective cannot be reconstructed from this checkout alone.

The standalone LOSO encoder driver defaults to 100 epochs with Adam at `1e-3`. It randomly selects three other users for training and uses held-out-user graphs for evaluation and checkpoint selection. This differs from the chapter's target-independent calibration protocol and from doctor-specific pretraining; see **Evaluation and reproducibility**.

## Probabilistic regression

`RegressionCNN` maps a 64-dimensional graph representation to a Gaussian distribution:

- Three 1D convolution blocks with 16, 32, and 64 channels, batch normalization, and GELU.
- Adaptive average pooling and dropout with probability `0.3`.
- Separate linear heads for the mean and standard deviation.
- `softplus` plus `1e-6` to keep the predicted standard deviation positive.

The wrapper `GCN_encoder_regression` freezes encoder parameters and evaluates the encoder inside `torch.no_grad()`. The decoder returns **`mu` and `sigma`**, with `sigma` denoting standard deviation, not variance.

Training minimizes Gaussian negative log-likelihood:

$$
\mathcal{L}_{\mathrm{NLL}}
=\frac{1}{N}\sum_{i=1}^{N}
\left[\frac{1}{2}\log(2\pi\sigma_i^2)
+\frac{(y_i-\mu_i)^2}{2\sigma_i^2}\right].
$$

The ensemble drivers use Adam at `1e-3`, up to 1,000 epochs per decoder, training batches of 128, and test/calibration batches of 24. Settings are edited in source; there is no unified command-line configuration.

## Virtual-doctor ensemble and triage

For each held-out target user, the experiment excludes that user and divides the remaining population into a calibration group and three doctor-training groups. With the intended 20-user cohort, the splitter selects four calibration users and three disjoint groups of five training users. Its partition seed is derived from the target ID.

`Ensemble_Doc_1.py`, `Ensemble_Doc_2.py`, and `Ensemble_Doc_3.py` differ only in their `Current_Doc` setting. Each loads an externally prepared doctor/fold-specific encoder and trains the ten assessment decoders.

For predictions from three doctors, the ensemble quantities are:

$$
\bar\mu=\frac{1}{3}\sum_{r=1}^{3}\mu_r,
\qquad
U_{\mathrm{alea}}=\frac{1}{3}\sum_{r=1}^{3}\sigma_r^2.
$$

The default diagnostic mode measures disagreement around the median:

$$
U_{\mathrm{epi}}=\frac{1}{3}\sum_{r=1}^{3}
\left(\mu_r-\operatorname{median}(\mu_1,\mu_2,\mu_3)\right)^2.
$$

An alternative `average` mode uses variance around the mean. Use the same mode for calibration and target evaluation.

`Read_threshold.py` estimates separate percentile thresholds for each target fold, assessment slot, and affect dimension from calibration-user predictions. `Diagnose_Docs.py` selects the **80th percentile**; the threshold function's default is 90 unless overridden.

| Case | Rule, in priority order | Interpretation |
| --- | --- | --- |
| Case 2 | Epistemic score exceeds its threshold. | High disagreement; refer for expert review. |
| Case 1 | Epistemic score is within threshold, but aleatoric score exceeds its threshold. | Data-complexity warning. |
| Case 0 | Both scores are within their thresholds. | Accepted assessment under the configured policy. |

The code labels these cases `System Failure (High Disagreement)`, `Data Complexity (High Uncertainty)`, and `Trusted Diagnosis`. Case 2 takes precedence when both uncertainty scores are high.

The current diagnostic utilities operate on **user/slot-level summaries parsed from logs**, rather than a stream of individual graph predictions. In particular, they square the logged mean standard deviations. This is not generally equal to averaging the original per-graph variances; retain the intended aggregation level when reproducing the thesis equations.

## Repository guide

| Files / directory | Purpose |
| --- | --- |
| `Actigraph_extract.py`, `RR_extract.py`, `Activity_extract.py` | Read and reconstruct MMASH streams. |
| `Merge_data.py`, `Reduce_dim_min.py` | Align streams and create minute-level descriptors. |
| `Signal_augmentation/` | Gaussian, jitter, and white-noise signal transformations. |
| `Graph_generation.py` | Construct full and reduced physiological graphs. |
| `Json_process/Pack_json_graph.py`, `Pack_json_graph_less.py` | Serialize graph collections. |
| `Json_process/Pack_json_label.py` | Extract questionnaire values into a label JSON file. |
| `Graph_encoder.py`, `Graph_encoder_less.py`, `Graph_encoder_loso.py` | Contrastive graph-encoder experiments. |
| `Graph_NLL.py` | Single-model probabilistic regression experiment. |
| `Ensemble_Doc_1.py`, `Ensemble_Doc_2.py`, `Ensemble_Doc_3.py` | Virtual-doctor regression experiments. |
| `Read_Doc.py`, `Read_threshold.py`, `Diagnose_Docs.py` | Parse predictions, calibrate thresholds, and assign triage categories. |
| `Doc_plot.py`, `Doc_ensemble.py`, `Prediction_plots.py`, `Plot_generation.py` | Prediction and comparison plots. |
| `Graph_classification.py`, `Graph_classification_2C.py` | Auxiliary classification experiments. |
| `Analyse_Accuracy.py`, `Analyse_Normalization.py`, `P_value_calculate.py` | Legacy analysis utilities. |

## Setup and required inputs

```bash
git clone https://github.com/des-cei/Empatia_Uncertainty.git
cd Empatia_Uncertainty
python -m venv .venv
source .venv/bin/activate
export EMPATIA_UNCERTAINTY_ROOT="$PWD"
export PYTHONPATH="$EMPATIA_UNCERTAINTY_ROOT${PYTHONPATH:+:$PYTHONPATH}"
```

Install compatible PyTorch and PyTorch Geometric packages for your machine. Additional dependencies inferred from imports include:

```bash
python -m pip install numpy scipy pandas scikit-learn matplotlib fastdtw
```

Some analysis utilities also import `dtw`; check the required API before selecting a compatible distribution. No pinned environment is provided. Core model scripts choose CUDA when available and otherwise CPU; `main.py` separately requires CUDA for its unrelated demonstration.

### Raw recordings

Supply the MMASH participant folders with these files:

| Example path | Content |
| --- | --- |
| `dataset/user_1/Actigraph.csv` | Accelerometry, activity-monitor channels, day, and time. |
| `dataset/user_1/RR.csv` | Inter-beat intervals, day, and time. |
| `dataset/user_1/Activity.csv` | Activity intervals. |
| `dataset/user_1/questionnaire.csv` | PANAS and questionnaire values. |

Readers select CSV fields by column position. Preserve the expected source layout or adapt the readers explicitly. The experiment excludes users 2 and 8 for incomplete annotations; some earlier scripts exclude only user 2, so use one documented cohort consistently.

### Generated JSON and model artifacts

The ensemble workflow expects:

| Artifact | Format / role |
| --- | --- |
| `Json_files/graph_normal.json` | Dictionary keyed by string user IDs, each mapping to an ordered list of graph dictionaries with `x` and `edge_index`. |
| `Json_files/graph_Gaussian.json` | Gaussian-augmented graphs with matching user/frame order. |
| `Json_files/graph_Jitter.json` | Jitter-augmented graphs with matching user/frame order. |
| `Json_files/graph_White.json` | White-noise-augmented graphs with matching user/frame order. |
| `Json_files/Label.json` | User dictionaries with keys such as `panas_pos_10`, `panas_neg_10`, and corresponding later slots. |
| Doctor-specific encoder directories | State dictionaries named, for example, `user_1_Doc1_encoder.pth`. |

The label packer replaces `9+1` in questionnaire column names with `33`. It reads the first questionnaire row and columns 1–19. Verify the resulting PANAS keys and units before training.

`Pack_json_graph.py` currently writes the original full-graph cache. Generate the other three full-graph views by explicitly selecting the matching augmentation readers and output filenames. `Pack_json_graph_less.py` currently writes only the reduced jitter cache; it does not create the full four-view set automatically.

## Required integration adjustments

| Issue | Action before running |
| --- | --- |
| Original project layout | Replace imports under absent `Graph_process`, `Map_process`, `Ensembles_1215`, and `Ensembles_Threshold` packages with the intended local modules, or restore the original package structure. Restore the missing contrastive loss separately. |
| Path inconsistencies | Resolve all dataset, JSON, checkpoint, and log paths against a shared project root. Root-level scripts often use `../Json_files` or `../dataset`, while packers use different working-directory assumptions. |
| Missing doctor encoders | Generate each encoder using the correct doctor-training subjects and fold, excluding calibration and target users. The generic encoder driver writes `user_<ID>_weights.pth`; those files are not a documented substitute for the doctor-specific checkpoints expected by the ensemble scripts. |
| Output directories | Create JSON, encoder, `Ensemble_checkpoints`, and plot output directories before writing. Replace the hard-coded encoder path in each doctor driver. |
| Graph batching | Regression drivers create `Data` objects from JSON lists and then expect stacked `(B, N, F)` tensors in `prepare_graph_batch`. Define a consistent collation contract. With standard tensor-based PyG batching, consume `batch.x`, `batch.edge_index`, and `batch.batch` directly instead of applying a second graph-index offset. |
| Recording length and time | Replace the fixed `total_graphs = 1199` and `10 + graph_index / 60` assumptions with validated recording lengths and actual timestamps. Keep augmented views aligned. |
| Missing targets | Several branches replace missing PANAS labels with `30`. Define and document an appropriate missing-label policy rather than silently treating imputed values as observed ground truth. |
| Evaluation mode | The epoch-level test loop enters `torch.no_grad()` without switching the model to evaluation mode. Call `model.eval()` for test/calibration, then restore training mode; otherwise dropout remains active and batch-normalization statistics can change on held-out inputs. |
| Incomplete batches | Test and calibration loaders use `drop_last=True`. Retain final batches for complete evaluation and check for empty loaders on small subsets. |
| Log completeness | Parsers depend on exact printed markers and, for calibration, four following user-result lines. Validate that all three doctors and all required calibration entries are present. Missing doctors must not silently lower disagreement. |
| Zero thresholds | Missing calibration scores can produce zero thresholds, while `analyze_distance` divides by those thresholds. Handle missing calibration and zero denominators explicitly. |

## Intended execution order

Complete the adjustments above first. The following commands identify the workflow; they assume all scripts now use consistent project-root paths.

### 1. Prepare graphs and labels

```bash
cd "$EMPATIA_UNCERTAINTY_ROOT"
python Json_process/Pack_json_graph.py
python Json_process/Pack_json_label.py
```

Generate and verify the augmented graph caches as described above. Confirm user IDs, frame counts, graph topology, and questionnaire slots before pretraining.

### 2. Pretrain encoders

`Graph_encoder_loso.py` provides a standalone encoder-training reference:

```bash
python Graph_encoder_loso.py
```

For the chapter's doctor ensemble, adapt that training stage to the doctor-specific partitions and required checkpoint names. The supplied standalone command does not generate the full doctor-specific encoder bank.

### 3. Train the three doctor models

Once the matching encoder files are available:

```bash
python Ensemble_Doc_1.py > Ensemble_Doc1.log 2>&1
python Ensemble_Doc_2.py > Ensemble_Doc2.log 2>&1
python Ensemble_Doc_3.py > Ensemble_Doc3.log 2>&1
```

Each script trains decoders for positive/negative affect and all five slots across target users. The default loops are substantial; verify one user/slot configuration before a full experiment.

Selected checkpoints are written as `Ensemble_checkpoints/U_<ID>_<p-or-n>_<slot>_Doc_<R>.pth`. They include model and optimizer state dictionaries, epoch, KL criterion, and prediction summaries.

### 4. Calibrate and analyze

```bash
python Diagnose_Docs.py > diagnosis_report.json
```

Configure its three doctor-log paths first. `Read_Doc.read_doc` currently also requires a separate single-model comparison log, although the uncertainty calculation uses only `Doc1`, `Doc2`, and `Doc3`. Supply that comparison log or decouple its parsing from the ensemble-only workflow.

The diagnostic script prints a nested JSON report indexed by user, assessment slot, and affect dimension, including the case code, uncertainty values, thresholds, and relative threshold exceedance. Plotting utilities require their own paths and output directories to be configured.

## Evaluation and reproducibility

Chapter 6 describes target-independent LOSO training, calibration-based checkpoint selection, and separate final target evaluation. The checked-in scripts preserve earlier experimental choices that differ from that protocol:

- **Encoder selection:** The generic LOSO encoder uses target-user contrastive loss to retain a checkpoint. Use a validation subset drawn from training subjects instead when reproducing strict target-independent evaluation.
- **Decoder selection:** Doctor scripts choose checkpoints by comparing the training-label distribution with the target user's predicted distribution, using a KL score multiplied by 10. The chapter instead selects on held-out calibration users. Move this selection to calibration data and reserve target predictions for final testing.
- **Distribution summaries:** The scripts average predicted standard deviations over batches. A Gaussian summary of a predictive mixture generally needs both the average predicted variance and dispersion of predicted means. Document the aggregation rule and preserve per-sample predictions for probabilistic metrics.
- **Calibration granularity:** Thresholds are estimated separately for each target/slot/dimension from four calibration-user summaries. Treat the percentile as an experimental policy parameter and retain the calibration sample counts.
- **Metrics:** Gaussian NLL, printed KL scores, uncertainty values, and legacy “in-range probability” summaries are different quantities. The existing accuracy-analysis script checks a `mu ± sigma` interval; do not interpret that automatically as the chapter's chosen interval coverage or error metrics.
- **Reproducibility:** Save the exact subject partitions, graph variant, timestamp handling, augmentation settings, seeds, label policy, encoder provenance, and checkpoints. The deterministic doctor splitter alone does not make the complete training run deterministic.

No pretrained benchmark results are bundled. Default execution should not be claimed to reproduce the chapter's final MMASH/WESAD scores or triage tables without aligning the experimental protocol.

## Citation

This code is associated with Chapter 6 of the doctoral thesis. A dedicated Chapter 6 paper citation is not specified in this repository. To reference the thesis:

```bibtex
@phdthesis{sun2026practical,
  author = {Sun, Junjiao},
  title  = {Toward Practical and Uncertainty-Aware Affective Computing with Wearable Physiological Signals},
  school = {Universidad Politécnica de Madrid},
  year   = {2026}
}
```

Please also cite the original dataset publications when using their data.

## License

No repository-level license file is included. Contact the maintainers to clarify reuse and redistribution terms. Dataset access and reuse remain subject to the respective providers' terms.
