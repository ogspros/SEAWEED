# Where the files went

The scripts were written and run in one flat directory. For the release they
were sorted into the tree described in the README, without being renamed, so
that every name in `RUNS.md`, in the logs and in the notes still finds its
file. The only edit to the moved Python files is the import of the workflow
modules, which now reads `from seaweed.<module> import ...`.

The shell drivers call the scripts at their new paths, and take the data
from the directory they are run in.

The state in which the paper's numbers were computed is the tag
`v1.0-as-run`. Any file below, moved or removed, can be read there:

```
git show v1.0-as-run:run_real.py               # print the file as it was
git checkout v1.0-as-run -- shape_decon.py     # bring a removed file back
```

## Moved

| as run | now |
|---|---|
| `anchors_from_consensus.py` | `seaweed/anchors_from_consensus.py` |
| `pick_consensus.py` | `seaweed/pick_consensus.py` |
| `pick_seafloor.py` | `seaweed/pick_seafloor.py` |
| `run_real.py` | `seaweed/run_real.py` |
| `seafloor_robust.py` | `seaweed/seafloor_robust.py` |
| `wavelet_extractor_qc.py` | `seaweed/wavelet_extractor_qc.py` |
| `wavelet_extractor_v2.py` | `seaweed/wavelet_extractor_v2.py` |
| `wavelet_to_segy.py` | `seaweed/wavelet_to_segy.py` |
| `eval_picker_real.py` | `validation/archive/eval_picker_real.py` |
| `fig_autocorr.py` | `validation/archive/fig_autocorr.py` |
| `fig_deghost.py` | `validation/archive/fig_deghost.py` |
| `fig_real_surveys.py` | `validation/archive/fig_real_surveys.py` |
| `fig_span.py` | `validation/archive/fig_span.py` |
| `ms29_deghost_check.py` | `validation/archive/ms29_deghost_check.py` |
| `run_span_experiment.sh` | `validation/archive/run_span_experiment.sh` |
| `track_from_consensus.py` | `validation/archive/track_from_consensus.py` |
| `wavelet_spread.py` | `validation/archive/wavelet_spread.py` |
| `add_reference_runs.py` | `validation/synthetic/add_reference_runs.py` |
| `airgun_generator.py` | `validation/synthetic/airgun_generator.py` |
| `build_matrix_v2.py` | `validation/synthetic/build_matrix_v2.py` |
| `check_rev2.py` | `validation/synthetic/check_rev2.py` |
| `check_snr_scale.py` | `validation/synthetic/check_snr_scale.py` |
| `check_tracking.py` | `validation/synthetic/check_tracking.py` |
| `compare_v5_v6.py` | `validation/synthetic/compare_v5_v6.py` |
| `count_padding.py` | `validation/synthetic/count_padding.py` |
| `eval_picker.py` | `validation/synthetic/eval_picker.py` |
| `generate_line2_hf.py` | `validation/synthetic/generate_line2_hf.py` |
| `generate_parallel_hf.py` | `validation/synthetic/generate_parallel_hf.py` |
| `make_fig_synthetic.py` | `validation/synthetic/make_fig_synthetic.py` |
| `padding_from_arrays.py` | `validation/synthetic/padding_from_arrays.py` |
| `phase_synth_test.py` | `validation/synthetic/phase_synth_test.py` |
| `picker_tolerance.py` | `validation/synthetic/picker_tolerance.py` |
| `provenance_padding.py` | `validation/synthetic/provenance_padding.py` |
| `qc_synth_case.py` | `validation/synthetic/qc_synth_case.py` |
| `rerun_v5.sh` | `validation/synthetic/rerun_v5.sh` |
| `rerun_v6.sh` | `validation/synthetic/rerun_v6.sh` |
| `rescore_final.py` | `validation/synthetic/rescore_final.py` |
| `run_matrix_fault.sh` | `validation/synthetic/run_matrix_fault.sh` |
| `run_matrix_shallow.sh` | `validation/synthetic/run_matrix_shallow.sh` |
| `run_matrix_snr.sh` | `validation/synthetic/run_matrix_snr.sh` |
| `run_matrix_v2.py` | `validation/synthetic/run_matrix_v2.py` |
| `run_matrix_v2.sh` | `validation/synthetic/run_matrix_v2.sh` |
| `run_matrix_v5.py` | `validation/synthetic/run_matrix_v5.py` |
| `run_matrix_v6.py` | `validation/synthetic/run_matrix_v6.py` |
| `run_matrix_z.sh` | `validation/synthetic/run_matrix_z.sh` |
| `run_nshots.sh` | `validation/synthetic/run_nshots.sh` |
| `shallow_discrim.py` | `validation/synthetic/shallow_discrim.py` |
| `source_library.py` | `validation/synthetic/source_library.py` |
| `tables_v5.py` | `validation/synthetic/tables_v5.py` |
| `tables_v5_extra.py` | `validation/synthetic/tables_v5_extra.py` |
| `test_line_length.py` | `validation/synthetic/test_line_length.py` |
| `test_nshots.py` | `validation/synthetic/test_nshots.py` |
| `test_reference.py` | `validation/synthetic/test_reference.py` |
| `true_rwb.py` | `validation/synthetic/true_rwb.py` |
| `velocity_generator.py` | `validation/synthetic/velocity_generator.py` |
| `analyse_viking_blocks.py` | `validation/viking/analyse_viking_blocks.py` |
| `check_weight_spread.py` | `validation/viking/check_weight_spread.py` |
| `count_zeros.py` | `validation/viking/count_zeros.py` |
| `farfield_window_test.py` | `validation/viking/farfield_window_test.py` |
| `make_paper_figures_vik.py` | `validation/viking/make_paper_figures_vik.py` |
| `run_viking_blocks.sh` | `validation/viking/run_viking_blocks.sh` |
| `test_avo.py` | `validation/viking/test_avo.py` |
| `test_jitter_ceiling.py` | `validation/viking/test_jitter_ceiling.py` |
| `test_minphase.py` | `validation/viking/test_minphase.py` |
| `test_multiple.py` | `validation/viking/test_multiple.py` |
| `viking_coherence_vs_truth.py` | `validation/viking/viking_coherence_vs_truth.py` |
| `viking_layer_polarity.py` | `validation/viking/viking_layer_polarity.py` |
| `viking_lobes.py` | `validation/viking/viking_lobes.py` |
| `viking_phase_test.py` | `validation/viking/viking_phase_test.py` |
| `viking_rerun_conv.sh` | `validation/viking/viking_rerun_conv.sh` |
| `viking_rotation.py` | `validation/viking/viking_rotation.py` |
| `viking_spread_vs_accuracy.py` | `validation/viking/viking_spread_vs_accuracy.py` |
| `viking_stacking_filter.py` | `validation/viking/viking_stacking_filter.py` |
| `viking_thin_layer.py` | `validation/viking/viking_thin_layer.py` |
| `viking_zones.py` | `validation/viking/viking_zones.py` |
| `where_disagreement.py` | `validation/viking/where_disagreement.py` |

## Removed from the release

They produced no number of the paper, or they have been superseded. They
remain in the tag `v1.0-as-run`.

**Neural-network project (not SEAWEED).** `MS29_validazione_profondita.md`, `apply_ms29_tt.py`, `apply_real_hf.py`, `architettura_token_traccia.md`, `baseline_classico.py`, `buchi.sh`, `check_consistency.py`, `data_loader.py`, `dataset_hf80_design_notes.md`, `dataset_hf80_design_notes_up1.md`, `dataset_hf80_design_notes_up2.md`, `delete_vintage.py`, `depth_probe.py`, `diagnose_qc.py`, `film_model.py`, `film_su_z_fase1.md`, `finetune_real.py`, `generate_dataset_gpu.py`, `generate_dataset_large.py`, `ghost_model.py`, `make_minphase_source.py`, `mute_test.py`, `net_tt.py`, `occlusion_test.py`, `precompute_windows.py`, `precompute_zest.py`, `profile_pipeline.py`, `qc_wavelet.py`, `rebaseline_extractor.py`, `rebaseline_extractor2.py`, `run_parallel_geomvar.sh`, `run_parallel_geomvar_add.sh`, `run_parallel_geomvar_reband.sh`, `run_parallel_hf.sh`, `run_precompute_windows.sh`, `sequenza_validazione_hf.md`, `test_finestratura.py`, `test_generazione.py`, `test_regime.py`, `token_trace_skeleton.py`, `trace_encoder.py`, `train_film.py`, `train_large.py`, `train_large_hf.py`, `train_tt.py`, `varpro_inversion.py`, `verifica_ghost_fisico.py`, `verify_predictions.py`

**Patch scripts, already applied to the files they patched.** `patch_fig_diag.py`, `patch_fig_grid_only.py`, `patch_fig_no_padding.py`, `patch_fig_plot_one.py`, `patch_fig_read_frac.py`, `patch_fig_read_wavelet.py`, `patch_fig_synthetic.py`, `patch_matrix_frac_padded.py`, `patch_matrix_only.py`, `patch_matrix_save_wavelet.py`, `patch_no_silent_pad.py`, `patch_roughness.py`, `patch_run_real_anchors.py`, `patch_run_real_blocks.py`, `patch_run_real_cube.py`, `patch_run_real_weights.py`, `patch_save_multiple.py`, `patch_shoot_z.py`, `patch_tail.py`, `patch_velocity_generator.py`, `patch_weights.py`

**Superseded versions.** `analyse_matrices.py`, `analyse_matrices_v3.py`, `compare_matrices.py`, `make_paper_figures_vik_old_4block.py`, `origin_lag_check.py`, `rebaseline_v3.py`, `rerun_all_synthetics.sh`, `rerun_v3.sh`, `rescore_v3.py`, `run_matrix_v3.py`, `run_matrix_v4.py`, `shallow_check_v3.py`, `span_experiment/anchors_from_consensus.py`, `viking_blocks/make_paper_figures_vik.py`, `viking_blocks/where_disagreement.py`

**Early prototypes, replaced by the workflow in seaweed/.** `adaptive_offset.py`, `align_check.py`, `autocorr_check.py`, `bubble_band_analysis.py`, `check_semblance.py`, `check_synthetic.py`, `direct_anchor.py`, `extract_ms29.py`, `extract_real_qc.py`, `extraction_stage_probe.py`, `ghost_ratio_test.py`, `independent_check.py`, `qc_single_line.py`, `qc_single_line_v2.py`, `seafloor_real.py`, `seafloor_track_real.py`, `seafloor_tracking.py`, `shape_decon.py`, `stretch_test.py`, `test_layer_geometry.py`, `triangulate_spectra.py`, `wavelet_length.py`, `zrec_autocorr.py`
