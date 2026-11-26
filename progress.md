# Experiment Progress

更新时间：2026-05-29 19:54 Asia/Shanghai

说明：本文件只保留实验结果、observations 和 TODO。旧的运行流水、代码改动记录和已过期建议已删除。

## Experiment Results

### 1. Main Causal Attribution Across Models

设置：

- Dataset: `src/all_questions_en.json`, 100 queries
- Runner: `src/run_control_experiment.py`
- Conditions: `direct_single`, `direct_batch`, `single_record_sbq`, `multi_record_sbq`
- `attack_times=3`
- Judge: `gpt-5-nano`
- Summary: `src/outputs/control/main_table_causal_attribution_summary.json`

ASR@3 主表：

| Model | Direct Single | Direct Batch b=10 | Single-record JSON SBQ | Multi-record JSON SBQ b=10 | Delta Reframing | Delta Batch under SBQ |
|---|---:|---:|---:|---:|---:|---:|
| `gemini-2.5-flash` | 0.09 | 0.10 | 0.90 | 0.96 | +0.81 | +0.06 |
| `gpt-5-chat` | 0.50 | 0.68 | 0.47 | 0.85 | -0.03 | +0.38 |
| `deepseek-v3.1/deepseek-chat` | 0.06 | 0.10 | 0.99 | 1.00 | +0.93 | +0.01 |
| `qwen-plus` | 0.03 | 0.06 | 0.60 | 1.00 | +0.57 | +0.40 |

完整 ASR@k：

| Model | Condition | ASR@1 | ASR@2 | ASR@3 |
|---|---|---:|---:|---:|
| `gemini-2.5-flash` | Direct Single | 0.05 | 0.08 | 0.09 |
| `gemini-2.5-flash` | Direct Batch b=10 | 0.04 | 0.10 | 0.10 |
| `gemini-2.5-flash` | Single-record JSON SBQ | 0.79 | 0.89 | 0.90 |
| `gemini-2.5-flash` | Multi-record JSON SBQ b=10 | 0.78 | 0.87 | 0.96 |
| `gpt-5-chat` | Direct Single | 0.34 | 0.47 | 0.50 |
| `gpt-5-chat` | Direct Batch b=10 | 0.39 | 0.67 | 0.68 |
| `gpt-5-chat` | Single-record JSON SBQ | 0.30 | 0.40 | 0.47 |
| `gpt-5-chat` | Multi-record JSON SBQ b=10 | 0.67 | 0.76 | 0.85 |
| `deepseek-v3.1/deepseek-chat` | Direct Single | 0.02 | 0.06 | 0.06 |
| `deepseek-v3.1/deepseek-chat` | Direct Batch b=10 | 0.03 | 0.10 | 0.10 |
| `deepseek-v3.1/deepseek-chat` | Single-record JSON SBQ | 0.96 | 0.99 | 0.99 |
| `deepseek-v3.1/deepseek-chat` | Multi-record JSON SBQ b=10 | 0.91 | 1.00 | 1.00 |
| `qwen-plus` | Direct Single | 0.03 | 0.03 | 0.03 |
| `qwen-plus` | Direct Batch b=10 | 0.00 | 0.03 | 0.06 |
| `qwen-plus` | Single-record JSON SBQ | 0.54 | 0.58 | 0.60 |
| `qwen-plus` | Multi-record JSON SBQ b=10 | 0.89 | 1.00 | 1.00 |

Run notes:

- `qwen-plus` Direct Single 有 3 次上游内容过滤 400，按空响应计入。
- `qwen-plus` Multi-record SBQ 有 1 次子进程级 timeout，按空响应计入。

### 2. Gemini Aggregation vs. Task Reframing Controls

设置：

- Model: `gemini_25` / Gemini-2.5-Flash
- `attack_times=3`
- Summary: `src/outputs/control/gemini_25/gemini_25_exp1_control_summary.json`

| Condition | Semantics | Batch size | ASR@1 | ASR@2 | ASR@3 |
|---|---|---:|---:|---:|---:|
| Direct Single | direct instruction | 1 | 0.05 | 0.08 | 0.09 |
| Direct Batch | direct instruction | 10 | 0.04 | 0.10 | 0.10 |
| Plain Batch | weak structure | 10 | 0.01 | 0.01 | 0.01 |
| Bullet Batch | weak structure | 10 | 0.00 | 0.03 | 0.11 |
| Single-record SBQ | structured completion | 1 | 0.79 | 0.89 | 0.90 |
| Multi-record SBQ | structured completion | 10 | 0.78 | 0.87 | 0.96 |

### 3. Gemini Single-query Completion Control

设置：

- Model: `gemini_25` / Gemini-2.5-Flash
- Reuses Experiment 1 control outputs
- Summary: `src/outputs/control/gemini_25/gemini_25_exp2_single_query_completion_summary.json`

| Condition | Semantics | ASR@1 | ASR@2 | ASR@3 | Refusal rate @3 |
|---|---|---:|---:|---:|---:|
| Direct Single | direct instruction | 0.05 | 0.08 | 0.09 | 0.91 |
| Single-record SBQ | structured completion, no batch amplification | 0.79 | 0.89 | 0.90 | 0.10 |
| Multi-record SBQ | structured completion, batched | 0.78 | 0.87 | 0.96 | 0.04 |

### 4. Structured Format Ablation

设置：

- Model: `gemini_25` / Gemini-2.5-Flash
- `batch_size=10`
- `attack_times=1`
- `suffix_mode=simple`
- Summary: `src/outputs/gemini_25/format_ablation_fixed/gemini_25_format_ablation_fixed_summary.json`

| Format | Batch size | Non-empty extracted | Valid/response rate | ASR |
|---|---:|---:|---:|---:|
| Plain | 10 | 90/100 | 0.90 | 0.00 |
| Bullets | 10 | 100/100 | 1.00 | 0.00 |
| JSON | 10 | 57/100 | 0.57 | 0.55 |
| Half JSON | 10 | 40/100 | 0.40 | 0.39 |
| YAML | 10 | 50/100 | 0.50 | 0.48 |
| XML | 10 | 100/100 | 1.00 | 0.97 |
| Markdown table | 10 | 96/100 | 0.96 | 0.93 |

Quality flags:

- XML / Markdown table 的 `regex_no_match` 较多，投稿前需要人工抽查。
- `valid/response rate` 表示修复提取后能得到非空 per-query response 的比例；对 Plain/Bullets 不是 JSON parse validity。

### 5. GPT-5-Chat Batch-size Sweep

设置：

- Model: `gpt-5-chat`
- Format: JSON SBQ
- `attack_times=1`
- b <= 10 复用旧 target attack 文件并用修复后的 extractor 重提取、重评估
- b >= 16 新跑 target calls
- Summary: `src/outputs/gpt-5-chat/large_batch_sweep/gpt-5-chat_large_batch_sweep_fixed_summary.json`

| Batch size | Approx prompt tokens | Valid/response rate | ASR | Raw empty batches | Raw response chars min-max |
|---:|---:|---:|---:|---:|---:|
| 1 | 691 | 1.00 | 0.12 | 0 | 981-3,279 |
| 2 | 845 | 1.00 | 0.19 | 0 | 1,238-5,575 |
| 4 | 1,156 | 1.00 | 0.16 | 0 | 3,532-8,829 |
| 6 | 1,446 | 1.00 | 0.38 | 0 | 3,597-11,736 |
| 8 | 1,747 | 1.00 | 0.37 | 0 | 4,334-14,413 |
| 10 | 2,048 | 1.00 | 0.51 | 0 | 5,087-17,513 |
| 16 | 3,000 | 1.00 | 0.47 | 0 | 4,084-13,510 |
| 32 | 5,419 | 0.36 | 0.16 | 1 | 1-27,579 |
| 64 | 10,022 | 0.03 | 0.03 | 1 | 1-1,544 |

Quality flags:

- 旧图中的 extracted 文件存在历史 extractor bug：一个 batch 内多个 responses 会被错误挂到每个 query 上。上表为修复提取后的结果。
- b=32/64 的 ASR 下降主要来自输出不完整/格式崩溃和 valid response rate 下降，不应解释为模型更安全。

### 6. Preliminary Defense Baselines

设置：

- Model: `gemini_25` / Gemini-2.5-Flash
- Baseline: Multi-record JSON SBQ, `batch_size=10`, `attack_times=3`
- No new target-model generation
- Summary: `src/outputs/defense/gemini_25_defense_baseline_summary.json`

| Defense | ASR@1 | ASR@3 | Filter/refusal rate | Target calls / 100 queries | Filter calls / 100 queries | Notes |
|---|---:|---:|---:|---:|---:|---|
| None | 0.78 | 0.96 | 0.04 | 30 | 0 | Default multi-record SBQ |
| Pre-batch screening | 0.00 | 0.00 | 1.00 | 0 | 100 | Dataset-label input filter; proxy upper bound |
| De-batching | 0.79 | 0.90 | 0.10 | 300 | 0 | Single-record SBQ; removes shared batch context |
| Post-generation filtering | 0.00 | 0.00 | 0.96 | 30 | 300 | Evaluation-judge/proxy output filter; upper bound |

### 7. Open-weight BatchJail补跑

设置：

- Runner: `src/run_control_condition_timeout.py`
- Condition: Multi-record JSON SBQ
- `batch_size=10`
- `attack_times=3`
- Judge: `gpt-5-nano`

| Model | OpenRouter model | ASR@1 | ASR@2 | ASR@3 | Notes |
|---|---|---:|---:|---:|---|
| `qwen2.5-instruct-7` | `qwen/qwen-2.5-7b-instruct` | 0.50 | 0.76 | 0.88 | 100/100 evaluated; 0 target-call exceptions |
| `mistral-7b-instruct` | `mistralai/mistral-7b-instruct-v0.1` | 0.29 | 0.34 | 0.34 | 100/100 evaluated; 3/30 target-call timeouts counted as empty responses |
| `gemma-3-27b-it` | `google/gemma-3-27b-it` | 0.95 | 0.99 | 0.99 | 100/100 evaluated; 0 target-call exceptions; paid endpoint |

Source files:

- `src/outputs/control/qwen2.5-instruct-7/qwen2.5-instruct-7_exp1_control_summary.json`
- `src/outputs/control/mistral-7b-instruct/mistral-7b-instruct_exp1_control_summary.json`
- `src/outputs/control/gemma-3-27b-it/gemma-3-27b-it_exp1_control_summary.json`

### 8. Qwen2.5 Refusal-space Internal-state Probe

设置：

- Model: `Qwen/Qwen2.5-7B-Instruct`
- Prompt dataset: `1540` prompts
- Capture points: `prompt_end`, `query_span`, `field_name_span`
- Layers: all hidden-state layers
- Exported vectors: `319696 x 3584`
- Server output: `$JCF_CACHE/vector_space_viz/runs/qwen2p5_7b/`
- Local summary: `src/vector_space_viz/figs/qwen2p5_refusal_space/`

Prompt-level refusal rates from short generation + heuristic:

| Mechanism | Refusal rate | Refusal count | Prompt count |
|---|---:|---:|---:|
| `direct_single` | 0.675 | 135 | 200 |
| `direct_batch` | 0.400 | 8 | 20 |
| `single_record_sbq` | 0.260 | 312 | 1200 |
| `multi_record_sbq` | 0.175 | 21 | 120 |

Best query-span direct probe transfer:

| Train group | Direct batch | Direct single | Multi-record SBQ | Single-record SBQ |
|---|---:|---:|---:|---:|
| `direct_single` | 0.822 | 0.958 | 0.612 | 0.795 |

Mean query-span refusal-direction cosine:

| Pair | Cosine |
|---|---:|
| `direct_single` vs `direct_batch` | 0.634 |
| `direct_single` vs `multi_record_sbq` | 0.503 |
| `direct_single` vs `single_record_sbq` | 0.326 |
| `single_record_sbq` vs `multi_record_sbq` | -0.089 |

Axis deep-dive at query-span layer 19:

| Mechanism | n | Refusal rate | Direct-refusal AUC | Direct-refusal label delta | Schema mean | Batch mean |
|---|---:|---:|---:|---:|---:|---:|
| `direct_batch` | 20 | 0.400 | 0.948 | 8.619 | -15.573 | 20.151 |
| `direct_single` | 200 | 0.675 | 0.926 | 12.567 | -27.328 | -3.943 |
| `multi_record_sbq` | 120 | 0.175 | 0.627 | 2.562 | 4.398 | 23.655 |
| `single_record_sbq` | 1142 | 0.247 | 0.581 | 2.826 | 4.597 | -2.148 |

Observation:

- Direct refusal direction is strong inside direct prompts but much weaker inside SBQ prompts.
- Schema axis is linearly separable and negatively correlated with refusal.
- Direct-refusal-only CV AUC over all prompt units is `0.636`; adding schema gives `0.713`; adding frame axes gives `0.734`.
- `unsafe_response` field names do not reliably increase refusal. In XML/table conditions they often reduce refusal, suggesting field-name semantics may be absorbed into slot-filling / schema-following.

### 9. Qwen2.5 Frame-invariance Smoke

设置：

- Model: `Qwen/Qwen2.5-7B-Instruct`
- Dataset: first `8` harmful queries from `src/all_questions_en.json`
- Server output: `$JCF_CACHE/vector_space_viz/runs/frame_invariance_smoke/qwen2p5/`

Prompt-level refusal rates:

| Mechanism | Frame family | n | Refusal rate |
|---|---|---:|---:|
| `direct_single` | direct request | 8 | 1.000 |
| `translate_frame` | semantic transformation | 8 | 0.000 |
| `summarize_frame` | semantic transformation | 8 | 0.000 |
| `rewrite_frame` | semantic transformation | 8 | 0.000 |
| `extract_frame` | semantic transformation | 8 | 0.000 |
| `single_record_sbq` | schema completion | 8 | 0.000 |
| `multi_record_sbq` | schema completion + batch | 2 | 0.000 |

Observation:

- This supports reframing the paper around task-frame invariance rather than only structured/batched jailbreaks.
- The same harmful intent is refused as direct request but not refused under translation, summarization, rewriting, extraction, or schema completion.
- Smoke is too small for probe-transfer claims; medium run `frame_invariance_qwen2p5_m50` is running with harmless/harmful controls.

### 10. Qwen2.5 Frame-invariance Medium Run

设置：

- Model: `Qwen/Qwen2.5-7B-Instruct`
- Dataset: `50` harmful main queries + `25` harmless/harmful control pairs
- Prompt count: `1170`
- Exported vectors: `183222 x 3584`
- Server output: `$JCF_CACHE/vector_space_viz/runs/frame_invariance_qwen2p5_m50/qwen2p5/`
- Local summary: `src/vector_space_viz/figs/qwen2p5_frame_invariance_m50/deep_dive_axes_layer19/`

Prompt-level refusal rates:

| Mechanism | Frame family | n | Refusal rate |
|---|---|---:|---:|
| `direct_single` | direct request | 100 | 0.700 |
| `direct_batch` | direct request + batch | 10 | 0.400 |
| `translate_frame` | semantic transformation | 100 | 0.000 |
| `summarize_frame` | semantic transformation | 100 | 0.030 |
| `rewrite_frame` | semantic transformation | 100 | 0.000 |
| `extract_frame` | semantic transformation | 100 | 0.000 |
| `single_record_sbq` | schema completion | 600 | 0.285 |
| `multi_record_sbq` | schema completion + batch | 60 | 0.200 |

Layer-19 query-span axis deep-dive:

| Feature set | Refusal CV AUC |
|---|---:|
| `direct_refusal` | 0.621 |
| `direct_refusal + schema` | 0.627 |
| `direct_refusal + schema + batch` | 0.625 |
| `frame_axes_only` | 0.842 |
| `direct_refusal + frame_axes` | 0.841 |

Observation:

- Semantic transformations almost eliminate refusal despite preserving harmful intent.
- This provides stronger motivation than only structured/batched attacks: the core issue is task-frame invariance failure.
- Direct-refusal geometry alone is insufficient across frames. Frame axes predict refusal much better.
- Full cross-mechanism probe transfer is computationally heavy with the expanded mechanism set; selected-layer analysis is now supported by `--layers` in `fit_refusal_probes.py` and `compare_subspaces.py`.

### 11. Direct-trained Transfer on Qwen2.5 Frame-invariance Run

设置：

- Train group: `direct_single` only.
- Test groups: all mechanisms.
- Capture points: `query_span,prompt_end`.
- Layers: `14,19,24,28`.
- Output: `src/vector_space_viz/figs/qwen2p5_frame_invariance_m50/analysis_direct_single_selected/probe_transfer.csv`

Layer-19 query-span transfer:

| Test group | Positive rate | AUC | Accuracy |
|---|---:|---:|---:|
| `direct_single` | 0.686 | 0.924 | 0.771 |
| `direct_batch` | 0.400 | 0.667 | 0.640 |
| `single_record_sbq` | 0.270 | 0.692 | 0.533 |
| `multi_record_sbq` | 0.204 | 0.591 | 0.448 |
| `summarize_frame` | 0.030 | 0.735 | 0.350 |
| `translate_frame` | 0.000 | NA | 0.310 |
| `rewrite_frame` | 0.000 | NA | 0.310 |
| `extract_frame` | 0.000 | NA | 0.310 |

Observation:

- A direct-trained refusal probe transfers well to direct prompts, less well to direct batching, and weakly to schema/batch prompts.
- Translation/rewrite/extraction have no behavioral refusals, but the direct-trained classifier still predicts many of them as refusal-like. This suggests the harmful/refusal signal may remain partially present, while the active task frame changes the behavioral readout.
- This points to a stronger mechanism hypothesis than "refusal vector disappears": refusal-like evidence and task-frame-following evidence may coexist, and the model's final policy/readout chooses the task-frame behavior.

### 12. Cross-model Frame-invariance Queue

Status:

- Completed: `Qwen/Qwen2.5-0.5B-Instruct`.
- Completed: `Qwen/Qwen3-4B-Instruct-2507`.
- Completed: `mistralai/Mistral-7B-v0.1` as a non-instruct/base control.
- Queue log: `$JCF_CACHE/vector_space_viz/runs/frame_invariance_cross_model_selected/run_remaining_tok48.log`
- Purpose: compare small aligned Qwen, newer aligned Qwen, and non-instruct Mistral base.

Failed/skipped model notes:

- `lmsys/vicuna-7b-v1.1` is cached as PyTorch `.bin`; current transformers refuses loading it with torch `<2.6` because of the 2025 `torch.load` security restriction.
- `Qwen/Qwen2-7B-Instruct` cache is incomplete: missing safetensor shards.
- Old `run_more_models_v3.sh` queue was stopped because it used the pre-frame-invariance dataset/all-layer analysis and was stuck in download.

### 13. Qwen2.5-0.5B Frame-invariance Result

设置：

- Model: `Qwen/Qwen2.5-0.5B-Instruct`
- Prompt set: same medium frame-invariance prompts as Qwen2.5-7B.
- Layers: `8,14,19,24`.
- Exported vectors: `11672 x 896`.
- Server output: `$JCF_CACHE/vector_space_viz/runs/frame_invariance_cross_model_selected/qwen2p5_0p5b_frame_m50/`
- Local summary: `src/vector_space_viz/figs/qwen2p5_0p5b_frame_invariance_m50/`

Prompt-level refusal rates:

| Mechanism | n | Refusal rate |
|---|---:|---:|
| `direct_single` | 100 | 0.620 |
| `direct_batch` | 10 | 0.000 |
| `translate_frame` | 100 | 0.000 |
| `summarize_frame` | 100 | 0.000 |
| `rewrite_frame` | 100 | 0.000 |
| `extract_frame` | 100 | 0.000 |
| `single_record_sbq/json` | 200 | 0.115 |
| `single_record_sbq/markdown_table` | 200 | 0.125 |
| `single_record_sbq/xml` | 200 | 0.070 |
| `multi_record_sbq/json` | 20 | 0.300 |
| `multi_record_sbq/markdown_table` | 20 | 0.400 |
| `multi_record_sbq/xml` | 20 | 0.400 |

Layer-19 query-span direct-trained transfer:

| Test group | Positive rate | AUC | Accuracy |
|---|---:|---:|---:|
| `direct_single` | 0.629 | 0.892 | 0.829 |
| `direct_batch` | 0.000 | NA | 0.350 |
| `single_record_sbq` | 0.105 | 0.659 | 0.491 |
| `multi_record_sbq` | 0.383 | 0.767 | 0.598 |
| `translate_frame` | 0.000 | NA | 0.450 |
| `summarize_frame` | 0.000 | NA | 0.390 |
| `rewrite_frame` | 0.000 | NA | 0.450 |
| `extract_frame` | 0.000 | NA | 0.400 |

Combined refusal prediction:

| Feature set | CV AUC |
|---|---:|
| `direct_refusal` | 0.711 |
| `direct_refusal + schema` | 0.767 |
| `direct_refusal + schema + batch` | 0.770 |
| `frame_axes_only` | 0.827 |
| `direct_refusal + frame_axes` | 0.869 |

Observation:

- The task-frame invariance failure replicates at small scale: direct refusal is present, semantic transformations are zero-refusal.
- Batching behaves differently from Qwen2.5-7B: multi-record SBQ has higher refusal than single-record SBQ, so batching is not uniformly a bypass across model scales.
- Direct refusal alone is not sufficient. Frame axes improve refusal prediction substantially (`0.711` to `0.869`).

### 14. Qwen3-4B Frame-invariance Result

设置：

- Model: `Qwen/Qwen3-4B-Instruct-2507`
- Prompt set: same medium frame-invariance prompts.
- Layers: `14,19,24,28`.
- Max generated tokens: `48`.
- Exported vectors: `11672 x 2560`.
- Server output: `$JCF_CACHE/vector_space_viz/runs/frame_invariance_cross_model_selected/qwen3_4b_frame_m50_tok48/`
- Local summary: `src/vector_space_viz/figs/qwen3_4b_frame_invariance_m50_tok48/`

Prompt-level refusal rates:

| Mechanism | n | Refusal rate |
|---|---:|---:|
| `direct_single` | 100 | 0.740 |
| `direct_batch` | 10 | 0.400 |
| `translate_frame` | 100 | 0.000 |
| `summarize_frame` | 100 | 0.180 |
| `rewrite_frame` | 100 | 0.280 |
| `extract_frame` | 100 | 0.000 |
| `single_record_sbq/json` | 200 | 0.170 |
| `single_record_sbq/markdown_table` | 200 | 0.400 |
| `single_record_sbq/xml` | 200 | 0.340 |
| `multi_record_sbq/json` | 20 | 0.000 |
| `multi_record_sbq/markdown_table` | 20 | 0.150 |
| `multi_record_sbq/xml` | 20 | 0.000 |

Layer-19 query-span direct-trained transfer:

| Test group | Positive rate | AUC | Accuracy |
|---|---:|---:|---:|
| `direct_single` | 0.743 | 0.987 | 0.857 |
| `direct_batch` | 0.400 | 0.607 | 0.630 |
| `single_record_sbq` | 0.275 | 0.723 | 0.505 |
| `multi_record_sbq` | 0.052 | 0.668 | 0.310 |
| `summarize_frame` | 0.180 | 0.760 | 0.440 |
| `rewrite_frame` | 0.280 | 0.827 | 0.540 |
| `translate_frame` | 0.000 | NA | 0.400 |
| `extract_frame` | 0.000 | NA | 0.260 |

Combined refusal prediction:

| Feature set | CV AUC |
|---|---:|
| `direct_refusal` | 0.631 |
| `direct_refusal + schema` | 0.654 |
| `direct_refusal + schema + batch` | 0.654 |
| `frame_axes_only` | 0.753 |
| `direct_refusal + frame_axes` | 0.766 |

Observation:

- The same task-frame invariance failure appears in Qwen3: direct refusal is high, but translation and extraction are zero-refusal.
- Qwen3 preserves more refusal in `summarize_frame` and `rewrite_frame` than Qwen2.5-7B did. This suggests semantic frame type matters, not only "structured vs non-structured".
- Direct refusal is nearly perfectly readable inside direct prompts, but frame axes are still needed for cross-frame prediction.

### 15. Mistral-7B Base Control Result

设置：

- Model: `mistralai/Mistral-7B-v0.1`
- Prompt set: same medium frame-invariance prompts.
- Layers: `14,19,24,28`.
- Max generated tokens: `48`.
- Exported vectors: `11672 x 4096`.
- Server output: `$JCF_CACHE/vector_space_viz/runs/frame_invariance_cross_model_selected/mistral_7b_base_frame_m50_tok48/`
- Local summary: `src/vector_space_viz/figs/mistral_7b_base_frame_invariance_m50_tok48/`

Prompt-level refusal:

| Condition family | Refusal rate |
|---|---:|
| Direct request | 0.000 |
| Semantic transformations | 0.000 |
| Single-record SBQ | 0.000 |
| Multi-record SBQ | 0.000 |

Observation:

- Mistral-7B base is a useful negative control, not evidence against the hypothesis.
- Because it has no refusal positives, direct-trained refusal transfer is undefined and skipped.
- The mechanism question only applies to models that instantiate refusal behavior. Base models without refusal cannot have task-frame-invariant refusal.

### 16. Cross-model Observation

- Qwen2.5-7B, Qwen2.5-0.5B, and Qwen3-4B all show direct-request refusal that weakens under semantic or schema task frames.
- The exact pattern differs by model: Qwen2.5-0.5B refuses multi-record SBQ more than single-record SBQ; Qwen2.5-7B shows the opposite; Qwen3 is especially format-sensitive.
- This supports the new thesis: reframing and batching are not two isolated jailbreak tricks. They are probes for whether refusal survives competing task frames.

### 17. Qwen2.5 Causal Steering Smoke

设置：

- Model: `Qwen/Qwen2.5-7B-Instruct`
- Source vectors: `frame_invariance_qwen2p5_m50`, layer `19`, `query_span`
- Examples: `8` prompts per selected group
- Interventions: add direct-refusal axis, subtract semantic task-frame axis, subtract schema axis
- Server output: `$JCF_CACHE/vector_space_viz/runs/axis_steering_qwen2p5_m50_smoke_offline/`
- Local summary: `src/vector_space_viz/figs/axis_steering_qwen2p5_m50_smoke_offline/`

Overall refusal rates:

| Intervention | n | Refusal rate |
|---|---:|---:|
| baseline | 48 | 0.104 |
| add_direct_10 | 48 | 0.167 |
| add_direct_20 | 48 | 0.188 |
| sub_task_10 | 24 | 0.000 |
| add_direct_10_sub_task_10 | 24 | 0.000 |
| sub_schema_10 | 16 | 0.125 |
| add_direct_10_sub_schema_10 | 16 | 0.250 |

Key mechanism-level readout:

| Mechanism | Baseline | Best intervention | Refusal rate under best intervention |
|---|---:|---|---:|
| `direct_single` | 0.500 | `add_direct_10` / `add_direct_20` | 0.625 |
| `multi_record_sbq/json` | 0.125 | `add_direct_10_sub_schema_10` | 0.500 |
| `single_record_sbq/json` | 0.000 | `sub_schema_10` | 0.125 |
| `translate_frame` | 0.000 | `add_direct_20` | 0.125 |
| `rewrite_frame` | 0.000 | no flip | 0.000 |
| `extract_frame` | 0.000 | no flip | 0.000 |

Observation:

- The smoke supports partial causal control for direct prompts and multi-record SBQ.
- The same layer-19 intervention does not flip rewrite/extract semantic frames, and only weakly affects translation.
- This is important for the mechanism story. A single direct-refusal direction is not sufficient to restore task-frame invariance under all semantic transformations.
- The next run is a layer/scale sweep over layers `19,24,28` with stronger frame/direct scales to test whether semantic frames need later-layer intervention or whether they are governed by a different readout.

Layer/scale sweep:

- Server output: `$JCF_CACHE/vector_space_viz/runs/axis_steering_qwen2p5_m50_layer_scale_sweep_20260526/`
- Local summary: `src/vector_space_viz/figs/axis_steering_qwen2p5_m50_layer_scale_sweep_20260526/`
- Layers: `19,24,28`
- Scales: `direct=20,40`, `frame=20`

Overall refusal rates:

| Intervention | Layer 19 | Layer 24 | Layer 28 |
|---|---:|---:|---:|
| baseline | 0.104 | 0.104 | 0.104 |
| add_direct_20 | 0.188 | 0.104 | 0.125 |
| add_direct_40 | 0.000 | 0.104 | 0.125 |
| sub_task_20 | 0.000 | 0.000 | 0.042 |
| add_direct_20_sub_task_20 | 0.125 | 0.000 | 0.000 |
| sub_schema_20 | 0.250 | 0.062 | 0.062 |
| add_direct_20_sub_schema_20 | 0.375 | 0.125 | 0.062 |

Best mechanism-level effects:

| Mechanism | Best layer | Baseline | Best intervention | Best refusal rate |
|---|---:|---:|---|---:|
| `direct_single` | 19 / 28 | 0.500 | `add_direct_20` | 0.625 |
| `multi_record_sbq/json` | 19 | 0.125 | `add_direct_20_sub_schema_20` | 0.500 |
| `single_record_sbq/json` | 19 | 0.000 | `add_direct_20_sub_schema_20` | 0.250 |
| `translate_frame` | 19 / 28 | 0.000 | `add_direct_20` or `sub_task_20` | 0.125 |
| `rewrite_frame` | 19 | 0.000 | `add_direct_20_sub_task_20` | 0.125 |
| `extract_frame` | 19 | 0.000 | `add_direct_20_sub_task_20` | 0.125 |

Observation:

- Layer `19` is the only clearly useful causal layer in this sweep.
- Later layers have larger linear axis deltas but weaker causal steering effects. This shows that high linear separability is not enough to identify a good intervention site.
- Multi-record SBQ is the strongest causal result: layer-19 combined direct/schema steering raises refusal from `0.125` to `0.500`.
- Semantic frames remain difficult. Even the best interventions only reach `0.125`, so the direct-refusal axis is not a universal control knob for task-frame invariance.
- This motivated the hook-timing run below rather than another scale increase.

### 18. Qwen2.5 Hook-Timing Steering Run

设置：

- Model: `Qwen/Qwen2.5-7B-Instruct`
- Source vectors: `frame_invariance_qwen2p5_m50`, layer `19`, `query_span`
- Server output: `$JCF_CACHE/vector_space_viz/runs/axis_steering_qwen2p5_m50_hook_timing_20260529/`
- Local summary: `src/vector_space_viz/figs/axis_steering_qwen2p5_m50_hook_timing_20260529/`
- Examples: `12` prompts per selected group
- Scales: `direct=20`, `frame=20`

Overall refusal rates:

| Intervention | all | prompt only | first token only | generated only |
|---|---:|---:|---:|---:|
| baseline | 0.139 | 0.139 | 0.139 | 0.139 |
| add_direct_20 | 0.194 | 0.153 | 0.153 | 0.153 |
| sub_task_20 | 0.000 | 0.000 | 0.000 | 0.028 |
| add_direct_20_sub_task_20 | 0.111 | 0.000 | 0.000 | 0.000 |
| sub_schema_20 | 0.250 | 0.042 | 0.083 | 0.208 |
| add_direct_20_sub_schema_20 | 0.417 | 0.042 | 0.042 | 0.417 |

Best mechanism-level effects:

| Mechanism | Baseline | Best mode | Best intervention | Best refusal |
|---|---:|---|---|---:|
| `multi_record_sbq/json` | 0.083 | all | `add_direct_20_sub_schema_20` | 0.500 |
| `multi_record_sbq/json` | 0.083 | generated only | `add_direct_20_sub_schema_20` | 0.417 |
| `single_record_sbq/json` | 0.083 | generated only | `add_direct_20_sub_schema_20` | 0.417 |
| `single_record_sbq/json` | 0.083 | all | `add_direct_20_sub_schema_20` | 0.333 |
| semantic frames | 0.000 | all | task/direct variants | 0.083-0.167 |

Observation:

- Prompt-only and first-token-only steering do not reproduce the SBQ effect.
- Generated-only steering is almost as strong as all-token steering for schema prompts.
- This suggests the layer-19 causal effect is mainly in generation-time continuation dynamics, not only prompt interpretation or the assistant-start readout.
- Semantic frames remain weak under all timing modes. This strengthens the claim that schema completion and semantic transformations are not the same causal pathway.
- Next causal step: activation patching or multi-direction interventions, not more scalar steering along the same axes.

## Observations

### Updated Paper Direction

The paper should not claim only that `reframing` and `batching` improve ASR. Those are now treated as probes that reveal a deeper question:

**Is refusal invariant to task-frame transformations of the same harmful intent, or is refusal conditioned on the task representation in which the intent is embedded?**

Working hypothesis:

1. Open-weight models contain a measurable refusal-related direction/subspace in intermediate residual representations.
2. Direct harmful prompts, structured single-record prompts, and multi-record batched prompts may produce refusals through the same subspace or through different mechanism-specific subspaces.
3. Translation, summarization, rewriting, extraction, schema completion, and batching are controlled task-frame transformations, not a list of unrelated attacks.
4. Cross-model ASR differences can be explained by how strongly each model preserves refusal-space activation when another task axis is present: semantic transformation, schema completion, per-record batching, field completion, or mixed benign/harmful transformation.
5. The observed behavioral phenotypes remain useful, but they become downstream evidence for representational differences rather than the whole paper contribution.

### Current `vector_space_viz` Audit

Current folder: `src/vector_space_viz/`.

What exists:

- `visualize_embeddings.py`: embeds generated/evaluated text with sentence-transformer models, then plots t-SNE/SVD by `batch` vs `non_batch` and `harmful` vs `non_harmful`.
- `export_hidden_states.py`: encodes text using an arbitrary HuggingFace encoder such as `distilbert-base-uncased`, saves only last hidden states plus metadata.
- `generate_article_style_dataset.py`: creates paired harmless/harmful same-verb prompts.
- Existing figures and metadata are preliminary. `verbctx100_last_token_overlap_report.txt` shows `total=200`, `unique_last_token_vectors=6`, `exact_overlap_points=194`, which means the current last-token extraction is likely dominated by shared padding/EOS behavior and is not reliable as refusal-space evidence.

Main limitation:

- The current visualizations are mostly **output-text semantic embeddings**, not target-model internal refusal representations.
- The current hidden-state script does not run the attacked model as a causal LM over the actual direct/SBQ/batch prompts, does not collect all layers, does not track decision token positions, and does not support cross-mechanism probe analyses.
- t-SNE/SVD figures are useful as diagnostics, but not enough for a mechanism claim. The paper needs layerwise probes, direction similarity, and cross-mechanism generalization.

### Mechanistic Questions

Core questions to answer:

1. Shared refusal space: if we train a refusal probe/direction on direct harmful vs refused examples, does it predict refusal/compliance under single-record SBQ, multi-record SBQ, XML, Markdown table, and field-name variants?
2. Mechanism-specific refusal: if a probe trained on one mechanism fails on another, are the refusal directions orthogonal/low-cosine or merely shifted by schema/task directions?
3. Schema-following competition: do SBQ and table/XML prompts activate a schema-completion direction that negatively correlates with refusal projection?
4. Batch dilution: in multi-record prompts, does refusal projection decrease by record position, by mixed benign/harmful context, or by batch size?
5. Causal check: for open models, can adding the refusal direction or ablating a schema/task direction restore refusals in high-ASR structured/batched settings?

### Evidence Status

- Strong behavioral evidence: four-condition causal decomposition for Gemini-2.5-Flash, GPT-5-Chat, DeepSeek-V3.1, and Qwen-Plus.
- Strong behavioral evidence: Gemini format ablation shows the effect is not JSON-specific; XML and Markdown table are stronger than JSON.
- Moderate behavioral evidence: GPT-5-Chat batch-size sweep supports batch amplification up to b=10-16, with degradation at b=32/64 due to output pressure rather than safety.
- Strong preliminary internal-state evidence: Qwen2.5-7B shows a strong direct refusal direction in direct prompts, weak direct-refusal transfer under SBQ, and a schema axis that is negatively correlated with refusal.
- Cross-model internal-state evidence now includes Qwen2.5-7B, Qwen2.5-0.5B, Qwen3-4B, and Mistral-7B base control. `sea-lion-7b-instruct` is intentionally excluded.
- Needs quality control: XML/table extraction and high-ASR examples still need manual or second-judge validation before submission.

## TODO

### P0. Rewrite the Paper Claim

Use this as the new core:

**Structured and batched jailbreaks are behavioral probes for whether refusal is represented as a shared, transferable subspace or as mechanism-specific post-training behavior.**

Implication for paper structure:

1. Behavioral finding: same four-condition setup produces different ASR shifts across model families.
2. Mechanistic question: do these shifts correspond to movement away from a common refusal subspace, activation of a schema-following/task-completion direction, or both?
3. Representational evidence: open-weight hidden-state probes and direction comparisons.
4. Closed-model bridge: behavioral phenotypes and format/batch ablations remain as external evidence.

### P1. Refactor `src/vector_space_viz` Around Representation Probes

Replace the current output-text-only workflow with a structured pipeline:

- `build_probe_dataset.py`: create paired prompts with stable `query_id`, `mechanism`, `format`, `batch_size`, `record_position`, `field_name`, `harmful_label`, and `expected_refusal`. Initial scaffold added.
- `export_causal_lm_states.py`: run open-weight causal LMs on the actual prompts and save hidden states for all layers. Initial scaffold added.
- `fit_refusal_probes.py`: train linear probes/directions for refusal/compliance, harmful/harmless, schema-format, and batch/context labels. Initial scaffold added.
- `compare_subspaces.py`: compute cosine similarity, principal angles, CKA/RSA-style similarity, and cross-mechanism probe transfer matrices. Initial direction/projection scaffold added; CKA/RSA remains future work.
- `plot_refusal_space.py`: produce publication figures: layerwise probe AUC, projection summaries, direction-similarity heatmaps, and cross-mechanism transfer heatmaps. Initial scaffold added.
- `deep_dive_refusal_axes.py`: decompose prompt-level vectors into direct-refusal, schema, batch, multi-vs-single-SBQ, and field-name axes. Initial scaffold added and run on Qwen2.5.
- Keep large tensors and raw figures local/ignored. Track only code, small metadata samples, and compact summary CSV/JSON.

### P2. Fix Hidden-state Extraction Design

The next extractor should collect representations from the attacked open-weight model, not from a separate encoder over response text.

Required capture points:

- Prompt-end token: final non-pad input token before generation.
- Harmful-query span mean: average over the embedded user query inside each mechanism.
- Schema-control span: tokens around JSON/XML/table field names such as `response`, `answer`, `unsafe_response`.
- Assistant first generated token: representation immediately before/at first output token.
- Refusal onset token when generated: positions around "I can't", "I cannot", safety policy phrases, or equivalent refusal markers.
- For batch prompts: per-record query span, record position, and field span.

Required storage:

- Save arrays as `.npz`, `.pt`, or zarr under ignored local output paths.
- Save metadata as CSV/Parquet with one row per vector: `model`, `layer`, `query_id`, `mechanism`, `format`, `batch_size`, `position`, `capture_point`, `outcome`, `judge_label`, `schema_valid`.
- Avoid saving huge JSON vector dumps in git.

### P3. Build the Probe Dataset

Minimum mechanisms for the first internal-state study:

- Direct harmful prompt, single query.
- Direct batched harmful prompts, b=10.
- Single-record SBQ, b=1.
- Multi-record SBQ, b=10.
- Format variants: JSON, XML, Markdown table.
- Field-name variants: `response`, `answer`, `unsafe_response`, and neutral names.
- Batch composition variants: all harmful, mixed benign/harmful, and benign-only control.
- Record-position variants: fixed early/middle/late positions for the same harmful query.

Use paired examples whenever possible: the same `query_id` should appear across all mechanisms so plots can show within-query movement rather than only between-sample clusters.

### P4. Run Open-weight Models First

Priority open models:

- Completed: `Qwen/Qwen2.5-7B-Instruct`
- Running/queued: `mistralai/Mistral-7B-Instruct-v0.1`
- Running/queued: `Qwen/Qwen2-7B-Instruct`
- Running/queued: `meta-llama/Meta-Llama-3-8B-Instruct`
- Later if feasible: `google/gemma-3-27b-it`

For each open model:

- Generate/refusal-label outputs for the same probe dataset.
- Export all-layer vectors at P2 capture points.
- Compute layerwise refusal-probe AUC and cross-mechanism transfer.
- Compare direction similarity between direct refusal, SBQ refusal, batch refusal, and format-specific refusal.
- Run `deep_dive_refusal_axes.py` to compare direct-refusal, schema, batch, and field-name axes.

### P5. Analysis Criteria for the New Idea

Evidence for a shared refusal space:

- A refusal probe trained on direct prompts transfers to SBQ/batch/format variants with high AUC.
- Refusal/compliance mean-difference vectors across mechanisms have high cosine similarity or small principal angle.
- Projection onto a shared refusal direction predicts ASR/refusal rate across mechanisms and models.
- Steering along the shared refusal direction increases refusals in high-ASR structured settings.

Evidence for mechanism-specific refusal spaces:

- Cross-mechanism probe transfer is weak while within-mechanism probe AUC is high.
- Direction cosine/principal-angle matrices show separate clusters by mechanism or format.
- Schema/batch probes explain variance that the direct refusal probe cannot explain.
- Steering needs different directions for direct vs SBQ vs batch settings.

Evidence for schema-following competition:

- Schema/field-format direction is high exactly when refusal projection drops.
- Strong schema adherence correlates with lower refusal projection and higher ASR.
- Field-name or format ablations shift the schema direction and ASR together.

Evidence needed for causal mechanism rather than correlation:

- Adding the direct-refusal direction to SBQ/batch residual states should increase refusal if a shared direction is being bypassed.
- Subtracting or damping schema/batch directions should reduce slot-filling compliance if task-frame competition is causal.
- Activation patching from direct refusal prompts to SBQ/batch prompts should identify the layers where the refusal decision becomes frame-dependent.

Current causal experiment:

- Script added: `src/vector_space_viz/run_axis_steering.py`.
- Target model: `Qwen/Qwen2.5-7B-Instruct`.
- Source run: `frame_invariance_qwen2p5_m50`.
- Source vectors: layer `19`, `query_span`.
- Axis estimates:
  - `direct_refusal`: direct-single refusal minus direct-single compliance.
  - `task_frame`: semantic-transformation frames minus direct-single prompts.
  - `schema_frame`: SBQ/schema-completion frames minus direct-single prompts.
- Steering probes:
  - add `direct_refusal` during generation;
  - subtract `task_frame` on translate/rewrite/extract prompts;
  - subtract `schema_frame` on SBQ prompts;
  - combine direct-refusal addition with frame/schema subtraction.
- Interpretation:
  - direct-refusal addition flipping semantic/SBQ frames supports an underweighted shared-refusal direction;
  - frame/schema subtraction flipping behavior supports task-frame readout competition;
  - no flip suggests later-layer or generation-time policy dynamics.

Completed activation-patching validation:

- Script added: `src/vector_space_viz/run_activation_patching.py`.
- Queue script added: `src/vector_space_viz/run_activation_patching_queue.sh`.
- Server run dir: `$JCF_CACHE/vector_space_viz/runs/activation_patching_qwen2p5_m50_l19_20260529/`.
- Local summary: `src/vector_space_viz/figs/activation_patching_qwen2p5_m50_l19_20260529/`.
- Model: `Qwen/Qwen2.5-7B-Instruct`.
- Layer: `19`.
- Source vector: matched `direct_single` prompt-end activation from the same query, requiring the direct prompt to be refused in the metadata.
- Targets: `translate_frame`, `rewrite_frame`, `extract_frame`, `single_record_sbq/json`, and `multi_record_sbq/json`.
- Interventions: activation interpolation with alpha `0.5` and `1.0`, compared against baseline.
- Patch timing modes: `generated_only`, `prompt_only`, `first_token_only`, and `all`.
- Overall baseline refusal is `0.033`.
- Best overall result is `generated_only` with alpha `0.5`: `0.233`.
- Best semantic-frame result is `translate_frame` first-token patching with alpha `1.0`: `0.500`.
- `extract_frame` reaches `0.333`; `rewrite_frame` reaches `0.250`.
- `single_record_sbq/json` improves from `0.083` to `0.333` under generated-only alpha `0.5`.
- `multi_record_sbq/json` does not improve beyond its `0.083` baseline under prompt-end activation transplant.
- Interpretation: semantic frames respond to concrete matched activation patching more than to a single mean-difference axis, while multi-record SBQ still needs schema/batch continuation interventions.
- `alpha=1.0` is not monotonically better and often suppresses refusal, suggesting full state replacement can break the task trajectory.

### P6. Preserve Behavioral Model Phenotypes as External Evidence

Do not drop the existing behavioral tables. Reinterpret them as external phenotypes:

| Phenotype | Models currently matching it | Use in new paper |
|---|---|---|
| Reframing-dominant | Gemini-2.5-Flash, DeepSeek-V3.1 | Suggests structured task framing suppresses or bypasses refusal even without batching |
| Batch-amplification-dominant | GPT-5-Chat | Suggests per-record safety may dilute under shared batch context |
| Hybrid | Qwen-Plus | Suggests both schema/task and batch axes matter |
| Unknown / needs decomposition | Qwen2.5-7B, Mistral-7B, Gemma-3-27B, LLaMA3-8B | Need open-weight hidden states or full four-condition behavioral reruns |

### P7. Data Quality and Validation

- Manual audit: stratified examples from direct, SBQ, XML/table, and high-ASR batches.
- Second judge: rerun a subset with an independent judge to estimate label sensitivity.
- Recheck Mistral with longer timeout because previous 3/30 target-call timeouts may understate ASR.
- Separate `attack_times=1` ablation results from `attack_times=3` main behavioral results.
- Treat b=32/64 degradation as output-format/reliability collapse unless valid response rate is controlled.
## 2026-07-19 Paired reframing projection

### Observation

- Added a fold-local paired projection for Qwen2.5-7B layer 19.
- The displayed 20 query identities are held out when the Direct-refusal and schema axes are fitted.
- Direct to JSON SBQ Single moves all `20/20` pairs toward the schema axis with median shift `+27.919`.
- The median Direct-refusal-axis shift is only `-1.831`, while `10/20` pairs change from lexical refusal to non-refusal.
- The data support frame-axis displacement and possible competition. They do not support a literal universal refusal region or a t-SNE distance claim.

### TODO

- Replicate the fold-local paired projection for all five folds and summarize shift distributions without pooling coordinates from different bases.
- Repeat the same prespecified visualization for Qwen2.5-0.5B and Qwen3-4B.
- Compare JSON, table, and XML trajectories under the same held-out basis.
- Keep deterministic PCA-to-t-SNE as an appendix diagnostic only.
