# Paper Tables Draft

本共享版保留历史实验数值。下面的来源路径指向当时的实验记录，其中逐条原文与未精简的评估文件保存在研究工作区。

说明：

- 除特别说明外，ASR 均为 query-level proportion，不是百分数；写入论文时可统一乘 100。
- `--` 表示当前输出目录中没有找到可直接使用的同设置结果。
- 主文推荐优先使用 Table 3 作为修稿后的核心主表；Table 1/2 是当前可用 baseline 对比结果，部分模型/方法缺失，需要决定是否补跑后再放主文。

## Table 0. Dataset Composition

Source: `src/all_questions_en.json`, `jailbreakbench_harmful_all.json`

| Category | # Queries |
|---|---:|
| Harassment/Discrimination | 10 |
| Malware/Hacking | 10 |
| Physical harm | 10 |
| Economic harm | 10 |
| Fraud/Deception | 10 |
| Disinformation | 10 |
| Sexual/Adult content | 10 |
| Privacy | 10 |
| Expert advice | 10 |
| Government decision-making | 10 |
| **Total** | **100** |

Safety-level metadata in `src/all_questions_en.json`:

| Safety level | # Queries |
|---|---:|
| low | 3 |
| medium | 67 |
| high | 20 |
| critical | 10 |

## Table 1. Main Effectiveness on Black-Box / API Models

Current available baseline comparison. BatchJail values use the revised multi-record JSON SBQ result when available. Baselines are single-query attacks unless otherwise noted.

Sources:

- BatchJail revised: `src/outputs/control/main_table_causal_attribution_summary.json`
- BatchJail legacy GPT-5-nano: `src/outputs/gpt-5-nano/gpt-5-nano_asr_results.json`
- PAP: `src/baselines/pap/*_evaluation.json`
- Logic: `src/baselines/logic/*_evaluation.json`
- ART: `src/baselines/art/*_evaluation.json`

| Model | BatchJail ASR@3 | PAP ASR@3 | Logic ASR@3 | ART ASR@3 | Best baseline | BatchJail - Best baseline |
|---|---:|---:|---:|---:|---:|---:|
| DeepSeek-V3.1 / deepseek-chat | 1.00 | 0.94 | 0.95 | -- | 0.95 | +0.05 |
| Qwen-Plus | 1.00 | 0.87 | 0.74 | -- | 0.87 | +0.13 |
| GPT-5-nano | 0.17 | 0.70 | 0.32 | 0.12 | 0.70 | -0.53 |

Notes:

- Gemini-2.5-Flash and GPT-5-Chat are omitted from this baseline table because matching baseline runs were not found in the current output tree.
- GPT-5-nano uses the older BatchJail result, not the new causal-attribution run.
- GPT-5-nano Logic ASR@3 is computed on 60 samples in the available file; other entries in this table use 100 samples.
- Logic also has a `qwen-plus` batch-10 evaluation with ASR@3 = 0.79, but the single-query baseline setting is ASR@3 = 0.74.
- ART has an available Qwen-Plus batch-10 evaluation with ASR@3 = 0.00, but no clean single-query ART Qwen-Plus ASR summary was found.

## Table 2. Available Open-Weight Model Results

Current available open-weight results. Qwen2.5, Mistral, and Gemma BatchJail values use the revised multi-record JSON SBQ result.

Sources:

- BatchJail LLaMA3-8B: `src/outputs/llama3_8b/llama3_8b_asr_results.json`
- BatchJail Qwen2.5: `src/outputs/control/qwen2.5-instruct-7/qwen2.5-instruct-7_exp1_multi_record_sbq_batch10_attack3_evaluation.json`
- BatchJail Mistral: `src/outputs/control/mistral-7b-instruct/mistral-7b-instruct_exp1_multi_record_sbq_batch10_attack3_evaluation.json`
- BatchJail Gemma: `src/outputs/control/gemma-3-27b-it/gemma-3-27b-it_exp1_multi_record_sbq_batch10_attack3_evaluation.json`
- PAP / Logic / ART: `src/baselines/*/*_evaluation.json`

| Model | BatchJail ASR@3 | PAP ASR@3 | Logic ASR@3 | ART ASR@3 | Best available method |
|---|---:|---:|---:|---:|---|
| LLaMA3-8B-Instruct | 0.74 | 0.75 | 0.65 | -- | PAP |
| Qwen2.5-Instruct-7B | 0.88 | 0.77 | 0.76 | 0.83 | BatchJail |
| Mistral-7B-Instruct | 0.34 | 0.76 | 0.82 | 0.70 | Logic |
| Gemma-3-27B-IT | 0.99 | 1.00 | 1.00 | 0.98 | PAP / Logic |

Notes:

- Mistral BatchJail had 3/30 target-call subprocess timeouts, counted as empty responses.
- Gemma BatchJail uses `google/gemma-3-27b-it`, not the free `:free` endpoint that previously returned upstream 429 errors.

## Table 3. Main Causal Attribution Across Models

This is the recommended revised main table for the reviewer response. All values are ASR@3 with `attack_times=3`.

Source: `src/outputs/control/main_table_causal_attribution_summary.json`

| Model | Direct Single | Direct Batch b=10 | Single-record JSON SBQ | Multi-record JSON SBQ b=10 | Delta Reframing | Delta Batch under SBQ |
|---|---:|---:|---:|---:|---:|---:|
| Gemini-2.5-Flash | 0.09 | 0.10 | 0.90 | 0.96 | +0.81 | +0.06 |
| GPT-5-Chat | 0.50 | 0.68 | 0.47 | 0.85 | -0.03 | +0.38 |
| DeepSeek-V3.1 / deepseek-chat | 0.06 | 0.10 | 0.99 | 1.00 | +0.93 | +0.01 |
| Qwen-Plus | 0.03 | 0.06 | 0.60 | 1.00 | +0.57 | +0.40 |

Interpretation for paper text:

- Gemini and DeepSeek are primarily driven by task reframing / structured completion.
- GPT-5-Chat shows strong batch amplification under the same SBQ format.
- Qwen-Plus shows both strong task reframing and strong batch amplification.

## Table 4. Full ASR@k for Main Causal Attribution

Source: `src/outputs/control/main_table_causal_attribution_summary.json`

| Model | Condition | ASR@1 | ASR@2 | ASR@3 |
|---|---|---:|---:|---:|
| Gemini-2.5-Flash | Direct Single | 0.05 | 0.08 | 0.09 |
| Gemini-2.5-Flash | Direct Batch b=10 | 0.04 | 0.10 | 0.10 |
| Gemini-2.5-Flash | Single-record JSON SBQ | 0.79 | 0.89 | 0.90 |
| Gemini-2.5-Flash | Multi-record JSON SBQ b=10 | 0.78 | 0.87 | 0.96 |
| GPT-5-Chat | Direct Single | 0.34 | 0.47 | 0.50 |
| GPT-5-Chat | Direct Batch b=10 | 0.39 | 0.67 | 0.68 |
| GPT-5-Chat | Single-record JSON SBQ | 0.30 | 0.40 | 0.47 |
| GPT-5-Chat | Multi-record JSON SBQ b=10 | 0.67 | 0.76 | 0.85 |
| DeepSeek-V3.1 / deepseek-chat | Direct Single | 0.02 | 0.06 | 0.06 |
| DeepSeek-V3.1 / deepseek-chat | Direct Batch b=10 | 0.03 | 0.10 | 0.10 |
| DeepSeek-V3.1 / deepseek-chat | Single-record JSON SBQ | 0.96 | 0.99 | 0.99 |
| DeepSeek-V3.1 / deepseek-chat | Multi-record JSON SBQ b=10 | 0.91 | 1.00 | 1.00 |
| Qwen-Plus | Direct Single | 0.03 | 0.03 | 0.03 |
| Qwen-Plus | Direct Batch b=10 | 0.00 | 0.03 | 0.06 |
| Qwen-Plus | Single-record JSON SBQ | 0.54 | 0.58 | 0.60 |
| Qwen-Plus | Multi-record JSON SBQ b=10 | 0.89 | 1.00 | 1.00 |

Run notes:

- Qwen-Plus direct-single had 3 upstream content-filter exceptions, counted as empty responses.
- Qwen-Plus multi-record SBQ had 1 subprocess timeout, counted as an empty response.

## Table 5. Aggregation vs. Task Reframing Controls on Gemini-2.5-Flash

Source: `src/outputs/control/gemini_25/gemini_25_exp1_control_summary.json`

| Condition | Semantics | Batch size | ASR@1 | ASR@2 | ASR@3 |
|---|---|---:|---:|---:|---:|
| Direct Single | direct instruction | 1 | 0.05 | 0.08 | 0.09 |
| Direct Batch | direct instruction | 10 | 0.04 | 0.10 | 0.10 |
| Plain Batch | weak structure | 10 | 0.01 | 0.01 | 0.01 |
| Bullet Batch | weak structure | 10 | 0.00 | 0.03 | 0.11 |
| Single-record SBQ | structured completion | 1 | 0.79 | 0.89 | 0.90 |
| Multi-record SBQ | structured completion | 10 | 0.78 | 0.87 | 0.96 |

## Table 6. Single-Query Completion Control on Gemini-2.5-Flash

Source: `src/outputs/control/gemini_25/gemini_25_exp2_single_query_completion_summary.json`

| Condition | Semantics | ASR@1 | ASR@2 | ASR@3 | Refusal rate @3 |
|---|---|---:|---:|---:|---:|
| Direct Single | direct instruction | 0.05 | 0.08 | 0.09 | 0.91 |
| Single-record SBQ | structured completion, no batch amplification | 0.79 | 0.89 | 0.90 | 0.10 |
| Multi-record SBQ | structured completion, batched | 0.78 | 0.87 | 0.96 | 0.04 |

## Table 7. Structured Format Ablation on Gemini-2.5-Flash

Settings: `batch_size=10`, `attack_times=1`, `suffix_mode=simple`.

Source: `src/outputs/gemini_25/format_ablation_fixed/gemini_25_format_ablation_fixed_summary.json`

| Format | Batch size | Non-empty extracted | Valid/response rate | ASR |
|---|---:|---:|---:|---:|
| Plain | 10 | 90/100 | 0.90 | 0.00 |
| Bullets | 10 | 100/100 | 1.00 | 0.00 |
| JSON | 10 | 57/100 | 0.57 | 0.55 |
| Half JSON | 10 | 40/100 | 0.40 | 0.39 |
| YAML | 10 | 50/100 | 0.50 | 0.48 |
| XML | 10 | 100/100 | 1.00 | 0.97 |
| Markdown table | 10 | 96/100 | 0.96 | 0.93 |

Suggested placement: structure/format ablation section. This table supports the claim that the effect is not JSON-specific.

## Table 8. Larger Batch-Size Sweep on GPT-5-Chat

Settings: JSON SBQ, `attack_times=1`.

Source: `src/outputs/gpt-5-chat/large_batch_sweep/gpt-5-chat_large_batch_sweep_fixed_summary.json`

| Batch size | Approx prompt tokens | Valid/response rate | ASR | Raw empty batches | Raw response chars min-max |
|---:|---:|---:|---:|---:|---:|
| 1 | 691 | 1.00 | 0.12 | 0 | 981-3,279 |
| 2 | 845 | 1.00 | 0.19 | 0 | 1,238-5,575 |
| 4 | 1,156 | 1.00 | 0.16 | 0 | 3,532-8,829 |
| 6 | 1,536 | 1.00 | 0.38 | 0 | 3,597-11,736 |
| 8 | 1,763 | 1.00 | 0.37 | 0 | 4,334-14,413 |
| 10 | 2,048 | 1.00 | 0.51 | 0 | 5,087-17,513 |
| 16 | 3,000 | 1.00 | 0.47 | 0 | 4,084-13,510 |
| 32 | 5,419 | 0.36 | 0.16 | 1 | 1-27,579 |
| 64 | 10,022 | 0.03 | 0.03 | 1 | 1-1,544 |

Interpretation:

- Batch amplification is visible up to batch sizes 10-16.
- At 32/64, the drop is mostly output-format pressure and low valid extraction rate, not evidence of stronger safety.

## Table 9. Preliminary Defense Baselines on Gemini-2.5-Flash

Source: `src/outputs/defense/gemini_25_defense_baseline_summary.json`

| Defense | ASR@1 | ASR@3 | Filter/refusal rate | Target calls / 100 queries | Filter calls / 100 queries | Notes |
|---|---:|---:|---:|---:|---:|---|
| None | 0.78 | 0.96 | 0.04 | 30 | 0 | Default multi-record SBQ |
| Pre-batch screening | 0.00 | 0.00 | 1.00 | 0 | 100 | Dataset-label input filter; proxy upper bound |
| De-batching | 0.79 | 0.90 | 0.10 | 300 | 0 | Single-record SBQ; removes shared batch context |
| Post-generation filtering | 0.00 | 0.00 | 0.96 | 30 | 300 | Evaluation-judge/proxy output filter; upper bound |

Paper wording note:

- Pre-batch screening and post-generation filtering should be described as oracle/proxy upper bounds, not as full adaptive defenses.
- De-batching alone is insufficient on Gemini because the single-record SBQ remains high.

## Table 10. Experiment Source Map

| Table | Main source file |
|---|---|
| Table 0 | `src/all_questions_en.json`; `jailbreakbench_harmful_all.json` |
| Table 1 | `src/outputs/control/main_table_causal_attribution_summary.json`; `src/baselines/*/*_evaluation.json` |
| Table 2 | `src/outputs/llama3_8b/llama3_8b_asr_results.json`; `src/outputs/control/qwen2.5-instruct-7/*_evaluation.json`; `src/outputs/control/mistral-7b-instruct/*_evaluation.json`; `src/outputs/control/gemma-3-27b-it/*_evaluation.json`; `src/baselines/*/*_evaluation.json` |
| Table 3 | `src/outputs/control/main_table_causal_attribution_summary.json` |
| Table 4 | `src/outputs/control/main_table_causal_attribution_summary.json` |
| Table 5 | `src/outputs/control/gemini_25/gemini_25_exp1_control_summary.json` |
| Table 6 | `src/outputs/control/gemini_25/gemini_25_exp2_single_query_completion_summary.json` |
| Table 7 | `src/outputs/gemini_25/format_ablation_fixed/gemini_25_format_ablation_fixed_summary.json` |
| Table 8 | `src/outputs/gpt-5-chat/large_batch_sweep/gpt-5-chat_large_batch_sweep_fixed_summary.json` |
| Table 9 | `src/outputs/defense/gemini_25_defense_baseline_summary.json` |
