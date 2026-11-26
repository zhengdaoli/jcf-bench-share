# JCF Benchmark Research Artifacts

This repository contains the analysis code and derived data for a study of
how language model refusal varies with task framing and batching.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Reproduce the offline analysis

Run these commands from the repository root. They use the included data and
work without model access, credentials, or a GPU.

```bash
python -m unittest discover -s src/vector_space_viz/tests -v
python -m src.analyze_paired_frame_transitions
python -m src.analyze_lexical_frame_transitions
python -m src.vector_space_viz.make_figure1_reframing
python -m src.vector_space_viz.make_figure2_transitions
python -m src.vector_space_viz.make_figure3_measurement
python -m src.vector_space_viz.make_figure4_frame_geometry
python -m src.vector_space_viz.make_main_generalization_summary
```

The paired analyses recalculate counts, confidence intervals, and tests using
anonymous record IDs, the saved judge outcomes, and the saved prefix-lexical
outcomes. The exported labels preserve the original alignment and attempt order.
Figure 4 reads the mean cross-fold cosine matrices. Figure generators write
rendered outputs under `src/vector_space_viz/figs` and `AuthorKit27/Figures`.

- [Experiment tables](tables.md) and [research progress](progress.md)
- [Main comparison](src/outputs/control/main_table_causal_attribution_summary.json)
- [Paired contrasts](src/outputs/control/paired_analysis/paired_frame_transitions.csv)
- [Derived representation and intervention statistics](src/vector_space_viz/figs)
- ART, PAP, and Logic summary metrics under [baselines](src/baselines)

The reproduction scope starts at the saved outcomes. Raw benchmark inputs,
model replies, fitted intervention vectors, and full attack runners remain in
the original research workspace.

## Optional provider check

For a one-request connectivity check with a harmless fixed input, copy
`config.example.json` to `config.local.json` and fill in the relevant values.
The local file is ignored by Git. Environment variables of the same name take
precedence. `JCF_LOCAL_CONFIG` selects a different local file.

```bash
python -m src.smoke --provider openrouter --model qwen/qwen-2.5-7b-instruct
```

The command prints a pass/fail message. OneAPI and other self-hosted endpoints
can be configured with their corresponding URL settings.
