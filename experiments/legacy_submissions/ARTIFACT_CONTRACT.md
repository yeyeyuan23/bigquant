> These routes are historical research examples, not the final competition upload. The confirmed final route is documented in `../../submissions/ARTIFACT_CONTRACT.md`.

# Historical research artifact contracts

This is a source-only release. No trained model, fitted preprocessing statistics,
market dataset, or historical checkpoint is bundled or downloaded automatically.
Missing files, missing payload keys, and incompatible model state fail explicitly.
There is no random-model or bundled-model fallback.

## Configure an inference route

Set the applicable environment variable to an existing local file that you own or
are authorized to use. Keep artifacts outside the repository. Paths are read at
load time, so you can configure them before calling `main`. For example:

```python
import os
from pathlib import Path

os.environ["BIGQUANT_M_RAW_CHECKPOINT"] = str(
    Path("../private-artifacts/m_raw.pt").resolve()
)
```

Keep each route's sibling Python modules on the import path, including its
`public_artifacts.py`. Run the notebook from that route's directory, or add that
directory to your Python import path. Data access dependencies such as the platform
`dai` library and the relevant numerical libraries must be available separately.
All routes still require your own authorized data sources and correct schemas.

| Route | Environment variable | Artifact structure |
| --- | --- | --- |
| Root `unified_x.py` | `BIGQUANT_X_CHECKPOINT` | PyTorch mapping with `config`, `state_dict`, and `candidate454_spec`; `config.input_dim` must equal 454; candidate spec contains `candidate_ids`, `component_specs`, and `gtja_specs` |
| `m_l5` | `BIGQUANT_M_L5_CHECKPOINT` | PyTorch mapping with `config` for `MicrostructureV2Model` and its network `state_dict` |
| `t_protocol_best` | `BIGQUANT_T_CHECKPOINT` | PyTorch mapping with `config` for `CandidateTemporalConfig` and the temporal network `state_dict` |
| `en454_t_residual_020` | `BIGQUANT_EN454_T_RESIDUAL_CHECKPOINT` | Same temporal payload structure, trained for the residual target; ElasticNet is fitted online |
| `x_tree` | `BIGQUANT_X_TREE_MODEL` | LightGBM text model with exactly 454 features named `candidate__<candidate_id>` in the order of the sibling readable `FROZEN_CANDIDATE_SPEC` |
| `m_multiaxis` | `BIGQUANT_M_MULTIAXIS_WEIGHTS` | JSON mapping with `config`, `raw_fields`, `axis_names`, `preprocessing`, `seed`, `training`, and `state_dict` |

The multi-axis JSON state maps parameter names to objects with `data`, `dtype`, and
`shape`; `dtype` names a PyTorch tensor dtype. `raw_fields` and `axis_names` must
match the runtime's declared order. `preprocessing` contains matching `fields`,
finite per-field `mean`, and finite positive per-field `std`. The runtime validates
field/axis order and statistics and strictly loads all model parameters. Its
`load_model(model_path=...)` also accepts an explicit path instead of the environment
variable. Do not replace trained normalization values with arbitrary constants.

The multi-axis training CLI accepts explicit input/output paths. An optional
`--stats-checkpoint` must point to a trusted PyTorch mapping containing
`preprocessing`; it is not discovered automatically. A training `.pt` checkpoint
and the inference JSON representation are different formats. Convert tensors to the
documented JSON structure only in your trusted local training workflow.

The ElasticNet-only route does not require a pretrained file: it reconstructs
features and fits on the causal rolling window from the data you provide. Fixed
architecture sizes, formula signs, regularization settings, lookbacks, and the
0.20 residual blend are algorithm settings, not retained fitted model values.

## Trust and validation

Use only artifacts you produced or obtained from a trusted source. Loading a model
is not proof that it is safe, correctly trained, or appropriate for the data.
PyTorch loads always use `weights_only=True`; runtimes lacking that capability fail
instead of retrying unrestricted pickle loading. No checkpoint was deserialized
while preparing this source-only release.

Candidate specifications are readable source declarations. The X-tree candidate
specification was preserved as plain Python data rather than encoded bytes; it
contains formula definitions and ordering, not model parameters.

Previous submission-bundle manifests and payloads are not shipped. Historical
aggregate reports and audit metadata remain under `reports/`; their recorded
hashes and validation statuses describe earlier private runs and do not validate
the cleaned source snapshot. Verify your own artifact provenance
and digest, schema/feature order, output contract, causality, and numerical behavior
in your own environment. Static parsing and loader unit tests do not establish
inference correctness, platform acceptance, or previous performance.

## Historical local bundle builders

The retained `build_*_submission.py` tools are historical private-workflow builders.
They may generate locally embedded model bundles, which must never be published or
committed. `replay_frozen_candidate454_models.py` targets the checked-in source-only
M entrypoints and their two-argument loader. For M-l5, pass
`--submission-package experiments/legacy_submissions/m_l5`; a legacy builder-generated bundle uses a
different loader interface and needs a separate adapter before replay. Training and
builder code has not been rerun as part of this source-only release.
