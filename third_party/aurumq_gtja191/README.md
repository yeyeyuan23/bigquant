# AurumQ GTJA191 minimal runtime

This directory vendors the minimal Python source required to execute the
retained GTJA191 formulas used by `scripts/build_gtja157_feature_matrix.py`.

- Upstream: https://github.com/yupoet/aurumq-rl
- Pinned commit: `5cf7e83637b85e4f855daec16099148b358b89b3`
- License: MIT; see `LICENSE`

The runtime is kept in-tree so a data rebuild does not depend on a mutable
package index or an untracked machine-local clone.
