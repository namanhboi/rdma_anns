#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"


# $SCRIPT_DIR/run_experiment.sh batch_size_8 1 bigann 100M SINGLE_SERVER distributed 8 8 false
# $SCRIPT_DIR/run_experiment.sh batch_size_8 1 bigann 100M SINGLE_SERVER distributed 16 8 false
# $SCRIPT_DIR/run_experiment.sh batch_size_8 1 bigann 100M SINGLE_SERVER distributed 24 8 false
# $SCRIPT_DIR/run_experiment.sh batch_size_8 1 bigann 100M SINGLE_SERVER distributed 32 8 false
# $SCRIPT_DIR/run_experiment.sh batch_size_8 1 bigann 100M SINGLE_SERVER distributed 48 8 false


# $SCRIPT_DIR/run_experiment.sh batch_size_8 2 bigann 100M STATE_SEND distributed 8 8 false
# $SCRIPT_DIR/run_experiment.sh batch_size_8 3 bigann 100M STATE_SEND distributed 8 8 false
# $SCRIPT_DIR/run_experiment.sh batch_size_8 4 bigann 100M STATE_SEND distributed 8 8 false
# $SCRIPT_DIR/run_experiment.sh batch_size_8 5 bigann 100M STATE_SEND distributed 8 8 false

# $SCRIPT_DIR/run_experiment.sh distributedann_bw_16 2 bigann 10M DISTRIBUTED_ANN local 2 8 false 16 4 false true
# $SCRIPT_DIR/run_experiment.sh distributedann_bw_32 2 bigann 10M DISTRIBUTED_ANN local 2 8 false 32 4 false true
# $SCRIPT_DIR/run_experiment.sh distributedann_bw_64 2 bigann 10M DISTRIBUTED_ANN local 2 8 false 64 4 false true

# $SCRIPT_DIR/run_experiment.sh bigann_1B_10_server 10 bigann 1B DISTRIBUTED_ANN distributed 6 8 false 4 20 false false 0
# $SCRIPT_DIR/run_experiment.sh bigann_1B_10_server 10 bigann 1B DISTRIBUTED_ANN distributed 6 8 false 8 20 false false 0
# $SCRIPT_DIR/run_experiment.sh bigann_1B_10_server 10 bigann 1B DISTRIBUTED_ANN distributed 7 8 false 16 10 false false 0
# $SCRIPT_DIR/run_experiment.sh bigann_1B_10_server 10 bigann 1B DISTRIBUTED_ANN distributed 6 8 false 32 20 false false 0
# $SCRIPT_DIR/run_experiment.sh bigann_1B_10_server 10 bigann 1B DISTRIBUTED_ANN distributed 6 8 false 64 10 false false 0
# $SCRIPT_DIR/run_experiment.sh bigann_1B_10_server 10 bigann 1B DISTRIBUTED_ANN distributed 6 8 false 128 20 false false 0

# $SCRIPT_DIR/run_experiment.sh test \
# 			      2 \
# 			      bigann \
# 			      10M \
# 			      DISTRIBUTED_ANN \
# 			      local \
# 			      3 \
# 			      8 \
# 			      false \
# 			      8 \
# 			      3 \
# 			      false \
# 			      false \
# 			      1000



$SCRIPT_DIR/run_experiment.sh test \
			      2 \
			      bigann \
			      10M \
			      STATE_SEND \
			      local \
			      4 \
			      8 \
			      false \
			      1 \
			      1 \
			      false \
			      false \
			      1000
