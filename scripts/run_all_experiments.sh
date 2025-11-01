#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"



# $SCRIPT_DIR/run_experiment.sh batch_size_8 1 bigann 100M SINGLE_SERVER distributed 8 8
# $SCRIPT_DIR/run_experiment.sh batch_size_8 1 bigann 100M SINGLE_SERVER distributed 16 8
# $SCRIPT_DIR/run_experiment.sh batch_size_8 1 bigann 100M SINGLE_SERVER distributed 24 8
# $SCRIPT_DIR/run_experiment.sh batch_size_8 1 bigann 100M SINGLE_SERVER distributed 32 8
# $SCRIPT_DIR/run_experiment.sh batch_size_8 1 bigann 100M SINGLE_SERVER distributed 48 8


# $SCRIPT_DIR/run_experiment.sh batch_size_8 2 bigann 100M STATE_SEND distributed 8 8
# $SCRIPT_DIR/run_experiment.sh batch_size_8 3 bigann 100M STATE_SEND distributed 8 8
# $SCRIPT_DIR/run_experiment.sh batch_size_8 4 bigann 100M STATE_SEND distributed 8 8
# $SCRIPT_DIR/run_experiment.sh batch_size_8 5 bigann 100M STATE_SEND distributed 8 8

$SCRIPT_DIR/run_experiment.sh batch_size_8 2 bigann 100M SCATTER_GATHER distributed 8 8
$SCRIPT_DIR/run_experiment.sh batch_size_8 3 bigann 100M SCATTER_GATHER distributed 8 8
$SCRIPT_DIR/run_experiment.sh batch_size_8 4 bigann 100M SCATTER_GATHER distributed 8 8
$SCRIPT_DIR/run_experiment.sh batch_size_8 5 bigann 100M SCATTER_GATHER distributed 8 8

