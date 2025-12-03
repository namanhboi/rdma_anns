#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"


# EXPERIMENT_3_SERVER=qps_recall_bigann_100M_3_server
# $SCRIPT_DIR/run_experiment.sh ${EXPERIMENT_3_SERVER} \
# 			      3 \
# 			      bigann \
# 			      100M \
# 			      DISTRIBUTED_ANN \
# 			      distributed \
# 			      8 \
# 			      8 \
# 			      false \
# 			      8 \
# 			      8 \
# 			      false \
# 			      true \
# 			      0

# sleep 10

# $SCRIPT_DIR/run_experiment.sh ${EXPERIMENT_3_SERVER} \
# 			      3 \
# 			      bigann \
# 			      100M \
# 			      SCATTER_GATHER \
# 			      distributed \
# 			      8 \
# 			      8 \
# 			      false \
# 			      1 \
# 			      8 \
# 			      false \
# 			      true \
# 			      0
# sleep 10



# $SCRIPT_DIR/run_experiment.sh ${EXPERIMENT_3_SERVER} \
# 			      3 \
# 			      bigann \
# 			      100M \
# 			      STATE_SEND \
# 			      distributed \
# 			      8 \
# 			      8 \
# 			      false \
# 			      1 \
# 			      8 \
# 			      false \
# 			      true \
# 			      0
# # sleep 10


# EXPERIMENT_5_SERVER=qps_recall_bigann_100M_5_server
# $SCRIPT_DIR/run_experiment.sh ${EXPERIMENT_5_SERVER} \
# 			      5 \
# 			      bigann \
# 			      100M \
# 			      STATE_SEND \
# 			      distributed \
# 			      8 \
# 			      8 \
# 			      false \
# 			      1 \
# 			      8 \
# 			      false \
# 			      true \
# 			      0
# sleep 10


# $SCRIPT_DIR/run_experiment.sh ${EXPERIMENT_5_SERVER} \
# 			      5 \
# 			      bigann \
# 			      100M \
# 			      SCATTER_GATHER \
# 			      distributed \
# 			      8 \
# 			      8 \
# 			      false \
# 			      1 \
# 			      8 \
# 			      false \
# 			      true \
# 			      0
# sleep 10



# EXPERIMENT_7_SERVER=qps_recall_bigann_100M_7_server
# $SCRIPT_DIR/run_experiment.sh ${EXPERIMENT_7_SERVER} \
# 			      7 \
# 			      bigann \
# 			      100M \
# 			      STATE_SEND \
# 			      distributed \
# 			      8 \
# 			      8 \
# 			      false \
# 			      1 \
# 			      8 \
# 			      false \
# 			      true \
# 			      0
# sleep 10


# $SCRIPT_DIR/run_experiment.sh ${EXPERIMENT_7_SERVER} \
# 			      7 \
# 			      bigann \
# 			      100M \
# 			      SCATTER_GATHER \
# 			      distributed \
# 			      8 \
# 			      8 \
# 			      false \
# 			      1 \
# 			      8 \
# 			      false \
# 			      true \
# 			      0
# sleep 10


EXPERIMENT_10_SERVER=qps_recall_bigann_100M_10_server
# $SCRIPT_DIR/run_experiment.sh ${EXPERIMENT_10_SERVER} \
# 			      10 \
# 			      bigann \
# 			      100M \
# 			      STATE_SEND \
# 			      distributed \
# 			      8 \
# 			      8 \
# 			      false \
# 			      1 \
# 			      8 \
# 			      false \
# 			      true \
# 			      0
# sleep 10


$SCRIPT_DIR/run_experiment.sh ${EXPERIMENT_10_SERVER} \
			      1 \
			      bigann \
			      100M \
			      SINGLE_SERVER \
			      distributed \
			      16 \
			      8 \
			      false \
			      1 \
			      8 \
			      false \
			      true \
			      0
sleep 10
$SCRIPT_DIR/run_experiment.sh ${EXPERIMENT_10_SERVER} \
			      1 \
			      bigann \
			      100M \
			      SINGLE_SERVER \
			      distributed \
			      24 \
			      8 \
			      false \
			      1 \
			      8 \
			      false \
			      true \
			      0
sleep 10



# $SCRIPT_DIR/run_experiment.sh ${EXPERIMENT_10_SERVER} \
# 			      1 \
# 			      bigann \
# 			      100M \
# 			      SIGNLER \
# 			      distributed \
# 			      8 \
# 			      8 \
# 			      false \
# 			      1 \
# 			      8 \
# 			      false \
# 			      true \
# 			      0
# sleep 10

