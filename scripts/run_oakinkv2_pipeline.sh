#!/usr/bin/env bash
# Run the full OakInk-v2 pipeline (process -> decompose -> generate_xml ->
# ik_fast -> mjwp_fast) on all 7 single-hand tasks. Saves viser exports for
# IK and MJWP-Fast for later inspection.
#
# Usage:
#   bash scripts/run_oakinkv2_pipeline.sh
set -e

cd "$(dirname "$0")/.."

# Headless OpenGL for mujoco offscreen rendering (video export).
export MUJOCO_GL=${MUJOCO_GL:-egl}

PYTHON=.venv/bin/python
ROBOT=${ROBOT:-xhand}
EMB=right
MANO=~/Downloads/mano_v1_2
# lift_board excluded: bimanual primitive (put_on_lid).
# unplug excluded: object barely moves.
TASKS="pick_spoon_bowl pour_tube stir_beaker uncap_alcohol_burner wipe_board"

# Per-robot IK weights. Sharpa benefits from a higher wrist_pos_cost for tighter
# wrist tracking; xhand uses the default.
case "$ROBOT" in
    sharpa)
        IK_WRIST_POS_COST=2.0
        IK_WRIST_ORI_COST=5.0
        ;;
    *)
        IK_WRIST_POS_COST=0.3
        IK_WRIST_ORI_COST=3.0
        ;;
esac
# Per-task wall-clock cap for mjwp_fast (each task may have multiple revert
# attempts; we cap at 5 minutes per task to keep the runner bounded).
MJWP_TIMEOUT=300

for TASK in $TASKS; do
    echo "================================================================="
    echo "Task: $TASK"
    echo "================================================================="

    echo ">>> [1/4] process_datasets/oakinkv2.py"
    $PYTHON spider/process_datasets/oakinkv2.py \
        --task=$TASK --no-show-viewer --mano-assets-root=$MANO \
        2>&1 | grep -E '(window|z_shift|Saved qpos|primitive=)' | head -3

    echo ">>> [2/4] preprocess/decompose_fast.py (with floor support + stability check)"
    $PYTHON spider/preprocess/decompose_fast.py \
        --dataset-name=oakinkv2 --task=$TASK \
        --embodiment-type=$EMB --data-id=0 --add-floor --check-stability \
        2>&1 | grep -E '(Stability|Updated|Failed)' | head -3

    echo ">>> [3/4] preprocess/generate_xml.py"
    $PYTHON spider/preprocess/generate_xml.py \
        --dataset-name=oakinkv2 --task=$TASK \
        --embodiment-type=$EMB --data-id=0 --robot-type=$ROBOT 2>&1 \
        | grep -E '(Saved model|Failed)' || true

    echo ">>> [4a/4] preprocess/ik_fast.py (save .viser + .mp4)"
    $PYTHON spider/preprocess/ik_fast.py \
        --dataset-name=oakinkv2 --task=$TASK --robot-type=$ROBOT \
        --embodiment-type=$EMB --data-id=0 \
        --wrist-pos-cost=$IK_WRIST_POS_COST \
        --wrist-ori-cost=$IK_WRIST_ORI_COST \
        --no-show-viewer --save-video --save-viser \
        --mano-assets-root=$MANO \
        2>&1 | grep -E '(Saved viser|Saved video|Saved /local|Failed)' | head -6

    echo ">>> [4b/4] examples/run_mjwp_fast.py (save .viser + .mp4, timeout=${MJWP_TIMEOUT}s)"
    timeout --preserve-status --signal=KILL $MJWP_TIMEOUT \
        $PYTHON examples/run_mjwp_fast.py +override=oakinkv2_fast \
        task=$TASK robot_type=$ROBOT embodiment_type=$EMB data_id=0 \
        viewer=viser save_viser=true +wait_on_finish=false \
        save_video=true show_viewer=true 2>&1 \
        | grep -E '(Saved viser|Saved video|Final object tracking|Saved info|Attempt)' || true
done

echo "================================================================="
echo "All 7 tasks complete. Viser exports saved:"
find example_datasets/processed/oakinkv2 -name '*.viser' 2>/dev/null | sort
