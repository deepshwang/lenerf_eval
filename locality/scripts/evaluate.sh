export CUDA_VISIBLE_DEVICES=${1:-0}
LOCALITY_TEXT=${2:-eyes}
EDIT_PROMPT=${3:-surprised}
ORIG_DIR="/home/nas4_user/sungwonhwang/logs/fenerf_styleclip_e4e/outputs/RGB_orig"
EDIT_DIR="/home/nas4_user/sungwonhwang/logs/fenerf_styleclip_e4e/outputs/${EDIT_PROMPT}/fenerf_final_beta"
LOG_DIR="/home/nas4_user/sungwonhwang/logs/fenerf_styleclip_e4e/outputs/metrics_locality"
mkdir $LOG_DIR
python main.py --original_image_dir $ORIG_DIR \
               --edited_image_dir $EDIT_DIR \
               --locality_text $LOCALITY_TEXT \
               --log_dir $LOG_DIR/edit_text_${EDIT_PROMPT}_locality_${LOCALITY_TEXT}_in_out_mse.txt