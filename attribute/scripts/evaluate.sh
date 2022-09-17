export CUDA_VISIBLE_DEVICES=${1:-0}
ATTRIBUTE_TEXT=${2:-Male}
ORIG_DIR="/home/nas4_user/sungwonhwang/logs/fenerf_styleclip_e4e/outputs/RGB_orig"
EDIT_DIR="/home/nas4_user/sungwonhwang/logs/fenerf_styleclip_e4e/outputs/${ATTRIBUTE_TEXT}/fenerf_final_beta"
#EDIT_DIR="/home/nas4_user/sungwonhwang/logs/fenerf_styleclip_e4e/outputs/${ATTRIBUTE_TEXT}/edited"
LOG_DIR="/home/nas4_user/sungwonhwang/logs/fenerf_styleclip_e4e/outputs/${ATTRIBUTE_TEXT}"
EVAL_THRES="0.85"

python main.py --original_image_dir $ORIG_DIR \
               --edited_image_dir $EDIT_DIR \
               --attribute_text $ATTRIBUTE_TEXT \
               --eval_thres $EVAL_THRES \
               --log_dir $LOG_DIR