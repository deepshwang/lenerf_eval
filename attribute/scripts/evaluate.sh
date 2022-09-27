export CUDA_VISIBLE_DEVICES=${1:-0}
ATTRIBUTE_TEXT=${2:-Male}
ORIG_DIR="/home/nas4_user/sungwonhwang/logs/fenerf_styleclip_e4e/outputs/RGB_orig"
EDIT_DIR="/home/nas4_user/sungwonhwang/logs/fenerf_styleclip_e4e/outputs/${ATTRIBUTE_TEXT}/fenerf_final_beta"
LOG_DIR="/home/nas4_user/sungwonhwang/logs/fenerf_styleclip_e4e/outputs/metrics_attribute"
#ORIG_DIR="/home/nas4_user/sungwonhwang/logs/fenerf_styleclip_e4e/outputs/e4e_inverted_images/RGB"
#EDIT_DIR="/home/nas4_user/sungwonhwang/logs/fenerf_styleclip_e4e/outputs/${ATTRIBUTE_TEXT}/edited"
#LOG_DIR="/home/nas4_user/sungwonhwang/logs/fenerf_styleclip_e4e/outputs/metrics_attribute_styleclip_only"
mkdir $LOG_DIR
EVAL_THRES="0.9"

python main.py --original_image_dir $ORIG_DIR \
               --edited_image_dir $EDIT_DIR \
               --attribute_text $ATTRIBUTE_TEXT \
               --eval_thres $EVAL_THRES \
               --log_dir $LOG_DIR/eval_stats_${ATTRIBUTE_TEXT}.txt