export CUDA_VISIBLE_DEVICES=${1:-0}
#ATTRIBUTE_TEXT=("big_lips" "smiling" "eyeglasses" "curly_hair" "straight_hair" "bald" \
#                "bangs" "arched_eyebrows" "bushy_eyebrows" "grey_hair" "brown_hair" \
#                "pale_skin" "mustache" "goatee" "big_nose")

ATTRIBUTE_TEXT=("eyeglasses2")

ORIG_DIR="/home/nas4_user/sungwonhwang/logs/fenerf_styleclip_e4e/outputs/RGB_orig"
#ORIG_DIR="/home/nas4_user/sungwonhwang/logs/fenerf_styleclip_e4e/outputs/e4e_inverted_images/RGB"
LOG_DIR="/home/nas4_user/sungwonhwang/logs/fenerf_styleclip_e4e/outputs/metrics_attribute"
mkdir $LOG_DIR
EVAL_THRES="0.9"
for i in "${!ATTRIBUTE_TEXT[@]}"; do
    EDIT_DIR="/home/nas4_user/sungwonhwang/logs/fenerf_styleclip_e4e/outputs/"${ATTRIBUTE_TEXT[i]}"/fenerf_final_beta"
    echo $EDIT_DIR
    python main.py --original_image_dir $ORIG_DIR \
                --edited_image_dir $EDIT_DIR \
                --attribute_text ${ATTRIBUTE_TEXT[i]} \
                --eval_thres $EVAL_THRES \
                --log_dir $LOG_DIR/eval_stats_${ATTRIBUTE_TEXT[i]}.txt
done