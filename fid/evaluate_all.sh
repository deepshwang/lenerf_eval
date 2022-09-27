export CUDA_VISIBLE_DEVICES=${1:-0}
#EDIT_PROMPT=("arched_eyebrows" "bald" "bangs" "big_lips" "big_nose" "blue_nose" \
#            "brown_hair" "bushy_eyebrows" "closed_eyes" "curly_hair" "elf_ear" "eyeglasses" \
#            "goatee" "green_lips" "grey_hair" "happy" "mustache" "open_mouth" \
#            "pale_skin" "purple_nose" "sad" "smiling" "straight_hair" "yellow_lips" \
#            "surprised" "angry" "purple_hair" "disgusted")

EDIT_PROMPT=("eyeglasses2" "beard")
REAL_DIR="/home/nas4_user/sungwonhwang/logs/fenerf_styleclip_e4e/outputs/RGB_real"
ORIG_DIR="/home/nas4_user/sungwonhwang/logs/fenerf_styleclip_e4e/outputs/RGB_orig"
#EDIT_DIR="/home/nas4_user/sungwonhwang/logs/fenerf_styleclip_e4e/outputs/${EDIT_PROMPT}/edited"
#ORIG_DIR="/home/nas4_user/sungwonhwang/logs/fenerf_styleclip_e4e/outputs/e4e_inverted_images/RGB"
LOG_DIR="/home/nas4_user/sungwonhwang/logs/fenerf_styleclip_e4e/outputs/metrics_fid"
mkdir $LOG_DIR
cd pytorch-fid/src/pytorch_fid
for i in "${!EDIT_PROMPT[@]}"; do
    EDIT_DIR="/home/nas4_user/sungwonhwang/logs/fenerf_styleclip_e4e/outputs/"${EDIT_PROMPT[i]}"/fenerf_final_beta"
    python fid_score.py --real_path ${REAL_DIR} \
                        --original_path ${ORIG_DIR} \
                        --edit_path ${EDIT_DIR} \
                        --log_dir ${LOG_DIR}/${EDIT_PROMPT[i]}_fid.txt
done