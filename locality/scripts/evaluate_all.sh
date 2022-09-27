export CUDA_VISIBLE_DEVICES=${1:-0}
#LOCALITY_TEXT=${2:-eyes} # skin, nose, eyes, eyebrows, mouth, lips, hair

#LOCALITY_TEXT=("eyebrows" "hair" "hair" "lips" "nose" "nose" \
#               "hair" "eyebrows" "eyes" "hair" "ears" "eyes" \
#               "skin" "lips" "hair" "face" "hair" "mouth" \
#               "skin" "nose" "face" "face" "hair" "lips" \
#               "face" "mouth" "face" "eyes" "hair" "face")

#EDIT_PROMPT=("arched_eyebrows" "bald" "bangs" "big_lips" "big_nose" "blue_nose" \
#            "brown_hair" "bushy_eyebrows" "closed_eyes" "curly_hair" "elf_ear" "eyeglasses" \
#            "goatee" "green_lips" "grey_hair" "happy" "mustache" "open_mouth" \
#            "pale_skin" "purple_nose" "sad" "smiling" "straight_hair" "yellow_lips" \
#            "surprised" "angry" "purple_hair" "disgusted")

LOCALITY_TEXT=("face" "eyes")
EDIT_PROMPT=("beard" "eyeglasses2")


#ORIG_DIR="/home/nas4_user/sungwonhwang/logs/fenerf_styleclip_e4e/outputs/RGB_orig"
ORIG_DIR="/home/nas4_user/sungwonhwang/logs/fenerf_styleclip_e4e/outputs/e4e_inverted_images/RGB"
#LOG_DIR="/home/nas4_user/sungwonhwang/logs/fenerf_styleclip_e4e/outputs/metrics_locality"
LOG_DIR="/home/nas4_user/sungwonhwang/logs/fenerf_styleclip_e4e/outputs/metrics_locality_styleclip_only"
mkdir $LOG_DIR
for i in "${!EDIT_PROMPT[@]}"; do
    #EDIT_DIR="/home/nas4_user/sungwonhwang/logs/fenerf_styleclip_e4e/outputs/"${EDIT_PROMPT[i]}"/fenerf_final_beta"
    EDIT_DIR="/home/nas4_user/sungwonhwang/logs/fenerf_styleclip_e4e/outputs/"${EDIT_PROMPT[i]}"/edited"
    python main.py --original_image_dir $ORIG_DIR \
                   --edited_image_dir $EDIT_DIR \
                   --locality_text ${LOCALITY_TEXT[i]} \
                   --log_dir $LOG_DIR/edit_text_${EDIT_PROMPT[i]}_locality_${LOCALITY_TEXT[i]}_in_out_mse.txt
done