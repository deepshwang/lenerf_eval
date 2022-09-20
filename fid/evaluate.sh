export CUDA_VISIBLE_DEVICES=${1:-0}
EDIT_PROMPT=${2:-surprised}
REAL_DIR="/home/nas4_user/sungwonhwang/logs/fenerf_styleclip_e4e/outputs/RGB_real"
ORIG_DIR="/home/nas4_user/sungwonhwang/logs/fenerf_styleclip_e4e/outputs/RGB_orig"
EDIT_DIR="/home/nas4_user/sungwonhwang/logs/fenerf_styleclip_e4e/outputs/${EDIT_PROMPT}/fenerf_final_beta"
#EDIT_DIR="/home/nas4_user/sungwonhwang/logs/fenerf_styleclip_e4e/outputs/${EDIT_PROMPT}/edited"
#ORIG_DIR="/home/nas4_user/sungwonhwang/logs/fenerf_styleclip_e4e/outputs/e4e_inverted_images/RGB"
LOG_DIR="/home/nas4_user/sungwonhwang/logs/fenerf_styleclip_e4e/outputs/metrics_fid"
mkdir $LOG_DIR
cd pytorch-fid/src/pytorch_fid
python fid_score.py --real_path ${REAL_DIR} \
                    --original_path ${ORIG_DIR} \
                    --edit_path ${EDIT_DIR} \
                    --log_dir ${LOG_DIR}/${EDIT_PROMPT}_fid.txt