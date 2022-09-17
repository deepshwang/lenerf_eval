export CUDA_VISIBLE_DEVICES=${1:-0}
LOCALITY_TEXT=${2:-eyes}
ORIG_DIR="/home/nas4_user/sungwonhwang/logs/FENeRF/render_ffhq/StyleCLIP_edit/surprised/inference_results/original"
EDIT_DIR="/home/nas4_user/sungwonhwang/logs/FENeRF/render_ffhq/StyleCLIP_edit/surprised/inference_results/edited"
LOG_DIR="/home/nas4_user/sungwonhwang/logs/FENeRF/render_ffhq/StyleCLIP_edit/surprised/inference_results"

python main.py --original_image_dir $ORIG_DIR \
               --edited_image_dir $EDIT_DIR \
               --locality_text $LOCALITY_TEXT \
               --log_dir $LOG_DIR