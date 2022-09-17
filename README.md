# In/Out MSE
 1. Download the weights from [here](https://drive.google.com/file/d/1iWd_J8Wa522Szuh3lpBxAedCnpDttGBX/view?usp=sharing) and save it to `locality/checkpoints/segNet-20Class.pth`

 2. Set the following directories in `scripts/evaluate.sh`
    *  **ORIG_DIR** : Directory that includes images before editing
    *  **EDIT_DIR** : Directory that includes images after editing
    *  **LOG_DIR** : Directory to save the evaluation result as text file.

3. Run `bash locality/scripts/evaluate.sh ${DEVICE_NUM} ${LOCALITY_TEXT}`
    * **DEVICE_NUM**: gpu device number to run on 
    * **LOCALITY_TEXT**: Regions to test for
        * skin
        * nose
        * eyes
        * eyebrows
        * mouth
        * hair
        * lips

### Example output
`eyes_in_out_mse.txt`

```
Evaluation text: eyes
Mean in-mse: 0.148
Mean out-mse: 0.081


In-mse: 0.182 for | /home/nas4_user/sungwonhwang/logs/FENeRF/render_ffhq/StyleCLIP_edit/surprised/inference_results/original/00000.png
Out-mse: 0.064 for | /home/nas4_user/sungwonhwang/logs/FENeRF/render_ffhq/StyleCLIP_edit/surprised/inference_results/original/00000.png
...

```


# Attribute Transition Rate (ATR)
1. Download the weights from [here](https://drive.google.com/file/d/1K-ZCiMnbK3CzgXhxg-ag8uFK4DZcRUuN/view?usp=sharing) and save it to `attribute/checkpoints/best_net.pth`

2. Set the following directories in `scripts/evaluate.sh`
    *  **ORIG_DIR** : Directory that includes images before editing
    *  **EDIT_DIR** : Directory that includes images after editing
    *  **LOG_DIR** : Directory to save the evaluation result as text file.

3. Run `bash attribute/scripts/evaluate.sh ${DEVICE_NUM} ${ATTRIBUTE_TEXT}`
    * **DEVICE_NUM**: gpu device number to run on 
    * **ATTRIBUTE_TEXT**: attribute text to test for

### Example output
`eval_stats_curly_hair.txt`
```
Average edit success rate: 0.571
00000.png : 
Success!
logit before edit: 0.00
logit after edit: 1.00

...

00004.png : 
Failure
logit before edit: 0.00
logit after edit: 0.00

...

00006.png : 
Original image is not completely negative
```