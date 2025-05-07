# U-Net_Basic
セグメンテーションモデルU-Netを使った画像推論を行う．
# .env: no such file or firectory
Use 'pip install -r requirements.txt'.
# Folder_list
train
dataset/images/
masks
dataset/masks/
# How to use
train
Use 'python train.py'
predict
Use 'python predict_batch.py --input_dir test_images/ --output_dir predicted_masks/ --model unet_model.pth'
