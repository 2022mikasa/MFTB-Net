# MFTB-Net

# Datasets
[Synapse](https://www.synapse.org/#!Synapse:syn3193805/wiki/217789) <br />
[ACDC](https://www.creatis.insa-lyon.fr/Challenge/acdc/databases.html) <br />


# train
python train.py --dataset Synapse --cfg configs/swin_tiny_patch4_window7_224_lite.yaml --max_epochs 300 --img_size 224 --base_lr 0.0001 --batch_size 16 <br />
# test
python test.py --dataset Synapse --cfg configs/swin_tiny_patch4_window7_224_lite.yaml --is_savenii --max_epoch 300 --base_lr 0.0001 --img_size 224 --batch_size 16 <br />
# References 

