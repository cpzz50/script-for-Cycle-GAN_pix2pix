


python ../datasets/combine_A_and_B.py --fold_A ../Input/A --fold_B ../Input/B --fold_AB ../Input/AB
python ../train.py --dataroot ../Input/AB/train --name palmAndEye --crop_size 128 --model pix2pix --netG FCN  --direction AtoB --dataset_mode aligned --norm batch