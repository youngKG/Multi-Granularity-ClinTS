export CUDA_VISIBLE_DEVICES=0,1,2,3

# base
# python Main.py --root_path /home/path/to/root/ --data_path path/to/data/ --d_model 16 --batch 80 --lr 1e-3 --epoch 40 --patience 5 --log path/to/log --save_path path/to/save --task 'mor' --seed 0 --dp_flag

# hourly segmentation
# python Main.py --root_path /home/path/to/root/ --data_path path/to/data/ --d_model 16 --batch 80 --lr 1e-3 --epoch 40 --patience 5 --log path/to/log --save_path path/to/save --task 'mor' --seed 0 --dp_flag --hie

# adaptive segmentation
python Main.py --root_path /home/path/to/root/ --data_path path/to/data/ --d_model 16 --batch 80 --lr 1e-3 --epoch 40 --patience 5 --log path/to/log --save_path path/to/save --task 'mor' --seed 0 --dp_flag --hie --adpt