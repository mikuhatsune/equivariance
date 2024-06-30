torchrun --nproc_per_node=4 main.py --dev=0,1,2,3 --tag=k3_eq1e-4_last_conv1 --k=3 --equi_coef=1e-4 --task=depth --equi_layers=last_conv1

