1. Create the data folder and download CIFAR10 dataset
2. Run on CIFAR10 with $\mathtt{NID1}_{0.05}$ and $\rho=10$
``` python
python main_fedsc.py --data_name cifar10 --num_classes 10 --non_iid_alpha 0.05 --imb_factor 0.1
```
