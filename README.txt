

We need to use a  NVIDIA A100-SXM4-80GB or NVIDIA H100 80GB HBM3 for the LPIPS to run properly.

We launch using:

'''
accelerate launch train_lpips_text_to_image.py
'''


Regarding the NVIDIA A100-SXM4-80GB:
condor_submit_bid 1000 -i -append request_memory=281920 -append request_cpus=10 -append request_disk=100G -append request_gpus=2 -append 'requirements = CUDADeviceName == "NVIDIA A100-SXM4-80GB"'
For the (L1 + LPIPS) loss version we can use a batch size 4 
For the L1 loss version we can use a batch size 8

or:

Regarding the NVIDIA H100 80GB HBM3:
condor_submit_bid 1000 -i -append request_memory=281920 -append request_cpus=10 -append request_disk=100G -append request_gpus=2 -append 'requirements = CUDADeviceName == "NVIDIA H100 80GB HBM3"'
For the (L1 + LPIPS) loss version we can use a batch size
For the L1 loss version we can use a batch size

Overall we aim for a Total train batch size (w. parallel, distributed & accumulation) = 1024
