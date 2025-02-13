

We need to use a  NVIDIA A100-SXM4-80GB or NVIDIA H100 80GB HBM3 for the LPIPS to run properly.

We launch using:

'''
accelerate launch train_lpips_emonet_text_to_image.py
'''

## CUDA Compatibility

This program requires **CUDA 11.8**. Please ensure that you have the correct version installed before proceeding.



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


Inside of EmocaProcessed_38k/geometry_detail and EmocaProcessed_38k/inputs:
- We have 38k cutoff of the original 41k Affectnet Images that are balanced by classes and results.
The cutoffs have the Flame render in "geometry_detail" and the simple processed original images cut around the faces in
"inputs"

- As of 13/10/2024, we want to modify this to have a Render on a image canvas sized like the original as if it could be
superimposed on top, like it does with the inputs.
- We can then process the original, for depth, semantics, and skeleton images, that would be used on a finetune.
- We also need to modify the loss to incorporate the EmonetLoss.

 
- The Semantic mapping is not really useful for faces as of right now.
We will use the following depth mapping: https://github.com/isl-org/ZoeDepth
We will use the following for face alignment: https://github.com/1adrianb/face-alignment
This is already used by EMOCA so it might be a repeat?


The best training is done with: 
final-sd-model-finetuned-l192_lpips08-emonet08-snr08-lr56-1024pics_224res


## Loss of the neural network

In this fine-tuning process of Stable Diffusion, the optimization incorporates multiple loss functions to enhance perceptual quality and emotional expressiveness. The primary loss component is the **L1 loss** (weighted at **0.92**), which ensures pixel-wise accuracy by minimizing absolute differences between predicted and target images.  

To improve perceptual similarity, the **LPIPS loss** (Learned Perceptual Image Patch Similarity, weighted at **0.08**) is introduced. Unlike traditional pixel-based losses, LPIPS leverages deep network feature activations to measure structural and perceptual differences, aligning the generated images more closely with human perception.  

Beyond structural fidelity, emotional coherence is refined through **EMONET-based loss terms**, which guide the model towards capturing emotional attributes. The **valence loss (0.03)** and **arousal loss (0.03)** quantify affective states, ensuring that generated images align with expected emotional intensities. Additionally, the **expression loss (0.02)** refines facial expressiveness, helping the model better reproduce nuanced emotional states.  

This weighted combination balances low-level reconstruction accuracy with high-level perceptual and emotional fidelity, optimizing the fine-tuning process for visually and affectively meaningful outputs.
