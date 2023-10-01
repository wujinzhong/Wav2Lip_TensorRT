# **Wav2Lip_TensorRT**

This is the TensorRT optimized Wav2Lip pipeline, thanks to Rudrabha for the wonderful Wav2Lip implementation at here, https://github.com/Rudrabha/Wav2Lip/tree/master. Please refer to the original repo for tech background. 

# System config
Please refer the original repo for system config, here is my config scripts just for reference, I modified requirement.txt to use latest torch.

> #follow the instructions here for system config, https://github.com/Rudrabha/Wav2Lip/tree/master. 
> 
> #download sample video and audio here, https://bhaasha.iiit.ac.in/lipsync/.
> 
> #download pre-trained models from here, https://github.com/Rudrabha/Wav2Lip/tree/master. 
>
> clear && CUDA_VISIBLE_DEVICES=1 /usr/local/bin/nsys profile /opt/conda/bin/python inference.py --checkpoint_path ./checkpoints/wav2lip.pth --face ./samples/game.mp4 --audio ./samples/game.wav --export_wav2lip_onnx=./onnx_trt/wav2lip_bs1_fp32_6000.onnx --export_wav2lip_trt=./onnx_trt/wav2lip _bs1_fp32_6000.engine --warm_up=2 --export_ s3fd _onnx=./onnx_trt/s3fd_bs43_fp32_6000.onnx --export_s3fd _trt=./onnx_trt/s3fd_bs43_fp32_6000.engine

I use system config:

Ubuntu 18.04

GPU Driver 530

CUDA 11.7

TensorRT 8.6.0

GPU RTX 6000

Torch 2.0.1+cu117


