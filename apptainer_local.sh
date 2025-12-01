#!/bin/bash

source ~/.bashrc

verl_dir='/home/vince/cofa/verl'
apptainer_image_path=/home/vince/cofa/verl_vllm.sif

# NOTE: this flag requires this installed to work https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html, the --nv flag should hopefully work on zaratan...
~/bin/apptainer exec --bind $verl_dir --nvccli $apptainer_image_path bash run_gsm_lora.sh

