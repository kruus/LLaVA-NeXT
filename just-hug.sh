# vim: sw=2 ts=2 et
root=$HOME          # for laptop
gpu=false

if [ `hostname` = "cipr-slurm-js01" ]; then
  gpu=true
elif [ `hostname` = "snake10" ]; then
  root=/local/kruus   # for snake10
  gpu=true
fi

if $gpu; then
	echo "Good, host may have gpu";
else
	echo "only for gpu host!"
	return 1 2> /dev/null
	exit 1
fi

if [ ! -d "${root}" ]; then
  echo "Please set \$root"
  return 1 2> /dev/null
  exit 1
fi

if [ ! -d "${root}/hug/transformers" ]; then
  echo "Please set run hugsetup.sh to set initial ${root}/hug/... git directories"
  return 1 2> /dev/null
  exit 1
fi

echo "Current envs:"
conda env list
echo ""

if ! conda env list | grep -q '^pyt-cpu '; then stage2=true; fi
if $gpu; then if ! conda env list | grep -q '^pyt '; then stage2=true; fi; fi
if $stage2; then
  cd "${root}/hug/transformers"
  # recreate basic envs: hug and hug-cpu
  # Extensions:
  # - mamba:
  #   - accelerate
  # - pip:
  #   - qwen-vl-utils
  { conda deactivate; conda activate
    echo "CONDA_DEFAULT_ENV = $CONDA_DEFAULT_ENV"
    if conda env list | grep -q '^hug '; then conda remove -y -n hug --all; fi
    if $gpu; then
      conda create -n hug --clone pyt
      conda activate hug;
      mamba install --override-channels -c conda-forge -c pytorch -y \
        'safetensors>=0.4.1' 'tqdm>=4.27' 'deepspeed>=0.9.3' 'pytest>=7.2.0,<8.0.0' \
        'pydantic' 'seaborn' git git-lfs accelerate mamba install einops \
        open-clip-torch pyav triton
      echo "CONDA_DEFAULT_ENV = $CONDA_DEFAULT_ENV"
      # bleeding edge transformers
      pip install pre-commit black pylint qwen-vl-utils 'git+https://github.com/huggingface/transformers'
      #pip install -e '.[dev-torch,testing,quality,decord,deepspeed]';
      # To install the newest AutoAWQ from PyPi, you need CUDA 12.1 installed.
      #      pip install autoawq
      # pip install autoawq@https://github.com/casper-hansen/AutoAWQ/releases/download/v0.2.0/autoawq-0.2.0+cu118-cp310-cp310-linux_x86_64.whl
      # or maybe
      # pip install autoawq@https://github.com/casper-hansen/AutoAWQ.git
      #     NO -- gives pytorch 2.4.0 and cuda 12.x
      # Try install from source
      # torch.cuda.get_device_capability()
      if [ `python -c 'import torch; dc = torch.cuda.get_device_capability(); print(dc[0]*10 + dc[1])'` -lt 75 ]; then
        echo "ohoh - cannot install AutoAWQ (video card too old)"
      else
        mkdir -p ${root}/hug/vidlabel
        cd ${root}/hug/vidlabel
        if [ ! -d "AutoAWQ" ]; then
          git clone https://github.com/casper-hansen/AutoAWQ
        fi
        cd AutoAWQ
        git pull
        pip install -e .

        mkdir -p ${root}/hug/vidlabel
        cd ${root}/hug/vidlabel
        if [ ! -d "AutoAWQ_kernels" ]; then
          git clone https://github.com/casper-hansen/AutoAWQ_kernels
        fi
        cd AutoAWQ_kernels
        git pull
        pip install -e .

        # This requires cuda 12, so snake10 65 is out.  let's assume we need device capability >7.5..
        mkdir -p ${root}/hug/vidlabel
        cd ${root}/hug/vidlabel
        if [ ! -d "AutoGPTQ" ]; then 
          git clone https://github.com/LLaVA-VL/LLaVA-NeXT
        fi
        cd LLaVa-NeXT
        pip install --upgrade pip
        pip install -e ".[train]"

      fi

      mkdir -p ${root}/hug/vidlabel
      cd ${root}/hug/vidlabel
      if [ ! -d "AutoGPTQ" ]; then 
        git clone git+https://github.com/PanQiWei/AutoGPTQ.git
      fi
      cd AutoGPTQ
      pip install -vvv --no-build-isolation -e .[triton]


      echo "Created env hug"
      echo ""
    fi
  } 2>&1 | tee env-hug.log
  echo "conda envs hug-cpu (and maybe hug) created!  See env-hug*.log"
  echo ""
fi

