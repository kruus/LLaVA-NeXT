# vim: ts=4 sw=4 et
# curl -o ~/bin/cog -L https://github.com/replicate/cog/releases/latest/download/cog_`uname -s`_`uname -m`
# sudo chmod +x ~/bin/cog
# ~/bin/cog run python
# Building Docker image from environment in cog.yaml...
# ⚠ Stripping patch version from Torch version 3.11 to 3.11
# unknown flag: --cache-to
# See 'docker --help'.
root=/home/ml/kruus
llavadir=${root}/hug/vidlabel/LLaVA-NeXT
source ~/bin/conda.sh		# this defaults to ~/miniconda3, which has env esm-pyt2
# BETTER?
#eval "$(command conda 'shell.bash' 'hook' 2>/dev/null)"

export PYTHON_VERSION=310
export TORCH_VERSION=2.4.0
reuse_base_env=false            # if true, we'll reuse any existing BASE_ENV
remove_main_env=true            # if true, we'll remove and clone MAIN_ENV from BASE_ENV
PKGGREP='accelerate\|attn\|awq\|cuda\|datasets\|deepspeed\|einops\|formers\|gptq\|llava\|numba\|numpy\|nvidia\|optimize\|peft\|pydantic\|scikit\|scipy\|sglang\|torch\|triton\|urllib3\|vllm'

cd ${llavadir}
git pull
conda env list

if $reuse_base_env && conda env list | grep -q '^llavabase '; then
#if false; then
    echo -e "Conda env llavabase exists -- reusing it"
    # oops -- typo, here is a patch...
    conda activate llava
else
    echo ""
    echo "Re-create conda env llavabase"
    echo ""
    set -x
	# recreate a test environment (WIP)
    cd ${llavadir}
    set +x; conda deactivate; conda activate; set -x; echo "CONDA_PREFIX = ${CONDA_PREFIX}"
	conda env remove -y -n llavabase
	conda create -y -n llavabase python==3.10
    set +x; conda deactivate; conda activate llavabase; set -x; echo "CONDA_PREFIX = ${CONDA_PREFIX}"

	mamba install --override-channels -c conda-forge -y \
		git git-lfs jupyter ipython jupyterlab ipywidgets nb_conda_kernels \
        huggingface_hub \
        httpx==0.24.0 \
        cuda -c nvidia/label/cuda-12.1.105
    conda list | grep -i "${PKGGREP}"
    cd ${llavadir}
    # --> httpx 0.27.2-->0.24.0 (required later) may degrade jupyter install a little
    if false; then
        mamba install --override-channels -c conda-forge -c pytorch -c nvidia -y \
            "pytorch==${PYTORCH_VERSION}" pytorch-cuda==12.1 triton open-clip-torch \
            "safetensors>=0.4.1" "tqdm>=4.27" "deepspeed>=0.9.3" "pytest>=7.2.0,<8.0.0" \
            "pydantic" "seaborn" git git-lfs einops \
            jupyter ipython jupyterlab ipywidgets nb_conda_kernels \
            cuda-libraries-dev cuda-nvcc cuda-nvtx cuda-cupti
    fi
    if false; then
        mamba install --override-channels -c nvidia -c conda-forge -y \
            cuda-libraries-dev cuda-nvcc cuda-nvtx cuda-cupti
        # "all requested packages already installed
    fi

	pip install --upgrade pip build setuptools wheel

	pip install --no-build-isolation cffi 'accelerate>=0.29.1' \
		einops einops-exts gradio==3.35.2 gradio_client==0.2.9 \
        ninja nvidia-cuda-nvcc-cu12 nvidia-npp-cu12 nvidia-nvfatbin-cu12
    # torch -> 2.4.1    and replaces most nvidia 12.1 packages

    set +x
    conda list -n llavabase > llavabase.list
    echo "TORCH_VERSION ${TORCH_VERSION}"
    set -x
    # -------------------- LLaVA-NeXT -----------------------
    cd ${llavadir}
    pip install -e ".[train]"
    export TRANSFORMERS_VERSION=`cat llavabase.list | grep '^transformers ' | awk '{print $2;}'`
    echo "TRANSFORMERS_VERSION ${TRANSFORMERS_VERSION}"     # transformers-4.44.2
    echo "TORCH_VERSION ${TORCH_VERSION}"

    PIPOPT='--no-build-isolation --no-cache-dir'   # forces long rebuild
    PIPOPT=''
	FLASH_ATTENTION_FORCE_BUILD=TRUE pip install flash-attn || return   # --> 2.6.3

    # remove lmms_eval (depends on transformers==4.37.2 OUCH)
    #pip install lmms_eval==0.1.1

    if false; then
        # flashinfer had a setuptools versioning issue 
        #pip install flashinfer -i https://flashinfer.ai/whl/cu121/torch2.4
        cd ${root}/hug/vidlabel
        if [ ! -d flashinfer ]; then
            git clone https://github.com/flashinfer-ai/flashinfer.git --recursive
        fi
        cd flashinfer
        git pull
        cd python
        #export TORCH_CUDA_ARCH_LIST="6.2 7.2 7.5 8.0 8.6 9.0"
        export TORCH_CUDA_ARCH_LIST="7.5 8.0 8.6 9.0"  # flashinfer requires 75+
        pip install -e .
        cd ${llavadir}
    else
        pip install flashinfer -i https://flashinfer.ai/whl/cu121/torch2.4/
        # -> flashinfer-0.1.6+cu121torch2.4
    fi

    set +x
    conda list | grep -i "${PKGGREP}"
    # --> pytorch 2.4.1 cuda 12.1.105 triton=3.0.0 flash_attn-2.6.3 torchvision-0.19.1
fi

set +x; conda deactivate; conda activate llavabase; set -x; echo "CONDA_PREFIX = ${CONDA_PREFIX}"
conda list > llavabase.list
cat llavabase.list | grep -i "{PKGGREP}"

#if true; then
if conda env list | grep "^llavabase "; then
    echo ""
    echo "llavabase env ---> llava by adding AWQ, sglang"
    echo ""
    set +x; conda deactivate; conda activate; set -x; echo "CONDA_PREFIX = ${CONDA_PREFIX}"
    if $remove_main_env; then
        conda env remove -n llava
        rm -rf $CONDA_PREFIX/envs/llava || true
    fi
    if conda env list | grep "^llava "; then
        echo "NOT recloning llava from llavabase"
    else
        echo "Recreating llava as a clone of llavabase"
        rm -rf $CONDA_PREFIX/envs/llava || true
        conda create -y -n llava --clone llavabase
    fi
    set +x; conda activate llava; set -x; echo "CONDA_PREFIX = ${CONDA_PREFIX}"

    set -x
    # awq...
    if false; then
        export TORCH_CUDA_ARCH_LIST="7.5 8.0 8.6 9.0"
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
        cd ${llavadir}
    elif false; then
        pip install --override-channels -c conda-forge autoawq autoawq-kernels
    else
        TORCH_VERSION=${TORCH_VERSION} COMPUTE_CAPABILITIES="75,80,86,90" INSTALL_KERNELS=1 \
            pip install git+https://github.com/casper-hansen/AutoAWQ.git \
            "torch==${TORCH_VERSION}" "transformers==${TRANSFORMERS_VERSION}" \
            "tokenizers<0.20,>=0.19" "pydantic>=2.8"
    fi

    # i have transformers 4.44.2
    pip install "sglang[all]" \
            "torch==${TORCH_VERSION}"  # "transformers==${TRANSFORMERS_VERSION}"
    # oh: sglang[all] =>vllm==0.5.5 =>transformers>=4.43.2, but llava=>transformers==4.40.0.dev0
    # and             =>vllm==0.5.5 =>fllm-flash-attn==2.6.1
    # err        autoawq 0.2.6+cu121 requires autoawq-kernels, which is not installed.

    conda list > llava.list
    cat llava.list | grep -i "${PKGGREP}"

    #cd ${llavadir}
	#pip install -e ".[train]"   # HUGE requirements.txt !!
    # ...reinstalls sentencepiece, pydnatic numpy einops tokenizers scikit-learn transformers
    #               lmms_eval-->peft
    # vllm 0.5.5 require pydantic>=2.8 but have 1.10.8
	#pip install auto-gptq --no-build-isolation
	#pip install autoawq autoawq-kernels
	#pip install autoawq@https://github.com/casper-hansen/AutoAWQ.git 

	pip install git+https://github.com/EvolvingLMMs-Lab/lmms-eval.git
    # lmms-eval/prproject.toml holds back to protobuf==3.20   Can this be relaxed?
    # before, had protobuf 5.28.2
    # also datasets==2.15.1 is kinda' old.
    set +x
fi

cd ${llavadir}
conda activate llava
echo "CONDA_PREFIX ${CONDA_PREFIX}"
conda list > llava.list
cat llava.list | grep -i "${PKGGREP}"

# select ONE of next 2 lines
#pip install flash-attn
MODEL_ARGS="attn_implementation=None"

huggingface-cli login --token=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 --add-to-git-credential

echo -e "\n\n\n single-image.py ... MODEL_ARGS=${MODEL_ARGS}\n\n\n"
python single-image.py

echo -e "\n\n\n image-text.py ... MODEL_ARGS=${MODEL_ARGS}\n\n\n"
python image-text.py
# Many warnings vision_model.head.mlp.fce.bias copying from a non-meta parameter
# to a meta prameter in current model is a no-op.  Did you maean to pass
# assign=True to assign state dict items to the module?
# ERROR: bad operand type for unary + 'str' in line 45 of image-text.py

echo -e "\n\n\n video.py ... MODEL_ARGS=${MODEL_ARGS}\n\n\n"
python video.py

#echo -e "\n\n\n chatbot.py ...\n\n\n"
#python chatbot.py

export MODEL_ARGS="attn_implementation=None"
echo -e "\n\n\n run0.py (no attn) ... MODEL_ARGS=${MODEL_ARGS}\n\n\n"
python run0.py

export MODEL_ARGS=""
echo -e "\n\n\n run0.py (w/ attn) ... MODEL_ARGS=${MODEL_ARGS}\n\n\n"
python run0.py


if false; then
    ## ------ this ran the model and dataset, but ended in AttributeError before results. -----
    ## # image tasks
    ## # --tasks ai2d,chartqa,docvqa_val,infovqa_val,mme,realworldqa,mathvista_testmini,llava_in_the_wild,mmvet,mmbench_en_dev,ocrbench,mmmu,mathverse_testmini_vision_intensive,mathverse_testmini_vision_only,seedbench,scienceqa_img,mmstar \
    mkdir -p ./logs
    accelerate launch --num_processes=8 \
    -m lmms_eval \
    --model llava_onevision \
    --model_args pretrained=lmms-lab/llava-onevision-qwen2-0.5b-si,conv_template=qwen_1_5,model_name=llava_qwen \
    --tasks ai2d \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix llava_onevision \
    --output_path ./logs/
    # output:
    #   |Tasks|Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr|
    #   |-----|-------|----------------|-----:|-----------|---|-----:|---|-----:|
    #   |ai2d |Yaml   |flexible-extract|     0|exact_match|↑  |0.5421|±  | 0.009|

# EOF
