#!/bin/bash
# 搭 ModernBERT 用的虚拟环境：继承生产 torch，只换 transformers>=4.48
set -euo pipefail
cd /work/workspace/ccl25
python3 -m venv venv-mb --system-site-packages
venv-mb/bin/pip install -q transformers==4.48.3 safetensors
venv-mb/bin/python -c "import transformers, torch; print('VENV_OK', transformers.__version__, torch.__version__, torch.cuda.is_available())"
