#!/bin/bash
# install_all_kernels.sh

echo "=== Installing faster_transformer (ft) ==="
cd ft && pip install --no-build-isolation -e . && cd ..

echo "=== Installing auto_gptq ==="
cd AutoGPTQ && pip install --no-build-isolation -e . && cd ..

echo "=== Installing hqq ==="
cd hqq && pip install --no-build-isolation -e . && cd ..

echo "=== All kernels installed! ==="