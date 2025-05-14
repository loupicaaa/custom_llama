This repo is a custom implementation of llama 3.2 1B (easily extendable to others llama 3 models)
The device map auto file allow to split the model across GPU layer per layer

Download model locally from HF

huggingface-cli download meta-llama/Llama-3.2-1B-Instruct \
  --include="*" \
  --local-dir ./Llama-3.2-1B-Instruct \
  --local-dir-use-symlinks False
