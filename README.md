Download model locally from HF

huggingface-cli download meta-llama/Llama-3.2-1B-Instruct \
  --include="*" \
  --local-dir ./Llama-3.2-1B-Instruct \
  --local-dir-use-symlinks False
