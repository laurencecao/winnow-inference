
python3 scripts/serve.py \
  --model ../Winnow-12B/gguf/Winnow-12B-Q8_0.gguf \
  --mmproj ../Winnow-12B/gguf/mmproj-Winnow-12B.gguf \
  --context 65536 --decision-parallel 4 --chat-parallel 1 \
  --n-gpu-layers -1 -ts 1,1,1,1 -fa on \
  --cache q8_0 --memory exclusive
