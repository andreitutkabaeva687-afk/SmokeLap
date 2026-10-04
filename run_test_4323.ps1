$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot

python scripts\evaluate_global_confusion.py `
  --config configs\smokelap_m20.yaml `
  --ckpt checkpoints\smokelap_m20\smokelap_m20_best.pth `
  --split test `
  --output_dir outputs\global_confusion\smokelap_m20_best\test `
  --batch_size 1 `
  --num_workers 0 `
  --device cuda
