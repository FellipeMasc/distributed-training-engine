#!/bin/bash
# Baixa tudo que o treino precisa ANTES de submeter ao Slurm. Rode no LOGIN NODE
# (que tem internet), a partir da raiz do repo em /scratch, com o env ativado:
#
#   source $SCRATCH/distributed-training-engine/tg/bin/activate
#   bash scripts/prepare_offline_assets.sh            # seq 512 (padrao do sbatch)
#   SEQ_LENGTH=1024 bash scripts/prepare_offline_assets.sh
#
# Os nos de computacao nao tem internet, entao o sbatch roda com HF_HUB_OFFLINE=1
# e le: hf_model_safetensors/ (checkpoint), $REPO/.cache/huggingface (tokenizer)
# e dataset/data/*.bin/.idx (dataset ja tokenizado).
set -euo pipefail

REPO_DIR=$(cd "$(dirname "$0")/.." && pwd)
cd "$REPO_DIR"
SEQ_LENGTH=${SEQ_LENGTH:-512}
MODEL_NAME=${MODEL_NAME:-NousResearch/Llama-3.2-1B}

# Mesmo HF_HOME que os scripts train_8gpu*.srm usam, para o cache bater.
export HF_HOME="$REPO_DIR/.cache/huggingface"
export TOKENIZERS_PARALLELISM=false
mkdir -p "$HF_HOME"

echo "== 1/4 checkpoint -> hf_model_safetensors/"
python - "$MODEL_NAME" <<'PY'
import os, sys, glob, huggingface_hub
name = sys.argv[1]; dst = "hf_model_safetensors"
if os.path.exists(f"{dst}/config.json") and glob.glob(f"{dst}/*.safetensors"):
    print(f"ja existe em {dst}/")
else:
    huggingface_hub.snapshot_download(name, repo_type="model", local_dir=dst,
        token=os.environ.get("HF_TOKEN"), allow_patterns=["*.safetensors", "*.json"])
    assert glob.glob(f"{dst}/*.safetensors"), "checkpoint sem .safetensors"
    print("ok")
PY

echo "== 2/4 tokenizer -> $HF_HOME (usado por tools/preprocessed_data.py pelo nome do hub)"
python - "$MODEL_NAME" <<'PY'
import sys
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(sys.argv[1])
print("ok, vocab", len(tok))
PY

echo "== 3/4 corpus bruto -> dataset/data/tinystories-portuguese.jsonl"
if [ -f dataset/data/tinystories-portuguese.jsonl ]; then
  echo "ja existe"
else
  (cd dataset && python jsonl_text_dataset.py)
  rm -f dataset/data/tinystories.jsonl   # so o filtro pt e usado; o arquivo completo e grande
fi

echo "== 4/4 dataset tokenizado para seq_length=$SEQ_LENGTH"
python - "$SEQ_LENGTH" <<'PY'
import sys
sys.path.insert(0, ".")
from dataset.indexed_dataset import IndexedDataset
from training.train import resolve_data_prefix, RAW_DATA_PATH, DATA_DIR, DATA_JSON_KEY
from tools.preprocessed_data import preprocess_jsonl
seq = int(sys.argv[1]); prefix = resolve_data_prefix(seq)
if IndexedDataset.exists(prefix):
    print(f"ja existe: {prefix}.{{bin,idx}}")
else:
    written = preprocess_jsonl(input_path=RAW_DATA_PATH,
        output_prefix=DATA_DIR / f"{RAW_DATA_PATH.stem}-{seq}", seq_length=seq,
        json_keys=(DATA_JSON_KEY,), append_eod=True, workers=8)
    print("ok:", written)
PY

echo "== verificacao offline (como o no de computacao vai enxergar)"
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python - "$MODEL_NAME" <<'PY'
import sys
from transformers import AutoConfig, AutoTokenizer
AutoConfig.from_pretrained("hf_model_safetensors", local_files_only=True)
AutoTokenizer.from_pretrained(sys.argv[1])
print("offline OK: config + tokenizer carregam sem rede")
PY
echo "pronto. Agora: sbatch train_8gpu_1node.srm"
