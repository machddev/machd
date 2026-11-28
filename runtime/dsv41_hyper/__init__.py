import os as _os
from pathlib import Path as _Path
_dv = _Path(__file__).resolve().parents[2] / "assets" / "dsv41_draft_vocab.txt"
if _dv.is_file():
    _os.environ.setdefault("DSV41_DRAFT_VOCAB_PATH", str(_dv))
from dsv41_hyper._dsv41_hyper import HyperEngine
from dsv41_hyper.distributed_engine import DistributedHyperEngine, GenerationResult
from dsv41_hyper.tokenizer import DeepSeekV41Tokenizer, parse_assistant_output
