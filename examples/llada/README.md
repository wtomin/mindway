# Large Language Diffusion Models

LLaDA (Large Language Diffusion with mAsking) is a diffusion model trained entirely from scratch, rivaling LLaMA3 8B in performance. Please see Arxiv paper from this [URL](https://arxiv.org/abs/2502.09992).

## Inference

```
from transformers import AutoConfig, AutoTokenizer
from mindway.transformers.models.llada import LLaDAModelLM

tokenizer = AutoTokenizer.from_pretrained("GSAI-ML/LLaDA-8B-Instruct", trust_remote_code=True)
config = AutoConfig.from_pretrained("GSAI-ML/LLaDA-8B-Instruct", trust_remote_code=True)
model = LLaDAModelLM.from_pretrained("GSAI-ML/LLaDA-8B-Instruct", mindspore_dtype=ms.bfloat16, config=config)
```

We provide `generate()` functions in `generate.py`. You can try inference with `python generate.py`

You can directly run `python chat.py` to have multi-round conversations with LLaDA-8B-Instruct.

## Acknowledgements

We would like to thank the contributors to the [LLaDA](https://github.com/ML-GSAI/LLaDA/tree/main), [transformers](https://github.com/huggingface/transformers) and [HuggingFace](https://huggingface.co) repositories, for their open research and exploration.
