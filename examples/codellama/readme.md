
# CodeLlama

## Overview

The Code Llama models were proposed in [Code Llama: Open Foundation Models for Code](https://ai.meta.com/research/publications/code-llama-open-foundation-models-for-code/).
The abstract from the paper is the following:

*We release Code Llama, a family of large language models for code based on Llama 2 providing state-of-the-art performance among open models, infilling capabilities, support for large input context, and zero-shot instruction following ability for programming tasks. We provide multiple flavors to cover a wide range of applications: foundation models (Code Llama), Python specializations (Code Llama - Python), and instruction-following models (Code Llama - Instruct), with 7B, 13B, and 34B parameters each. All models are trained on sequences of 16k tokens and show improvements on inputs with up to 100k tokens. 7B and 13B Code Llama and Code Llama - Instruct variants support infilling based on surrounding content. Code Llama reaches state-of-the-art performance among open models on several code benchmarks, with scores of up to 53% and 55% on HumanEval and MBPP, respectively. Notably, Code Llama - Python 7B outperforms Llama 2 70B on HumanEval and MBPP, and all our models outperform every other publicly available model on MultiPL-E. We release Code Llama under a permissive license that allows for both research and commercial use.*

All Code Llama model checkpoints can be found [here](https://huggingface.co/models?search=code_llama), and the officially released checkpoints at [meta llama org](https://huggingface.co/meta-llama).

This model was contributed by [ArthurZucker](https://huggingface.co/ArthurZ). The original code can be found [here](https://github.com/facebookresearch/llama).

## Usage tips and examples

<Tip warning={true}>

The base `Llama2` family models used for Code Llama were trained using `bfloat16`, but the original inference uses `float16`. Let's look at the different precisions:

* `float32`: PyTorch convention on model initialization is to load models in `float32`, no matter the `dtype` the model weights were stored with. `transformers` follows this convention for consistency with PyTorch. This is selected by default. If you want the `AutoModel` API to cast the load checkpoint to a specific `dtype`, you must specify `torch_dtype="auto"`, e.g. `model = AutoModelForCausalLM.from_pretrained("path", torch_dtype = "auto")`.
* `bfloat16`: Code Llama was trained with this precision, so we recommend using it for further training or fine-tuning.
* `float16`: We recommend using this precision for inference as it's usually faster than `bfloat16` and no significant degradation in evaluation metrics is observed compared to `bfloat16`. You can also run inference with bfloat16. After fine-tuning, we recommend checking inference results in both float16 and bfloat16.

As mentioned above, the `dtype` of the storage weights is mostly irrelevant unless you use `torch_dtype="auto"` when initializing the model. The reason is that the model will first be downloaded (using the `dtype` of the checkpoint online), then cast to the default `dtype` of `torch` (which becomes `torch.float32`). If a specified `torch_dtype` is provided, that will be used instead.

</Tip>

Tips:
- Infilling tasks are supported out of the box. You should use `tokenizer.fill_token` where you want the input to be filled.
- The model conversion script is the same as for the `Llama2` family.

Here's an example usage:

```bash
python src/transformers/models/llama/convert_llama_weights_to_hf.py \
    --input_dir /path/to/downloaded/llama/weights --model_size 7B --output_dir /output/path
Note that running the script requires enough CPU RAM to host the whole model in float16 precision (even with the largest version). There are several checkpoints, each containing part of the model weights, so all of them need to be loaded in RAM.

After conversion, the model and tokenizer can be loaded as follows:

>>> from transformers import CodeLlamaTokenizer
>>> from mindway.transformers.models.llama import LlamaForCausalLM
>>> import mindspore as ms 

>>> tokenizer = CodeLlamaTokenizer.from_pretrained("meta-llama/CodeLlama-7b-hf")
>>> model = LlamaForCausalLM.from_pretrained("meta-llama/CodeLlama-7b-hf")
>>> PROMPT = '''def remove_non_ascii(s: str) -> str:
    """ <FILL_ME>
    return result
'''
>>> input_ids = ms.Tensor(tokenizer(prompt, return_tensors="np").input_ids, ms.int32)
>>> generated_ids = model.generate(input_ids, max_new_tokens=128,  do_sample=False).asnumpy()

>>> filling = tokenizer.batch_decode(generated_ids[:, input_ids.shape[1]:], skip_special_tokens = True)[0]
>>> print(PROMPT.replace("<FILL_ME>", filling))
def remove_non_ascii(s: str) -> str:
    """ Remove non-ASCII characters from a string.

    Args:
        s: The string to remove non-ASCII characters from.

    Returns:
        The string with non-ASCII characters removed.
    """
    result = ""
    for c in s:
        if ord(c) < 128:
            result += c
    return result

Internally, the tokenizer automatically splits by <FILL_ME> to create a formatted input string following the original training pattern. This is more robust than preparing the pattern yourself as it avoids very hard-to-debug pitfalls like token glueing. To check how much CPU and GPU memory this model (or others) needs, try this calculator which can help determine those values.

The LLaMA tokenizer is a BPE model based on sentencepiece. One quirk of sentencepiece is that when decoding a sequence, if the first token is the start of a word (e.g., "Banana"), the tokenizer does not prepend the prefix space to the string.

Code Llama has the same architecture as the Llama2 models. For API reference, see the Llama2 documentation page.
Find the reference for the Code Llama tokenizer below.