# MoLE
Official code of ''Mixture of Lookup Experts''

<p align="left">
<a href="https://arxiv.org/abs/2503.15798" alt="arXiv">
    <img src="https://img.shields.io/badge/arXiv-2503.15798-b31b1b.svg?style=flat" /></a>
</p>


<p align="center">
<img src="https://arxiv.org/html/2503.15798v1/x2.png" width="700">
</p>

## Environment
+ torch                     2.0.1
+ transformers              4.38.2

## Models
#### Dense Models
+ modeling_dense.py
  
#### Moe Models
+ modeling_moe.py
  
#### MoLE Models
+ modeling_mole.py (training)
+ modeling_mole_rep.py (inference)
  
## HF Checkpoints 
To be uploaded

## Reparameterize MoLE for Inference
```bash
python reparameterize.py --from_path <training_model_path> --to_path <inference_model_path>
```

## Inference
```python3
from transformers import AutoTokenizer
from modeling_mole_rep import MoleForCausalLM
model = MoleForCausalLM.from_pretrained(model_path, device_map='cuda')
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
inputs = tokenizer("Hello, I am", return_tensors="pt").to(model.device)
tokens = model.generate(**inputs, max_length=10)
print(tokenizer.decode(tokens[0]))
```

Note that since the offloading of LUTs involves the support of the file system, the above demo still puts LUTs in the GPU memory. Alternatively, you can try the following demo, which offloads LUTs to CPU memory. This demo has not been specially optimized, so there may be some inefficiencies.
```python3
from transformers import AutoTokenizer
from modeling_mole_rep import MoleForCausalLM
model = MoleForCausalLM.from_pretrained(model_path, device_map='cpu')
model.model.embed_tokens.cuda()
model.model.layers.cuda()
model.model.norm.cuda()
model.lm_head.cuda()
model.model._buffers["causal_mask"] = model.model._buffers["causal_mask"].cuda()
model.model.moe_table.weight.data = model.model.moe_table.weight.data.pin_memory()
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
inputs = tokenizer("Hello, I am", return_tensors="pt").to('cuda')
tokens = model.generate(**inputs, max_length=10)
print(tokenizer.decode(tokens[0]))
```

## Citation

```
@article{jie2025mole,
  title={Mixture of Lookup Experts},
  author={Jie, Shibo and Tang, Yehui and Han, Kai and Li, Yitong and Tang, Duyu and Deng, Zhi-Hong and Wang, Yunhe},
  journal={arXiv preprint arXiv:2503.15798},
  year={2025}
}
```
