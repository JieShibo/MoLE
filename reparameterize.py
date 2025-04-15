import argparse
import torch
from modeling_mole import MoleForCausalLM
from modeling_mole_rep import MoleForCausalLM as MoleForCausalLM_rep

parser = argparse.ArgumentParser()
parser.add_argument('--from_path', type=str)
parser.add_argument('--to_path', type=str)
args = parser.parse_args()

model = MoleForCausalLM.from_pretrained(args.from_path, device_map="cuda")
model_rep = MoleForCausalLM_rep.from_pretrained(args.from_path, device_map="cuda")

with torch.no_grad():
    output_all = []
    for layer in model.model.layers:
        output_layer = []
        token_embeds = layer.expert_layernorm(model.model.embed_tokens.weight)
        for expert in layer.experts:
            output = expert(token_embeds)
            output_layer.append(output)
        output_layer = torch.stack(output_layer, dim=1)
        output_all.append(output_layer.cpu())
    output_all = torch.stack(output_all, dim=1)
    output_all = output_all.reshape(output_all.size(0), -1)

state_dict = {'model.moe_table.weight': output_all}
model_rep.load_state_dict(state_dict, False)
model_rep = model_rep.half()
model_rep.save_pretrained(args.to_path)