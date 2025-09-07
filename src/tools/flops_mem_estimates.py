import argparse
import numpy as np
import src.tools.inference_estimates as inference_estimates
import inference_estimates_lstm
import csv


parser = argparse.ArgumentParser()
parser.add_argument('--model_path', required=True)
parser.add_argument('--model_name', required=True)
parser.add_argument('--training', action='store_true')
parser.add_argument('--batch_size', default=32, required=False)
args = parser.parse_args()

model_name = args.model_name
model_path = args.model_path
training_mode = args.training
batch_size = args.batch_size

input_size = {
    'bert': (128,),
    'resnet': (3, 224, 224),
    'mobnet': (3, 224, 224),
    'yolo': (3, 640, 640),
    'lstm': (32,),
    'fcn': (256,),
    'pw_conv': (160, 7, 7),
    'bottleneck': (160, 56, 56)
}

if model_name == 'lstm':
    collector = inference_estimates_lstm.LayerInfoCollectorONNX(model_path, input_size[model_name], batch_size=batch_size)
    layer_infos = collector.run()
else:
    collector = inference_estimates.LayerInfoCollectorONNX(model_path, input_size[model_name], batch_size=batch_size)
    layer_infos = collector.run()
    
total_flops, total_mem, contri = 0, 0, {}

print("==========START===============")
for layer_name, info in layer_infos.items():
    print(f"Layer: {layer_name}")
    flops = 0
    mem_accesses = 0
    for key, value in info.items():
        print(f"  {key}: {value}")
        if key == 'FLOPs':
            total_flops += value
            flops = value
            if layer_name not in contri:
                contri[layer_name] = 0
            contri[layer_name] += value
        if key == 'Memory Accesses':
            total_mem += value
            mem_accesses = value

print("==========END===============")            

factor = 1
if training_mode:
    factor = 3

print(f"Total FLOPs: {total_flops * factor/1e9} GFLOPs")
print(f"Total Memory Accesses: {total_mem * factor/1e6} MB")
print(f"Arithematic Intensity: {total_flops/total_mem} FLOPs/Byte")

# for k, v in contri.items():
#     contri[k] = v 
# print('\n\n\nContribution:')
# pprint(contri)
