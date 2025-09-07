import numpy as np
import pandas as pd
import argparse

def _ncu_get_flops_double(kernel_data: dict, col: str) -> float:
    flops = (kernel_data['smsp__sass_thread_inst_executed_op_dadd_pred_on.sum.per_cycle_elapsed [inst/cycle]'] \
                + kernel_data['smsp__sass_thread_inst_executed_op_dmul_pred_on.sum.per_cycle_elapsed [inst/cycle]'] \
                + kernel_data['derived__smsp__sass_thread_inst_executed_op_dfma_pred_on_x2 [inst]']) \
            * kernel_data['smsp__cycles_elapsed.avg.per_second [Ghz]'] \
            * kernel_data[col] 

    return flops

def _ncu_get_flops_single(kernel_data: dict, col: str, breakdown=None) -> float:
    flops = (kernel_data['smsp__sass_thread_inst_executed_op_fadd_pred_on.sum.per_cycle_elapsed [inst/cycle]'] \
                + kernel_data['smsp__sass_thread_inst_executed_op_fmul_pred_on.sum.per_cycle_elapsed [inst/cycle]'] \
                + kernel_data['derived__smsp__sass_thread_inst_executed_op_ffma_pred_on_x2 [inst]']) \
            * kernel_data['smsp__cycles_elapsed.avg.per_second [Ghz]'] \
            * kernel_data[col]
    if breakdown:
        return flops, kernel_data['smsp__sass_thread_inst_executed_op_fadd_pred_on.sum.per_cycle_elapsed [inst/cycle]'], kernel_data['smsp__sass_thread_inst_executed_op_fmul_pred_on.sum.per_cycle_elapsed [inst/cycle]'], kernel_data['derived__smsp__sass_thread_inst_executed_op_ffma_pred_on_x2 [inst]']
    return flops


def _ncu_get_flops_half(kernel_data: dict, col: str) -> float:
    flops = (kernel_data['smsp__sass_thread_inst_executed_op_hadd_pred_on.sum.per_cycle_elapsed [inst/cycle]'] \
                + kernel_data['smsp__sass_thread_inst_executed_op_hmul_pred_on.sum.per_cycle_elapsed [inst/cycle]'] \
                + kernel_data['derived__smsp__sass_thread_inst_executed_op_hfma_pred_on_x2 [inst]']) \
            * kernel_data['smsp__cycles_elapsed.avg.per_second [Ghz]'] \
            * kernel_data[col]

    return flops

def _have_strings(name: str, *strings):
    return any(s in name for s in strings)

def _ncu_get_flops_tensor(kernel_data: dict, col: str, breakdown=None) -> float:
    kernel_name = kernel_data['Function Name']
    factor = 2048 

    # ampere (A100 etc) fp16
    if _have_strings(kernel_name, '16816', 'tensor16x8x16'):
        factor = 4096
    elif _have_strings(kernel_name, '1688', 'tensor16x8x8'):
        factor = 2048
    elif _have_strings(kernel_data['Demangled Name'], '1688', 'tensor16x8x8'):
        factor = 2048

    # ampere (A100 etc) int8
    elif _have_strings(kernel_name, 'i8i8_i8i32_f32') \
        and _have_strings(kernel_name, 'tensor16x8x32'):
        factor = 8192
    elif _have_strings(kernel_name, 'i8i8_i32_f32'):
        factor = 8192
    elif _have_strings(kernel_name, 'i8i8_i8i32_f32') \
        and _have_strings(kernel_name, 'tensor8x8x16'):
        factor = 2048
    elif _have_strings(kernel_name, 'imma') and _have_strings(kernel_name, 'ampere'):    # ampere_first_layer_filter3x3_imma_fwd_swish_execute_filter3x3_swish_kernel_trt
        factor = 2048

    # TODO: need to verify
    # volta (V100 etc), HMMA.884.F16.F16 fix
    elif (
            (_have_strings(kernel_name, 'h884') or
                (_have_strings(kernel_name, 'f16f16_f16f16_f16') and _have_strings(kernel_name, 'tensor8x8x4'))
            ) and not _have_strings(kernel_name, 's884')
        ):
        factor = 1024
        
    flops = kernel_data['smsp__inst_executed_pipe_tensor.sum.per_cycle_elapsed [inst/cycle]'] \
            * factor \
            * kernel_data['smsp__cycles_elapsed.avg.per_second [Ghz]'] \
            * kernel_data[col]
    
    if breakdown:
        return flops, kernel_data['smsp__inst_executed_pipe_tensor.sum.per_cycle_elapsed [inst/cycle]'], factor
    return flops

def ncu_get_flops(kernel_data: dict, data_width: int, col: str) -> float:
    """return all double/single/half/tensor FLOPs (count of FLoat OP)"""
    # double = _ncu_get_flops_double(kernel_data, col)
    # single, fadd, fmul, ffma = _ncu_get_flops_single(kernel_data, col, breakdown=True)
    # half = _ncu_get_flops_half(kernel_data, col)
    # tensor, tensor_sum, factor = _ncu_get_flops_tensor(kernel_data, col, breakdown=True)
    
    # if _have_strings(kernel_data['Function Name'], 'gemm'):
    #     print(f"{kernel_data['Function Name']}")
    #     print(f"double: {double}")
    #     print(f"single: {single}")
    #     print(fadd, fmul, ffma)
    #     print(f"half: {half}")
    #     print(f"tensor: {tensor}")
    #     print(tensor_sum, factor)
    
    """if tensor == 0:
        print(f"{kernel_data['Function Name']}")
        print(f"single: {single}")
        print(fadd, fmul, ffma)
        print("-"*50)

    if (single != 0) and (tensor != 0):
        print(f"{kernel_data['Function Name']}")
        print(f"single: {single}")
        print(fadd, fmul, ffma)
        print(f"tensor: {tensor}")
        print(tensor_sum, factor)
        print("*"*50)"""
        
    all_flops = (
        _ncu_get_flops_double(kernel_data, col),
        _ncu_get_flops_single(kernel_data, col),
        _ncu_get_flops_half(kernel_data, col),
        _ncu_get_flops_tensor(kernel_data, col)
    )
    flops = sum(all_flops)
    """if not flops:
        flops = _ncu_get_flops_fallback(kernel_data, data_width)    # not good!"""
    return flops, all_flops[3]


parser = argparse.ArgumentParser()
parser.add_argument('--file_path', required=True, help='Path of .csv file')
parser.add_argument('--epochs', type=int, default=20, help='Number of epochs for the DNN workload')
args = parser.parse_args()
file_path = args.file_path
epochs = args.epochs


df = pd.read_csv(file_path)
    
time_col = ''
for x in df.columns:
    if 'gpu__time_duration.sum' in x:
        time_col = x
df[time_col] = df[time_col].astype(float)
print(time_col)

# df['dram__bytes.sum'] = df['dram__bytes.sum'].astype(float)
for x in df.columns:
    if 'dram' in x:
        print(x)
        
filtered_df = df[["dram__bytes.sum.per_second [Gbyte/s]", time_col]]
filtered_df["mem"] = df["dram__bytes.sum.per_second [Gbyte/s]"] * df[time_col]

mem = np.sum(filtered_df['mem'].values)
print(f'Memory Accesses: {mem/1e3/epochs:.5f} MB')

all_flops, all_tensors = 0, 0
for i in range(len(df)):
    f, tensors = ncu_get_flops(df.iloc[i], 32, time_col)
    all_flops += f
    all_tensors += tensors
    
print(f'FLOPs: {all_flops/1e6/epochs:.5f} G')
print(f'Tensor FLOPs: {all_tensors/1e6/epochs:.2f} G')
