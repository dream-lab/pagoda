import os
import numpy as np
import pandas as pd

data_dir = './roofline/ncu_work/results/csvs/ert_results'
file_names = os.listdir(data_dir)

file_paths = [os.path.join(data_dir, file_name) for file_name in file_names]
dataframes = {
    file_name.split('_')[0] + '_' + file_name.split('_')[3]: pd.read_csv(file_path)
    for file_name, file_path in zip(file_names, file_paths)
}

columns = ['sm__sass_thread_inst_executed_op_ffma_pred_on.sum', 'sm__sass_thread_inst_executed_op_fmul_pred_on.sum', 'sm__sass_thread_inst_executed_op_fadd_pred_on.sum',
           'sm__inst_executed_pipe_tensor.sum', 'dram__bytes.sum', 'lts__t_bytes.sum', 'l1tex__t_bytes.sum']

for name, dfmetric in dataframes.items():
    column_map = { col: [] for col in columns }
    count = 0
    for col_key in columns:
        for col_value in dfmetric.columns:
            if col_key in col_value:
                column_map[col_key].append(col_value)
                count += 1

    # https://gitlab.com/NERSC/roofline-on-nvidia-gpus/-/blob/master/custom-scripts/postprocess.py
    dfmetric['CC FLOPs'] = 2 * dfmetric[column_map['sm__sass_thread_inst_executed_op_ffma_pred_on.sum'][0]] \
                            + dfmetric[column_map['sm__sass_thread_inst_executed_op_fmul_pred_on.sum'][0]] \
                            + dfmetric[column_map['sm__sass_thread_inst_executed_op_fadd_pred_on.sum'][0]]
                            
    dfmetric['TC FLOPs']= 2048 * dfmetric[column_map['sm__inst_executed_pipe_tensor.sum'][0]]
    dfmetric['all FLOPs']= dfmetric['CC FLOPs'] + dfmetric['TC FLOPs']
    
    dfmetric['HBM'] = dfmetric[column_map['dram__bytes.sum'][0]]
    dfmetric['L2'] = dfmetric[column_map['lts__t_bytes.sum'][0]]
    dfmetric['L1'] = dfmetric[column_map['l1tex__t_bytes.sum'][0]]
    
    print('\n\nName:', name)
    print('FLOPs:', dfmetric['all FLOPs'].sum() / 1e9)
    print('Tensor FLOPs:', dfmetric['TC FLOPs'].sum())
    print('Memory Accesses:', dfmetric['HBM'].sum())
    print('Memory Accesses unit:', column_map['dram__bytes.sum'][0])
    
    

    


