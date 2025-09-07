import onnx
import numpy as np
from collections import OrderedDict

sequence_length = 32
class LayerInfoCollectorONNX:
    def __init__(self, onnx_model_path, input_size, batch_size, data_type_size=4):
        self.model_path = onnx_model_path
        self.input_size = input_size  # Input size excluding batch dimension, e.g., (3, 224, 224)
        self.data_type_size = data_type_size  # Size of data type in bytes (e.g., 4 bytes for float32)
        self.layer_infos = OrderedDict()
        self.batch_size = batch_size

        # Load ONNX model
        self.onnx_model = onnx.load(onnx_model_path)
        self.graph = self.onnx_model.graph

        # Run shape inference on the model
        self.onnx_model = self._infer_shapes(self.onnx_model)
        self.graph = self.onnx_model.graph  # Update the graph after shape inference

        # Prepare a mapping from node output names to shapes
        self.value_infos = self._get_value_infos()
        self.initializers = {init.name: init for init in self.graph.initializer}

    def _infer_shapes(self, model):
        try:
            inferred_model = onnx.shape_inference.infer_shapes(model)
            print("Shape inference successful.")
            return inferred_model
        except Exception as e:
            print(f"Shape inference failed: {e}")
            return model  # Return the original model if shape inference fails

    def _get_value_infos(self):
        value_infos = {}
        for vi in self.graph.value_info:
            value_infos[vi.name] = vi
        for vi in self.graph.input:
            value_infos[vi.name] = vi
        for vi in self.graph.output:
            value_infos[vi.name] = vi
        return value_infos

    def _get_tensor_shape(self, tensor_name):
        if tensor_name in self.value_infos:
            tensor_type = self.value_infos[tensor_name].type.tensor_type
            shape = [d.dim_value if (d.dim_value > 0) else 1 for d in tensor_type.shape.dim]
            return shape
        else:
            # If shape is not available, return None
            return None

    def run(self):
        # Collect layer information
        for node in self.graph.node:
            layer_type = node.op_type
            layer_name = node.name if node.name else node.output[0]

            input_shapes = [self._get_tensor_shape(inp) for inp in node.input]
            output_shapes = [self._get_tensor_shape(outp) for outp in node.output]

            # Assume the first input and output for simplicity
            input_shape = input_shapes[0] if input_shapes else None
            output_shape = output_shapes[0] if output_shapes else None

            flops, mem_accesses, param_mem = self.compute_flops_mem_onnx(node, input_shape, output_shape)

            self.layer_infos[layer_name] = {
                'layer_type': layer_type,
                'input_shape': input_shape,
                'output_shape': output_shape,
                'FLOPs': flops,
                'Memory Accesses': mem_accesses,
                'Params': param_mem,
                'AI': flops / mem_accesses if mem_accesses > 0 else None
            }

        return self.layer_infos

    def compute_flops_mem_onnx(self, node, input_shape, output_shape):
        flops = 0
        mem_accesses = 0
        param_mem = 0
        dtype_size = self.data_type_size

        if input_shape is None or output_shape is None:
            #print(f"Shape information missing for node {node.name}. Skipping FLOPs computation.")
            return flops, mem_accesses, param_mem

        batch_size = self.batch_size

        if node.op_type == "Conv":
            # Get attributes
            attrs = {attr.name: self._get_attribute_value(attr) for attr in node.attribute}
            strides = attrs.get('strides', [1, 1])
            kernel_shape = attrs.get('kernel_shape')
            pads = attrs.get('pads', [0, 0, 0, 0])
            group = attrs.get('group', 1)

            # Get input/output channels
            W = self.initializers.get(node.input[1])
            if W is not None:
                weight_shape = W.dims  # Correctly get the shape from W.dims
                Cout, Cin_per_group, Kh, Kw = weight_shape
                Cin = Cin_per_group * group

                H_out, W_out = output_shape[2], output_shape[3]

                # FLOPs calculation
                output_elems = batch_size * Cout * H_out * W_out
                reduce_elems = Cin_per_group * Kh * Kw
                flops = (2 * output_elems * reduce_elems) + output_elems

                # Memory access calculation
                param_mem = Cout * Cin_per_group * Kh * Kw * dtype_size
                input_mem = batch_size * Cin * input_shape[2] * input_shape[3] * dtype_size
                output_mem = batch_size * Cout * H_out * W_out * dtype_size
                mem_accesses = param_mem + input_mem + output_mem
                print(f"Name: {node.name}")
                print(f"Input mem: {input_mem}")
                print(f"Output mem: {output_mem}")
                print(f"Param mem: {param_mem}")
                
            else:
                print(f"Weight initializer not found for Conv node {node.name}. Skipping FLOPs computation.")
                return flops, mem_accesses, param_mem

        elif node.op_type == "Gemm":
            # Handle fully connected layers
            # TODO: FIX LATER
            W = self.initializers.get(node.input[1])
            if W is not None:
                weight_shape = W.dims  # Correctly get the shape from W.dims
                N_out, N_in = weight_shape

                flops_per_instance = N_in * N_out * 2
                flops = batch_size * flops_per_instance

                # Memory access calculation
                param_mem = N_in * N_out * dtype_size
                input_mem = batch_size * N_in * dtype_size
                output_mem = batch_size * N_out * dtype_size
                mem_accesses = param_mem + input_mem + output_mem
                
                print(f"Name: {node.name}")
                print(f"Input mem: {input_mem}")
                print(f"Output mem: {output_mem}")
                print(f"Param mem: {param_mem}")
            else:
                print(f"Weight initializer not found for {node.op_type} node {node.name}. Skipping FLOPs computation.")
                return flops, mem_accesses, param_mem

        elif node.op_type in ["Relu", "Sigmoid", "Tanh", "HardSigmoid", "HardSwish", "Swish", "Softmax", "Erf"]:
            num_elements = np.prod(output_shape) * batch_size
            flops_per_element = self._get_activation_flops(node.op_type)
            flops = num_elements * flops_per_element

            # Memory access calculation
            input_mem = num_elements * dtype_size
            output_mem = num_elements * dtype_size
            mem_accesses = input_mem + output_mem
            
            print(f"Name: {node.name}")
            print(f"Input mem: {input_mem}")
            print(f"Output mem: {output_mem}")

        elif node.op_type == "BatchNormalization":
            num_elements = np.prod(output_shape) * batch_size
            flops = num_elements * 2  # Scale and shift

            # Memory access calculation
            param_mem = 2 * output_shape[1] * dtype_size  # Assuming scale and bias
            input_mem = num_elements * dtype_size
            output_mem = num_elements * dtype_size
            mem_accesses = param_mem + input_mem + output_mem
            
            print(f"Name: {node.name}")
            print(f"Input mem: {input_mem}")
            print(f"Output mem: {output_mem}")
            print(f"Param mem: {param_mem}")

        elif node.op_type == "MaxPool" or node.op_type == "AveragePool":
            # Get attributes
            attrs = {attr.name: self._get_attribute_value(attr) for attr in node.attribute}
            kernel_shape = attrs.get('kernel_shape', [1, 1])
            strides = attrs.get('strides', [1, 1])

            C = output_shape[1]
            H_out, W_out = output_shape[2], output_shape[3]
            num_elements = batch_size * C * H_out * W_out

            if node.op_type == "MaxPool":
                flops_per_element = np.prod(kernel_shape) - 1
            else:
                flops_per_element = np.prod(kernel_shape)  # For sum, division can be considered negligible

            flops = num_elements * flops_per_element

            # Memory access calculation
            input_mem = batch_size * input_shape[1] * input_shape[2] * input_shape[3] * dtype_size
            output_mem = num_elements * dtype_size
            mem_accesses = input_mem + output_mem
            
            print(f"Name: {node.name}")
            print(f"Input mem: {input_mem}")
            print(f"Output mem: {output_mem}")
            
        elif node.op_type == "MatMul":
            # Extract input shapes, focusing only on the first input
            input_shapes = [self._get_tensor_shape(inp) for inp in node.input]
            input_shapes[0][1] = sequence_length  # temp for lstm
            if len(input_shapes) < 1 or input_shapes[0] is None:
                print(f"Shape information missing for MatMul node {node.name}. Skipping FLOPs computation.")
                return flops, mem_accesses, param_mem

            input_a_shape = input_shapes[0]  # Use only the first input shape
            output_shape = self._get_tensor_shape(node.output[0])
            output_shape[0] = sequence_length  # temp for lstm

            # print('[DEBUG] Operator Type:', node.op_type) 
            # print('[DEBUG] Operator Name', node.name)
            # print('[DEBUG] input_shapes', input_shapes)
            # print('[DEBUG] output_shapes', output_shape)     
            
            # Handle potential 3D/4D inputs by reducing them to 2D if necessary
            if len(input_a_shape) > 2:
                input_a_shape = [np.prod(input_a_shape[:-1]), input_a_shape[-1]]

            if len(output_shape) > 2:
                output_shape = [np.prod(output_shape[:-1]), output_shape[-1]]

            if len(input_a_shape) != 2 or output_shape is None:
                print(f"Unexpected shapes for MatMul node {node.name}. Skipping FLOPs computation.")
                return flops, mem_accesses, param_mem

            print('[DEBUG] input_a_shape', input_a_shape)
            # FLOPs calculation
            output_elements = batch_size * output_shape[0] * output_shape[1]  # Output elements with batch size
            reduce_count = input_a_shape[-1]  # Number of reductions for each output element
            flops_per_compute_and_reduce = 2  # MAC operation requires 2 FLOPs (Multiply + Accumulate)
            flops = output_elements * reduce_count * flops_per_compute_and_reduce
            
            print('[DEBUG][MatMul]:', flops/1e9)

            # Memory access calculation
            param_mem = reduce_count * output_shape[1] * dtype_size  # Memory for weights
            input_mem = batch_size * np.prod(input_a_shape) * dtype_size
            output_mem = batch_size * np.prod(output_shape) * dtype_size
            mem_accesses = param_mem + input_mem + output_mem

            # print(f"Name: {node.name}")
            # print(f"Input A shape: {input_a_shape}")
            # print(f"Output shape: {output_shape}")
            # print(f"Output elements: {output_elements}")
            # print(f"Reduce count: {reduce_count}")
            # print(f"FLOPs per compute and reduce: {flops_per_compute_and_reduce}")
            # print(f"FLOPs for MatMul node: {flops}")
            # print(f"Memory Accesses for MatMul node: {mem_accesses}")
            print(f"Name: {node.name}")
            print(f"Input mem: {input_mem}")
            print(f"Output mem: {output_mem}")
            print(f"Param mem: {param_mem}")

        elif node.op_type in ["Add", "Sub", "Mul", "Div"]:
            num_elements = np.prod(output_shape) * batch_size
            flops = num_elements  # One operation per element

            # Memory access calculation
            input_mem = num_elements * dtype_size * 2  # Two inputs
            output_mem = num_elements * dtype_size
            mem_accesses = input_mem + output_mem
            
            print(f"Name: {node.name}")
            print(f"Input mem: {input_mem}")
            print(f"Output mem: {output_mem}")

        elif node.op_type == "LSTM":
            print("IN LSTM!!!! ***********************")
            
            # Extract input shapes
            input_shapes = [self._get_tensor_shape(inp) for inp in node.input]
            if len(input_shapes) < 1 or input_shapes[0] is None:
                print(f"Shape information missing for LSTM node {node.name}. Skipping FLOPs computation.")
                return flops, mem_accesses, param_mem

            # Assume first input is the sequence tensor
            input_shape = input_shapes[0]  # [batch_size, sequence_length, input_dim]
            if len(input_shape) != 3:
                print(f"Unexpected input shape for LSTM node {node.name}. Skipping FLOPs computation.")
                return flops, mem_accesses, param_mem
            
            # Extract LSTM attributes
            hidden_size = next((attr.i for attr in node.attribute if attr.name == "hidden_size"), 1)  # Default hidden size to 1
            num_gates = 4  # LSTM has 4 gates: input, forget, output, and candidate
            
            # Input dimensions
            batch_size = input_shape[0]
            # sequence_length = input_shape[1]  # Comes from input data, not a model property
            input_dim = input_shape[2]  # Embedding dimension

            print('[DEBUG]: input_shape:', input_shape)
            print('[DEBUG]: batch_size:', batch_size)
            print('[DEBUG]: sequence_length:', sequence_length)
            print('[DEBUG]: input_dim:', input_dim)
            print('[DEBUG]: hidden_size:', hidden_size)

            # ===== FLOPs Computation =====
            
            # (1) Input-to-Hidden Matrix Multiplication (MACs counted as 2 FLOPs each)
            input_to_hidden_flops = num_gates * input_dim * hidden_size * batch_size * sequence_length * 2 * 4

            # (2) Hidden-to-Hidden Matrix Multiplication (MACs counted as 2 FLOPs each)
            hidden_to_hidden_flops = num_gates * hidden_size * hidden_size * batch_size * sequence_length * 2 * 4

            # (3) Additions for Gates (each gate has an addition operation)
            gate_additions = num_gates * hidden_size * batch_size * sequence_length * 4

            # (4) Activation Functions: Sigmoid (3 gates) and Tanh (1 gate)
            activation_flops = (3 * hidden_size * batch_size * sequence_length * 4)  # Sigmoid: 4 FLOPs per unit
            activation_flops += (hidden_size * batch_size * sequence_length * 5)     # Tanh: 5 FLOPs per unit

            # (5) Element-wise Multiplications (for cell state updates and outputs)
            elementwise_flops = 3 * hidden_size * batch_size * sequence_length  # i_t * g_t, f_t * c_{t-1}, o_t * tanh(c_t)

            # (6) Extra element-wise addition for cell state update
            cell_state_additions = hidden_size * batch_size * sequence_length  # New cell state c_t = f_t * c_{t-1} + i_t * g_t

            # Total FLOPs
            flops = (input_to_hidden_flops +
                    hidden_to_hidden_flops +
                    gate_additions +
                    activation_flops +
                    elementwise_flops +
                    cell_state_additions)

            # Print component breakdown
            print('[I2H]:', input_to_hidden_flops)
            print('[H2H]:', hidden_to_hidden_flops)
            print('[GA]:', gate_additions)
            print('[Act]:', activation_flops)
            print('[Elem]:', elementwise_flops)
            print('[Cell State Add]:', cell_state_additions)

            # ===== Memory Access Computation =====
            
            dtype_size = 4  # Assume 4 bytes per float (float32)
            
            # Memory for weights
            # weight_memory = (input_dim * hidden_size + hidden_size * hidden_size) * num_gates * dtype_size
            weight_memory = (input_dim * hidden_size + hidden_size + sequence_length * (hidden_size * hidden_size + hidden_size)) * num_gates * dtype_size

            # Memory for inputs and outputs
            input_memory = batch_size * sequence_length * input_dim * dtype_size
            output_memory = batch_size * sequence_length * hidden_size * dtype_size

            # Memory for hidden and cell states
            hidden_cell_memory = 2 * batch_size * sequence_length * hidden_size * dtype_size  # Hidden + Cell states

            # Total memory accesses
            mem_accesses = weight_memory + input_memory + output_memory + hidden_cell_memory

            # print(f"Name: {node.name}")
            # print(f"Input shape: {input_shape}")
            # print(f"Hidden size: {hidden_size}")
            # print(f"Batch size: {batch_size}")
            # print(f"Sequence length: {sequence_length}")
            # print(f"Input dimension: {input_dim}")
            # print(f"FLOPs for LSTM node: {flops}")
            # print(f"Memory Accesses for LSTM node: {mem_accesses}")
            # print("-" * 50)
            
            print(f"Name: {node.name}")
            print(f"Input mem: {input_memory}")
            print(f"Output mem: {output_memory}")
            print(f"Param mem: {weight_memory}")



        elif node.op_type == "Transpose":
            print("IN TRANSPOSE!!!! ***********************")
            print(node.input)
            
            input_shape = self._get_tensor_shape(node.input[0])
            output_shape = self._get_tensor_shape(node.output[0])

            if input_shape is None or output_shape is None:
                print(f"Shape information missing for Transpose node {node.name}. Skipping FLOPs computation.")
                return flops, mem_accesses, param_mem

            # Transpose involves no FLOPs, but memory access equals input/output tensor size
            mem_accesses = batch_size * (np.prod(input_shape) + np.prod(output_shape)) * dtype_size

            # print(f"Name: {node.name}")
            # print(f"Input shape: {input_shape}")
            # print(f"Output shape: {output_shape}")
            # print(f"Memory Accesses for Transpose node: {mem_accesses}")
            
        elif node.op_type == "Concat":
            print("IN CONCAT!!!! ***********************")
            print(node.input)

            input_shapes = [self._get_tensor_shape(inp) for inp in node.input if self._get_tensor_shape(inp) is not None]
            output_shape = self._get_tensor_shape(node.output[0])

            if len(input_shapes) < 2 or output_shape is None:
                print(f"Shape information missing for Concat node {node.name}. Skipping FLOPs computation.")
                return flops, mem_accesses, param_mem

            # Concat involves copying all input tensors into a new tensor
            input_mem = batch_size * sum(np.prod(shape) * dtype_size for shape in input_shapes)
            output_mem = batch_size * np.prod(output_shape) * dtype_size
            mem_accesses = input_mem + output_mem

            # print(f"Name: {node.name}")
            # print(f"Input shapes: {input_shapes}")
            # print(f"Output shape: {output_shape}")
            # print(f"Memory Accesses for Concat node: {mem_accesses}")
            
            print(f"Name: {node.name}")
            print(f"Input mem: {input_mem}")
            print(f"Output mem: {output_mem}")

        elif node.op_type == "ConstantOfShape":
            print("IN CONSTANTOFSHAPE!!!! ***********************")
            print(node.input)

            output_shape = self._get_tensor_shape(node.output[0])
            if output_shape is None:
                print(f"Shape information missing for ConstantOfShape node {node.name}. Skipping FLOPs computation.")
                return flops, mem_accesses, param_mem

            # No FLOPs, but memory required to allocate tensor
            mem_accesses = batch_size * np.prod(output_shape) * dtype_size

            # print(f"Name: {node.name}")
            # print(f"Output shape: {output_shape}")
            # print(f"Memory Accesses for ConstantOfShape node: {mem_accesses}")

        elif node.op_type == "GlobalAveragePool":
            C = output_shape[1]
            H_in, W_in = input_shape[2], input_shape[3]
            num_elements = batch_size * C

            # FLOPs calculation: Sum over H_in * W_in elements
            flops_per_element = H_in * W_in
            flops = num_elements * flops_per_element

            # Memory access calculation
            input_mem = batch_size * input_shape[1] * H_in * W_in * dtype_size
            output_mem = num_elements * dtype_size
            mem_accesses = input_mem + output_mem

            print(f"Name: {node.name}")
            print(f"Input mem: {input_mem}")
            print(f"Output mem: {output_mem}")
            
        elif node.op_type == "Flatten":
            # Flatten operation has negligible FLOPs
            flops = 0
            input_mem = np.prod(input_shape) * dtype_size * batch_size
            output_mem = np.prod(output_shape) * dtype_size * batch_size
            mem_accesses = input_mem + output_mem
            
            print(f"Name: {node.name}")
            print(f"Input mem: {input_mem}")
            print(f"Output mem: {output_mem}")


        elif node.op_type == "Identity":
            # Identity operation has negligible FLOPs
            flops = 0
            input_mem = np.prod(input_shape) * dtype_size * batch_size
            output_mem = np.prod(output_shape) * dtype_size * batch_size
            mem_accesses = input_mem + output_mem
            
            print(f"Name: {node.name}")
            print(f"Input mem: {input_mem}")
            print(f"Output mem: {output_mem}")

        else:
            print(f"Warning: Node {node.op_type} not supported yet. Skipping FLOPs computation.")
            flops = 0
            mem_accesses = 0

        return flops, mem_accesses, param_mem

    def _get_attribute_value(self, attr):
        if attr.type == onnx.AttributeProto.INT:
            return attr.i
        elif attr.type == onnx.AttributeProto.INTS:
            return list(attr.ints)
        elif attr.type == onnx.AttributeProto.FLOAT:
            return attr.f
        elif attr.type == onnx.AttributeProto.FLOATS:
            return list(attr.floats)
        elif attr.type == onnx.AttributeProto.STRING:
            return attr.s.decode('utf-8')
        elif attr.type == onnx.AttributeProto.STRINGS:
            return [s.decode('utf-8') for s in attr.strings]
        else:
            return None

    def _get_activation_flops(self, op_type):
        # Approximate FLOPs per element for different activations
        if op_type == "Relu":
            return 1
        elif op_type == "Sigmoid":
            return 4  # Approximate
        elif op_type == "Tanh":
            return 5  # Approximate
        elif op_type == "HardSigmoid":
            return 4  # Approximate
        elif op_type == "HardSwish":
            return 5  # Approximate
        elif op_type == "Swish":
            return 5  # Approximate
        elif op_type == "Softmax":
            return 7
        elif op_type == "Erf":
            return 8
        else:
            return 1  # Default to 1 if unknown


class MemoryONNX_LSTM:
    def __init__(self, onnx_model_path, input_size, batch_size, data_type_size=4):
        self.model_path = onnx_model_path
        self.input_size = input_size  # Input size excluding batch dimension, e.g., (3, 224, 224)
        self.data_type_size = data_type_size  # Size of data type in bytes (e.g., 4 bytes for float32)
        self.layer_infos = OrderedDict()
        self.batch_size = batch_size

        # Load ONNX model
        self.onnx_model = onnx.load(onnx_model_path)
        self.graph = self.onnx_model.graph

        # Run shape inference on the model
        self.onnx_model = self._infer_shapes(self.onnx_model)
        self.graph = self.onnx_model.graph  # Update the graph after shape inference

        # Prepare a mapping from node output names to shapes
        self.value_infos = self._get_value_infos()
        self.initializers = {init.name: init for init in self.graph.initializer}

    def _infer_shapes(self, model):
        try:
            inferred_model = onnx.shape_inference.infer_shapes(model)
            print("Shape inference successful.")
            return inferred_model
        except Exception as e:
            print(f"Shape inference failed: {e}")
            return model  # Return the original model if shape inference fails

    def _get_value_infos(self):
        value_infos = {}
        for vi in self.graph.value_info:
            value_infos[vi.name] = vi
        for vi in self.graph.input:
            value_infos[vi.name] = vi
        for vi in self.graph.output:
            value_infos[vi.name] = vi
        return value_infos

    def _get_tensor_shape(self, tensor_name):
        if tensor_name in self.value_infos:
            tensor_type = self.value_infos[tensor_name].type.tensor_type
            shape = [d.dim_value if (d.dim_value > 0) else 1 for d in tensor_type.shape.dim]
            return shape
        else:
            # If shape is not available, return None
            return None

    def calculate_total_memory_accesses(self):
        total_params = 0
        total_io_memory = 0
        total_activation_memory = 0

        for node in self.graph.node:
            input_shapes = [self._get_tensor_shape(inp) for inp in node.input]
            output_shapes = [self._get_tensor_shape(outp) for outp in node.output]

            param_mem = 0
            input_mem = 0
            output_mem = 0

            if node.op_type in ["Conv", "Gemm", "MatMul"]:
                W = self.initializers.get(node.input[1])
                if W is not None:
                    param_mem = np.prod(W.dims) * self.data_type_size
                    total_params += np.prod(W.dims)

            for shape in input_shapes:
                if shape:
                    input_mem += np.prod(shape) * self.data_type_size * self.batch_size

            for shape in output_shapes:
                if shape:
                    output_mem += np.prod(shape) * self.data_type_size * self.batch_size

            total_io_memory += input_mem
            total_activation_memory += output_mem

        return {
            "Total Params": total_params,
            "Total I/O Memory": total_io_memory,
            "Total Activation Memory": total_activation_memory
        }

