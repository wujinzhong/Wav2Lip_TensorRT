
'''
> pip install nvidia-pyindex
> pip install polygraphy
> pip install nvtx
> pip install onnx_graphsurgeon
#copy libGL*, libX*, libx* from host /usr/lib/x86_64-linux-gnu/ to docker /usr/lib/x86_64-linux-gnu/
> pip install opencv-python opencv-contrib-python
> apt-get update -y && apt-get install -y libbsd-dev
> pip install moviepy
> pip install onnxruntime onnxruntime-gpu
'''

'''
#template for profiling:
from InferenceUtil import NVTXUtil, check_onnx, TRT_Engine, SynchronizeUtil, TorchUtil, Memory_Manager
mm = Memory_Manager()
mm.add_foot_print("prev-E2E")
torchutil = TorchUtil(gpu=0, memory_manager=mm)

with NVTXUtil("stage0", "red", mm), SynchronizeUtil(torchutil.torch_stream):
    xxx

mm.add_foot_print("stage0")
with NVTXUtil("stagen", "red", mm), SynchronizeUtil(torchutil.torch_stream):
    xxx
'''

import argparse
import os
import ctypes
from typing import Optional, List
import tensorrt as trt
from cuda import cuda, cudart

from polygraphy.backend.common import bytes_from_path
from polygraphy.backend.trt import CreateConfig, Profile
from polygraphy.backend.trt import engine_from_bytes, engine_from_network, network_from_onnx_path, save_engine
from polygraphy.backend.trt import util as trt_util

from polygraphy import cuda as pg_cuda
import gc
import numpy as np
import warnings
import torch
import nvtx
import onnxruntime
import onnx_graphsurgeon as gs
import onnx
import random

import tempfile
from distutils.util import strtobool
from typing import List

import numpy as np
import PIL.Image
import PIL.ImageOps
import cv2
import moviepy
import moviepy.video.io.ImageSequenceClip
import subprocess
from torch import nn

MY_VERBOSE = False

# Map of numpy dtype -> torch dtype
numpy_to_torch_dtype_dict = {
    np.uint8      : torch.uint8,
    np.int8       : torch.int8,
    np.int16      : torch.int16,
    np.int32      : torch.int32,
    np.int64      : torch.int64,
    np.float16    : torch.float16,
    np.float32    : torch.float32,
    np.float64    : torch.float64,
    np.complex64  : torch.complex64,
    np.complex128 : torch.complex128
}

if np.version.full_version >= "1.24.0":
    numpy_to_torch_dtype_dict[np.bool_] = torch.bool
else:
    numpy_to_torch_dtype_dict[np.bool] = torch.bool

# Map of torch dtype -> numpy dtype
torch_to_numpy_dtype_dict = {value : key for (key, value) in numpy_to_torch_dtype_dict.items()}

def collect_garbage():
	torch.cuda.empty_cache()
	gc.collect()
        
def get_available_devices():
    """Get IDs of all available GPUs.

    Returns:
        device (torch.device): Main device (GPU 0 or CPU).
        gpu_ids (list): List of IDs of all GPUs that are available.
    """
    gpu_ids = []
    if torch.cuda.is_available():
        gpu_ids += [gpu_id for gpu_id in range(torch.cuda.device_count())]
        device = torch.device(f'cuda:{gpu_ids[0]}')
        #torch.cuda.set_device(device)
    else:
        device = torch.device('cpu')

    return gpu_ids 
    #return device, gpu_ids 

#copy from https://blog.csdn.net/weicao1990/article/details/115307821, Pytorch DataParallel多卡训练模型导出onnx模型
#not verified!!!!!!!!!!!!!!!!
def load_model_from_pth( model_path="model.pth"):
    assert False, "not verified!!!!!!!!!!!!!!!!"
    import torch
    import torch.nn as nn
    from  collections import OrderedDict
    import torch.onnx
    import onnx
    import os    

    state_dict = torch.load(model_path)
    new_state_dict = OrderedDict()
    for k,v in  state_dict.items():
        if name.startwith("module."):
            name = k[7:]  #remove "module."
        else:
            name = k
        new_state_dict[name] = v

    model = None
    #from QualityNet import QualityNet
    #model = QualityNet()
    #model= model.to(device)
    #model.load_state_dict(new_state_dict)
    #model.eval()

    return model

'''
Memory_Manager usage:
memory_manager=Memory_Manager()
memory_manager.add_foot_print("phase1")
memory_manager.add_foot_print("phase2")
memory_manager.add_foot_print("phase3")
memory_manager.summary()
'''
class Memory_Manager:
    def __init__(self):
        self.devices = get_available_devices()
        self.foot_prints = [[] for _ in range(len(self.devices))]
    def add_foot_print(self, name=""):
        cur_device = torch.cuda.current_device()
        for device in self.devices:
            torch.cuda.set_device(device)
            _, free_mem, total_mem = cudart.cudaMemGetInfo()
            self.foot_prints[device].append( (name, total_mem-free_mem) )
        torch.cuda.set_device(cur_device)
    
    def summary(self):
        for device in self.devices:
            print('| {:^30} | {:^37} |'.format('', ''))
            print('| {:^30} | {:^37} |'.format(f'stage(gpu{device})', f'device mem usage(gpu{device})'))
            print('| {:^30} | {:^37} |'.format('', ''))
            last_mem = 0
            for name, used_mem in self.foot_prints[device]:
                print('| {:^30} | {:>19.2f} GB device memory. |'.format(name, (used_mem-last_mem)/(1024*1024*1024)))
                last_mem = used_mem
            print('| {:^30} | {:^37} |'.format('', ''))

def print_bindings( trt_engine:trt.ICudaEngine = None ):
    if trt_engine is not None:
        for i in range(trt_engine.num_bindings):
            tensor_name = trt_engine.get_tensor_name(i)
            tensor_mode = trt_engine.get_tensor_mode(tensor_name)
            tensor_shape = trt_engine.get_tensor_shape(tensor_name)
            tensor_dtype = trt_engine.get_binding_dtype(tensor_name)
            if MY_VERBOSE: print(f"binding {i}, {tensor_name}, {tensor_mode}, {tensor_shape}, {tensor_dtype}")

def check_cuda_err(err):
    if isinstance(err, cuda.CUresult):
        if err != cuda.CUresult.CUDA_SUCCESS:
            raise RuntimeError("Cuda Error: {}".format(err))
    if isinstance(err, cudart.cudaError_t):
        if err != cudart.cudaError_t.cudaSuccess:
            raise RuntimeError("Cuda Runtime Error: {}".format(err))
    else:
        raise RuntimeError("Unknown error type: {}".format(err))
    
def cuda_call(call):
    err, res = call[0], call[1:]
    check_cuda_err(err)
    if len(res) == 1:
        res = res[0]
    return res

class HostDeviceMem:
    """Pair of host and device memory, where the host memory is wrapped in a numpy array"""
    def __init__(self, size: int, dtype: np.dtype, name: str="", idx_in_trt:int=-1):
        nbytes = size * dtype.itemsize
        host_mem = cuda_call(cudart.cudaMallocHost(nbytes))
        
        if dtype == np.float16:
            pointer_type = ctypes.POINTER(np.ctypeslib.as_ctypes_type(np.int16))
            if MY_VERBOSE: print(f"dtype: {dtype}, pointer_type: {pointer_type}")

            self._host = np.ctypeslib.as_array(ctypes.cast(host_mem, pointer_type), (size,)).astype(np.float16)
            
        else:
            pointer_type = ctypes.POINTER(np.ctypeslib.as_ctypes_type(dtype))
            print(f"dtype: {dtype}, pointer_type: {pointer_type}")

            self._host = np.ctypeslib.as_array(ctypes.cast(host_mem, pointer_type), (size,))

        self._device = cuda_call(cudart.cudaMalloc(nbytes))
        self._nbytes = nbytes
        self._name = name
        self._idx_in_trt = idx_in_trt

    @property
    def name(self) -> str:
        return self._name
    
    @property
    def host(self) -> np.ndarray:
        return self._host

    @host.setter
    def host(self, arr: np.ndarray):
        if arr.size > self.host.size:
            raise ValueError(
                f"Tried to fit an array of size {arr.size} into host memory of size {self.host.size}"
            )
        np.copyto(self.host[:arr.size], arr.flat, casting='safe')

    @property
    def device(self) -> int:
        return self._device
    
    def device_from_D2D(self, src_device_ptr: int, src_nbytes: int, stream: cudart.cudaStream_t = None):
        if src_nbytes > self._nbytes:
            raise ValueError(
                f"Tried to fit an device memory of size {src_nbytes} into device memory of size {self._nbytes}"
            )
        memcpy_device_to_device(self._device, src_device_ptr, src_nbytes, stream)

    @property
    def nbytes(self) -> int:
        return self._nbytes

    def __str__(self):
        return f"Host:\n{self.host}\nDevice:\n{self.device}\nSize:\n{self.nbytes}\nName:\n{self.name}\nIdx:\n{self._idx_in_trt}\n"

    def __repr__(self):
        return self.__str__()

    def free(self):
        cuda_call(cudart.cudaFree(self.device))
        cuda_call(cudart.cudaFreeHost(self.host.ctypes.data))

# Frees the resources allocated in allocate_buffers
def free_buffers(inputs: List[HostDeviceMem], outputs: List[HostDeviceMem], stream: cudart.cudaStream_t = None):
    for mem in inputs + outputs:
        mem.free()
    
    if stream is not None:
        cuda_call(cudart.cudaStreamDestroy(stream))

# Frees the resources allocated in allocate_buffers
def free_stream(stream: cudart.cudaStream_t = None):
    cuda_call(cudart.cudaStreamDestroy(stream))

# Wrapper for cudaMemcpy which infers copy size and does error checking
def memcpy_host_to_device(device_ptr: int, host_arr: np.ndarray):
    nbytes = host_arr.size * host_arr.itemsize
    cuda_call(cudart.cudaMemcpy(device_ptr, host_arr, nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice))


# Wrapper for cudaMemcpy which infers copy size and does error checking
def memcpy_device_to_host(host_arr: np.ndarray, device_ptr: int):
    nbytes = host_arr.size * host_arr.itemsize
    cuda_call(cudart.cudaMemcpy(host_arr, device_ptr, nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost))

# Wrapper for cudaMemcpy which infers copy size and does error checking
def memcpy_device_to_device(dst_device_ptr: int, src_device_ptr: int, nbytes: int, stream: cudart.cudaStream_t = None):
    if stream is None:
        cuda_call(cudart.cudaMemcpy(dst_device_ptr, src_device_ptr, nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToDevice))
    else:
        cuda_call(cudart.cudaMemcpyAsync(dst_device_ptr, src_device_ptr, nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToDevice, stream))

def _do_inference_base(trt_inputs_buf, outputs, stream, execute_async, no_data_transfer_list, 
                       is_sync=True, enable_D2H=True, 
                       inputs_ptr=None):
    # Transfer input data to the GPU.
    if inputs_ptr is not None: # copy date from device pointers
        kind0 = cudart.cudaMemcpyKind.cudaMemcpyDeviceToDevice
        [cuda_call(cudart.cudaMemcpyAsync(inp.device, inp_ptr.ptr, inp.nbytes, kind0, stream)) for (inp, inp_ptr) in zip(trt_inputs_buf, inputs_ptr)]
    else:               
        kind0 = cudart.cudaMemcpyKind.cudaMemcpyHostToDevice
        [cuda_call(cudart.cudaMemcpyAsync(inp.device, inp.host, inp.nbytes, kind0, stream)) for inp in trt_inputs_buf if inp.name not in no_data_transfer_list]

    # Run inference.
    execute_async()
    # Transfer predictions back from the GPU.
    if enable_D2H:
        kind1 = cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost
        [cuda_call(cudart.cudaMemcpyAsync(out.host, out.device, out.nbytes, kind1, stream)) for out in outputs if out.name not in no_data_transfer_list]
    
        # Synchronize the stream
        if is_sync:
            cuda_call(cudart.cudaStreamSynchronize(stream))

        # Return only the host outputs.
        ret = [out.host for out in outputs]
        if MY_VERBOSE: print(f"_do_inference_base ret: {ret}")
    else:
        ret = [out.device for out in outputs]
    
    return ret

def create_stream():
    return cuda_call(cudart.cudaStreamCreate())

def synchronize_stream(stream):
    cuda_call(cudart.cudaStreamSynchronize(stream))

# This function is generalized for multiple inputs/outputs.
# inputs and outputs are expected to be lists of HostDeviceMem objects.
def do_inference(context, bindings, inputs, outputs, stream, no_data_transfer_list=[], is_sync=True, batch_size=1):
    def execute_async():
        context.execute_async(batch_size=batch_size, bindings=bindings, stream_handle=stream)
    return _do_inference_base(inputs, outputs, stream, execute_async, no_data_transfer_list, is_sync)

# This function is generalized for multiple inputs/outputs for full dimension networks.
# inputs and outputs are expected to be lists of HostDeviceMem objects.
def do_inference_v2(context, bindings, inputs, outputs, stream, no_data_transfer_list=[], is_sync=True):
    def execute_async():
        context.execute_async_v2(bindings=bindings, stream_handle=stream)
    return _do_inference_base(inputs, outputs, stream, execute_async, no_data_transfer_list, is_sync)

def save_tensor_to_numpy_dat( torch_cuda_tensor, dst_dat_path="./xxx.dat" ):
    torch_cuda_tensor.cpu().numpy().tofile(dst_dat_path)

def save_tensors_to_numpy_dat( torch_cuda_tensor, dst_dat_path="./xxx.dat" ):
    if type(torch_cuda_tensor) == list:
        for i, t in enumerate(torch_cuda_tensor):
            path = dst_dat_path.split(".dat")[0] + f"_{i}.dat"
            print(f"save tensor to path: {path}")
            save_tensor_to_numpy_dat( torch_cuda_tensor[i], dst_dat_path=path )
    else:
        save_tensor_to_numpy_dat( torch_cuda_tensor, dst_dat_path=dst_dat_path )

def save_tensor_as_json( torch_cuda_tensor, json_path="xxx.json" ):
    import json
    json_object = json.dumps(torch_cuda_tensor.cpu().numpy().tolist())
    # Writing to json file
    with open(json_path, "w") as outfile:
        outfile.write(json_object)

def save_tensors_as_json( torch_cuda_tensors, json_path="xxx.json" ):
    if type(torch_cuda_tensors) == list:
        for i, t in enumerate(torch_cuda_tensors):
            path = json_path.split(".json")[0] + f"_{i}.json"
            print(f"save tensor to path: {path}")
            save_tensor_as_json( torch_cuda_tensors[i], json_path=path )
    else:
        save_tensor_as_json( torch_cuda_tensors, json_path=json_path )

def compare_2_tensors( torch_cuda_tensor0, torch_cuda_tensor1 ):
    if type(torch_cuda_tensor0) == list:
        assert len(torch_cuda_tensor0)==len(torch_cuda_tensor1)
        for t0, t1 in zip(torch_cuda_tensor0, torch_cuda_tensor1):
            np0 = t0.cpu().numpy()
            np1 = t1.cpu().numpy()

            np10 = np.abs(np1-np0).reshape((-1))

            np0 = np.abs(np0).reshape((-1))
            np1 = np.abs(np1).reshape((-1))
            np0mm = [np0.max(), np0.min()]
            np1mm = [np1.max(), np1.min()]
            np10mm = [np10.max(), np10.min()]
            print(f"diff {np10mm}, tensor0 {np0mm}, tensor1 {np1mm} ")
    else:
        np0 = torch_cuda_tensor0.cpu().numpy()
        np1 = torch_cuda_tensor1.cpu().numpy()

        np10 = np.abs(np1-np0).reshape((-1))

        np0 = np.abs(np0).reshape((-1))
        np1 = np.abs(np1).reshape((-1))
        np0mm = [np0.max(), np0.min()]
        np1mm = [np1.max(), np1.min()]
        np10mm = [np10.max(), np10.min()]
        print(f"diff {np10mm}, tensor0 {np0mm}, tensor1 {np1mm} ")

def print_onnx(graph):
    #graph = gs.import_onnx(onnx.load(onnx_path))

    print("\n\n----------------------inputs------------------")
    inputs = graph.inputs
    for input in inputs:
        print(f"input: \n{input}")

    print("\n\n----------------------outputs------------------")
    outputs = graph.outputs
    for output in outputs:
        print(f"output: \n{output}")

    print("\n\n----------------------nodes------------------")
    for node in graph.nodes:
        print(f"node: \n{node}")
        for attr in node.attrs.values():
            print(f"attr: {attr}")
    
    print("\n\n----------------------tensors------------------")
    tensors = graph.tensors()
    for tensor in tensors:
        print(f"tensor: \n{tensor}")

    #graph.cleanup().toposort()
    ## Export the onnx graph from graphsurgeon
    #onnx.save_model(gs.export_onnx(graph), model_file)
    pass

def check_onnx(onnx_path:str):
    import onnx
    #model = onnx.load(onnx_path, load_external_data=False)
    #onnx.checker.check_model(model)
    
    shape_inference_onnx_path = onnx_path.split('.onnx')[0]+'_shape_inference.onnx'
    if MY_VERBOSE: print(f"shape_inference_onnx_path: {shape_inference_onnx_path}")

    if not os.path.exists(shape_inference_onnx_path):
        onnx_model = onnx.load(onnx_path)
        shaped_onnx_model = onnx.shape_inference.infer_shapes( onnx_model )
        onnx.save(shaped_onnx_model, shape_inference_onnx_path)
        #onnx.shape_inference.infer_shapes_path(onnx_path, shape_inference_onnx_path)
    else:
        shaped_onnx_model = onnx.load(shape_inference_onnx_path)

    #print(f"{shape_inference_onnx_path}: {shaped_onnx_model}")
    
    #print_onnx(shaped_onnx_model.graph)
    # model is an onnx model
    graph = shaped_onnx_model.graph
    # graph inputs
    for input_name in graph.input:
        print(input_name)
    # graph outputs
    for output_name in graph.output:
        print(output_name)
    # iterate over nodes
    for node in graph.node:
        # node inputs
        for idx, node_input_name in enumerate(node.input):
            print(idx, node_input_name)
        # node outputs
        for idx, node_output_name in enumerate(node.output):
            print(idx, node_output_name)
    
    print(f"value_info: {graph.value_info}")
    
    onnx.checker.check_model(onnx_path)

#torch.onnx.export template
def torch_onnx_export_template(onnx_model, fp16=False, onnx_model_path="model.onnx", maxBatch=1 ):
    if not os.path.exists(onnx_model_path):
        dynamic_axes = {
            "latent_model_input":   {0: "bs_x_2"},
            "prompt_embeds":        {0: "bs_x_2"},
            "noise_pred":           {0: "batch_size"}
        }

        device = torch.device("cuda:0")
        
        onnx_model2= onnx_model #onnx_model2= UNet_x(onnx_model)
        if isinstance(onnx_model2, torch.nn.DataParallel):
            onnx_model2 = onnx_model2.module

        onnx_model2.eval()
        onnx_model2 = onnx_model2.to(device=device)
        
        if fp16: dst_dtype = torch.float16
        else: dst_dtype = torch.float32

        dummy_inputs = {
            "latent_model_input": torch.randn((2*maxBatch, 4, 64, 32, 32), dtype=dst_dtype).to(device).contiguous(),
            "timestep": torch.tensor([1], dtype=torch.int64).to(device).contiguous(),
            "prompt_embeds": torch.randn((2*maxBatch, 77, 1024), dtype=dst_dtype).to(device).contiguous(),
            'prompt_embeds': torch.randint(1, 10, (1, 1, 1024, 2048), dtype=torch.int32).to(device).contiguous(),
            #"negative_prompt_embeds": None,
            #"orig_im_size": torch.tensor([1500, 2250], dtype=torch.float).contiguous(),
        }
        # output_names = ["masks", "iou_predictions", "low_res_masks"]
        output_names = ["noise_pred"]

        #import apex
        with torch.no_grad():
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
                warnings.filterwarnings("ignore", category=UserWarning)
                if True:
                    torch.onnx.export(
                        onnx_model,
                        tuple(dummy_inputs.values()),
                        onnx_model_path, #f,
                        export_params=True,
                        verbose=True,
                        opset_version=18,
                        do_constant_folding=False,
                        input_names=list(dummy_inputs.keys()),
                        output_names=output_names,
                        #dynamic_axes=dynamic_axes,
                    )  
                else:
                    with open(onnx_model_path, "wb") as f:
                        torch.onnx.export(
                        onnx_model,
                        tuple(dummy_inputs.values()),
                        f,
                        export_params=True,
                        verbose=True,
                        opset_version=18,
                        do_constant_folding=False,
                        input_names=list(dummy_inputs.keys()),
                        output_names=output_names,
                        #dynamic_axes=dynamic_axes,
                        )  
    check_onnx(onnx_model_path)
    return

def build_trt_engine_onnx_trtexec(  engine_path, #"unet.engine", 
                                    onnx_model_path, #="unet.onnx", 
                                    device=0, 
                                    fp16=False,):
    assert engine_path is not None
    assert onnx_model_path is not None
    if os.path.exists(engine_path):
        return 
    assert os.path.exists(onnx_model_path)
    
    os.system("CUDA_VISIBLE_DEVICES={} trtexec --onnx={} {} --saveEngine={} \
              --workspace=4096 \
              --exportOutput=output_{}.json". 
              format(device, onnx_model_path, 
                     "--fp16" if fp16 else "",
                     engine_path,
                     "fp16" if fp16 else "fp32"))

def build_trt_engine_onnx(engine_path="unet.engine", onnx_model_path="unet.onnx", fp16=True):
    if os.path.exists(engine_path):
        return
    
    import tensorrt as trt
    from pathlib import Path
    
    logger = trt.Logger(trt.Logger.VERBOSE)
    builder = trt.Builder(logger)
    config = builder.create_builder_config()
    workspace = 2
    config.max_workspace_size = workspace * 1 << 30
    flag = (1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    network = builder.create_network(flag)
    parser = trt.OnnxParser(network, logger)
    if not parser.parse_from_file(onnx_model_path):
        raise RuntimeError(f'failed to load ONNX file: {onnx_model_path}')
        pass

    inputs = [network.get_input(i) for i in range(network.num_inputs)]
    outputs = [network.get_output(i) for i in range(network.num_outputs)]
    #print(f"network inputs: \n{inputs}\noutputs: \n{outputs}")
    for inp in inputs:
        if MY_VERBOSE: print(f'input "{inp.name}" with shape{inp.shape} {inp.dtype}')
        #pass
    for out in outputs:
        if MY_VERBOSE: print(f'output "{out.name}" with shape{out.shape} {out.dtype}')
        #pass
    
    '''
    network inputs:
    [<tensorrt_bindings.tensorrt.ITensor object at 0x7f3c9303efb0>, <tensorrt_bindings.tensorrt.ITensor object at 0x7f3c9303ensorrt_bindings.tensorrt.ITensor object at 0x7f3c9303ef30>]
    outputs:
    [<tensorrt_bindings.tensorrt.ITensor object at 0x7f3c9303eef0>]
    input "latent_model_input" with shape(2, 4, 64, 32, 32) DataType.HALF
    input "timestep" with shape(1,) DataType.INT32
    input "prompt_embeds" with shape(2, 77, 1024) DataType.HALF
    output "noise_pred" with shape(2, 4, 64, 32, 32) DataType.HALF
    '''
    profile = builder.create_optimization_profile()
    profile.set_shape('latent_model_input', 
                      (2, 4, 64, 32, 32), 
                      (2, 4, 64, 32, 32),
                      (2, 4, 64, 32, 32))
    profile.set_shape('timestep', 
                      (1,), 
                      (1,), 
                      (1,))
    profile.set_shape('prompt_embeds', 
                      (2, 77, 1024), 
                      (2, 77, 1024), 
                      (2, 77, 1024))
    profile.set_shape('noise_pred', 
                      (2, 4, 64, 32, 32), 
                      (2, 4, 64, 32, 32), 
                      (2, 4, 64, 32, 32))
    config.add_optimization_profile(profile)

    use_fp16 = builder.platform_has_fast_fp16 and fp16
    if MY_VERBOSE: print(f'building FP{16 if use_fp16 else 32} engine as {engine_path}')
    if use_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
    with builder.build_engine(network, config) as engine, open(engine_path, 'wb') as t:
        t.write(engine.serialize())

def trt_build(onnx_path, engine_path, fp16, input_profile=None, 
              enable_refit=False, enable_preview=False, enable_all_tactics=False, 
              timing_cache=None, workspace_size=0):
    if MY_VERBOSE: print(f"Building TensorRT engine for {onnx_path}: {engine_path}")
    p = Profile()
    if input_profile:
        for name, dims in input_profile.items():
            assert len(dims) == 3
            p.add(name, min=dims[0], opt=dims[1], max=dims[2])

    config_kwargs = {}
    config_kwargs['preview_features'] = [trt.PreviewFeature.DISABLE_EXTERNAL_TACTIC_SOURCES_FOR_CORE_0805]
    if enable_preview:
        # Faster dynamic shapes made optional since it increases engine build time.
        config_kwargs['preview_features'].append(trt.PreviewFeature.FASTER_DYNAMIC_SHAPES_0805)
    if workspace_size > 0:
        config_kwargs['memory_pool_limits'] = {trt.MemoryPoolType.WORKSPACE: workspace_size}
    if not enable_all_tactics:
        config_kwargs['tactic_sources'] = []

    engine = engine_from_network(
        network_from_onnx_path(onnx_path),
        config=CreateConfig(fp16=fp16,
            refittable=enable_refit,
            profiles=[p],
            load_timing_cache=timing_cache,
            **config_kwargs
        ),
        save_timing_cache=timing_cache
    )
    save_engine(engine, engine_path)
    return engine

def build_trt_engine_onnx_polygraphy(engine_path="unet_polygraphy.engine", onnx_model_path="unet.onnx", fp16=True):
    def cal_max_workspace_size():
        _, free_mem, _ = cudart.cudaMemGetInfo()
        GiB = 2 ** 30
        if free_mem > 6*GiB:
            activation_carveout = 4*GiB
            max_workspace_size = free_mem - activation_carveout
        else:
            max_workspace_size = 0
        return max_workspace_size

    # Build TensorRT engines
    if not os.path.exists(engine_path):
        input_profile = {'latent_model_input': [(2, 4, 64, 32, 32), 
                                                (2, 4, 64, 32, 32),
                                                (2, 4, 64, 32, 32)],
                        'timestep':[(1,), 
                                    (1,), 
                                    (1,)],
                        'prompt_embeds': [(2, 77, 1024), 
                                          (2, 77, 1024), 
                                          (2, 77, 1024)],
                        # this is outputs:
                        #'noise_pred': [(2, 4, 64, 32, 32), 
                        #               (2, 4, 64, 32, 32), 
                        #               (2, 4, 64, 32, 32)],
                        }
        engine = trt_build(onnx_model_path, engine_path,
                            fp16=fp16,
                            input_profile=input_profile,
                            enable_refit=False,
                            enable_preview=False,
                            enable_all_tactics=False,
                            timing_cache="./tmp/.cache",
                            workspace_size=cal_max_workspace_size())
    else:
        engine = None
    return engine

class ONNX_Wraper():
    def __init__(self, onnx_path:str, provider:str):
        
        self.load_onnx_model(onnx_path, provider)

    def load_onnx_model(self, onnx_path:str, provider:str):
        providers = onnxruntime.get_available_providers()
        assert provider in providers
        if MY_VERBOSE: print(f"onnx providers: {providers}, try to use {provider}")
        self.ort_session = onnxruntime.InferenceSession(onnx_path, providers=[provider])

    def inference(self, torch_inputs):
        if MY_VERBOSE: print(f"onnxruntime inferencing via torch tensor inputs.......")

        assert len(torch_inputs)==3
        ort_inputs = {
        "latent_model_input": torch_inputs[0].cpu().numpy(),
        "timestep": torch_inputs[1].cpu().numpy(),
        "prompt_embeds": torch_inputs[2].cpu().numpy(),
        }
        ort_sess_outputs = self.ort_session.run(None, ort_inputs)
        if MY_VERBOSE: print(f"ort_sess_outputs: {ort_sess_outputs[0,0,0,0]}")
        return ort_sess_outputs
    
def build_TensorRT_engine_CLI( src_onnx_path:str="./model.onnx", dst_trt_engine_path:str="./model.engine" ):
    assert dst_trt_engine_path
    assert dst_trt_engine_path.endswith(".engine"), "to be built TRT model file should end with .engine"
    if not os.path.exists(dst_trt_engine_path) and src_onnx_path:
        print(f"building TensorRT engine: {dst_trt_engine_path}")
        command = f"trtexec --onnx={src_onnx_path} --saveEngine={dst_trt_engine_path} --workspace=4096  --exportOutput=output.json"
        subprocess.call(command, shell=True)
'''
usage:
if trt_engine is None: 
    trt_engine = TRT_Engine(trt_engine_path, gpu_id=gpu, torch_stream=torch_stream)
    assert trt_engine
    ......
    if trt_engine:
        trt_output = trt_engine.inference(inputs=[midas_input.to(torch.float32)],
                                                  outputs = trt_engine.output_tensors)
        output = trt_engine.output_tensors[0].to(torch.float16)
    else:
        output = torch_model(midas_input)
'''

class TRT_Engine():
    def __init__(self, trt_engine_path:str, gpu_id, torch_stream=None, onnx_inputs_name:str=None, onnx_outputs_name:str=None):
        if not os.path.exists(trt_engine_path):
            assert False
        self.trt_engine_path = trt_engine_path
        self.gpu_id = gpu_id
        self.torch_stream = torch_stream
        self.onnx_inputs_name = onnx_inputs_name
        self.onnx_outputs_name = onnx_outputs_name
        
        trt.init_libnvinfer_plugins(trt.Logger(trt.Logger.ERROR), '')
        self._init_execution_context( self.trt_engine_path )
        
    def _init_execution_context( self, trt_engine_path ):
        if True: # deserialize TRT engines
            if os.path.exists(trt_engine_path):
                if MY_VERBOSE: print(f"Loading saved TRT engine from {trt_engine_path}")
                rng = nvtx.start_range(message="deserialize", color="blue")
                with open(trt_engine_path, "rb") as f:
                    runtime = trt.Runtime(trt.Logger(trt.Logger.VERBOSE))
                    runtime.max_threads = 10
                    self.engine = runtime.deserialize_cuda_engine(f.read())
                nvtx.end_range(rng)
                print_bindings( self.engine )
            else:
                self.engine = None
                if MY_VERBOSE: print("path is not exist!")
                #raise RuntimeError(f'failed to load TRT engine file: {trt_engine_path}')   
            print(f"{trt_engine_path}: {self.engine}")
            
        if self.engine is not None: # create execution context
            print(f"trt_engine_path: {trt_engine_path}")
            rng = nvtx.start_range(message="trt_exec_context", color="blue")
            self.trt_exec_context = self.engine.create_execution_context()
            nvtx.end_range(rng)

            rng = nvtx.start_range(message="allocate_trt_buffers", color="blue")
            create_stream = True if self.torch_stream is None else False
            self.inputs, self.outputs, \
            self.bindings, \
            cur_stream, \
            self.input_tensors, self.output_tensors, \
            self.inputs_name, self.outputs_name = self.allocate_buffers(self.engine, gpu_id=self.gpu_id, profile_idx=None, create_stream=create_stream)
            self.torch_stream = cur_stream if create_stream else self.torch_stream
            nvtx.end_range(rng)

            if MY_VERBOSE: print(f"trt engine {trt_engine_path}:")
            if MY_VERBOSE: print(f"inputs: {self.inputs}")
            if MY_VERBOSE: print(f"outputs: {self.outputs}")
            if MY_VERBOSE: print(f"bindings: {self.bindings}")

    # Allocates all buffers required for an engine, i.e. host/device inputs/outputs.
    # If engine uses dynamic shapes, specify a profile to find the maximum input & output size.
    def allocate_buffers(self, engine: trt.ICudaEngine, gpu_id, 
                         profile_idx: Optional[int] = None, create_stream=True):
        inputs = []
        outputs = []
        inputs_name = []
        outputs_name = []
        bindings = []
        input_tensors = []
        output_tensors = []
        #stream = cuda_call(cudart.cudaStreamCreate()) if create_stream else None
        stream = torch.cuda.Stream() if create_stream else None
        tensor_names = [engine.get_tensor_name(i) for i in range(engine.num_io_tensors)]
        if MY_VERBOSE: print(f"tensor_names: {tensor_names}")
        print(f"engine.max_batch_size = {engine.max_batch_size}")
        for i, binding in enumerate(tensor_names):
            # get_tensor_profile_shape returns (min_shape, optimal_shape, max_shape)
            # Pick out the max shape to allocate enough memory for the binding.
            shape = engine.get_tensor_shape(binding) if profile_idx is None else engine.get_tensor_profile_shape(binding, profile_idx)[-1]
            #if binding=="masks":# magic code
            #    shape = engine.get_tensor_shape(binding)
            #if binding=="scores":
            #    shape = engine.get_tensor_shape(binding)

            shape_valid = np.all([s >= 0 for s in shape])
            if not shape_valid and profile_idx is None:
                raise ValueError(f"Binding {binding} has dynamic shape, " +\
                    "but no profile was specified.")
            size = trt.volume(shape)
            if engine.has_implicit_batch_dimension:
                size *= engine.max_batch_size
            
            #dtype = np.dtype(trt.nptype(engine.get_tensor_dtype(binding)))
            dtype = trt.nptype(engine.get_binding_dtype(binding))
            print(f"{binding} size=shape * maxbs: {size}={shape} x {engine.max_batch_size}, dtype={dtype}")

            # Allocate host and device buffers
            bindingMemory = HostDeviceMem(size, np.dtype(dtype), binding, idx_in_trt=i)

            # Append the device buffer to device bindings.
            bindings.append(int(bindingMemory.device))
            print(f"numpy_to_torch_dtype_dict[dtype]: {numpy_to_torch_dtype_dict[dtype]}")
            tensor = torch.empty(tuple(shape), dtype=numpy_to_torch_dtype_dict[dtype]).to(device=gpu_id)
            tensor = tensor.contiguous()
            # Append to the appropriate list.
            if engine.get_tensor_mode(binding) == trt.TensorIOMode.INPUT:
                input_tensors.append(tensor)
                inputs.append(bindingMemory)
                inputs_name.append(binding)
            else:
                output_tensors.append(tensor)
                outputs.append(bindingMemory)
                outputs_name.append(binding)
            
        return inputs, outputs, bindings, stream, input_tensors, output_tensors, inputs_name, outputs_name

    def inference_from_HostDeviceMem(self, context, bindings, inputs, outputs, stream, no_data_transfer_list=[], 
                                     is_sync=True, enable_D2H=True):
        if MY_VERBOSE: print(f"trt inferencing via HostDeviceMem inputs.......")
        def execute_async():
            context.execute_async_v2(bindings=bindings, stream_handle=stream)

        trt_outputs = _do_inference_base(inputs, outputs, stream, execute_async, no_data_transfer_list, 
                                         is_sync, enable_D2H=enable_D2H)

        if MY_VERBOSE: print(f"trt_outputs: {trt_outputs}")
    
    def inference_from_torch_tensor(self, context, bindings, inputs, outputs, stream, no_data_transfer_list=[], 
                                    is_sync=True, enable_D2H=True):
        if MY_VERBOSE: print(f"trt inferencing via torch.Tensor inputs.......")
        start_binding, end_binding = trt_util.get_active_profile_bindings(context)
        print(f"start_binding, end_binding: {start_binding, end_binding}")
        assert end_binding==(start_binding+len(inputs)+len(outputs))
        inputs_ptr = []
        for i, input in enumerate(inputs):
            if MY_VERBOSE: print(f"input {i}: {input.shape, input.device, input.dtype}")
            inputs_ptr.append(pg_cuda.DeviceView(ptr=input.data_ptr(), shape=input.shape, dtype=torch_to_numpy_dtype_dict[input.dtype]))
        def execute_async():
            context.execute_async_v2(bindings=bindings, stream_handle=stream)

        trt_outputs = _do_inference_base(self.inputs, outputs, stream, execute_async, no_data_transfer_list, 
                                         is_sync, enable_D2H=enable_D2H, inputs_ptr=inputs_ptr)

        if MY_VERBOSE: print(f"trt_outputs[0]: {trt_outputs[0].shape}")
        return trt_outputs
    
    def inference_from_torch_tensor_external_inputs_outputs(self, context, inputs, outputs, stream):
        if MY_VERBOSE: print(f"trt inferencing via torch.Tensor inputs and outputs.......")
        start_binding, end_binding = trt_util.get_active_profile_bindings(context)
        assert end_binding==(start_binding+len(inputs)+len(outputs))
        inputs_ptr = []
        outputs_ptr = []
        new_bindings = []
        for i, input in enumerate(inputs):
            if MY_VERBOSE: print(f"input {i}: {input.shape, input.device, input.dtype}")
            inputs_ptr.append(pg_cuda.DeviceView(ptr=input.data_ptr(), shape=input.shape, dtype=torch_to_numpy_dtype_dict[input.dtype]))
        for i, output in enumerate(outputs):
            if MY_VERBOSE: print(f"outputs {i}: {output.shape, output.device, output.dtype}")
            outputs_ptr.append(pg_cuda.DeviceView(ptr=output.data_ptr(), shape=output.shape, dtype=torch_to_numpy_dtype_dict[output.dtype]))
        
        for ptr in inputs_ptr:
            new_bindings.append(ptr.ptr)
        for ptr in outputs_ptr:
            new_bindings.append(ptr.ptr)
        def execute_async():
            context.execute_async_v2(bindings=new_bindings, stream_handle=stream)

        execute_async()
        trt_outputs = None

        return trt_outputs

    def inference_from_torch_tensor_external_outputs(self, context, bindings, inputs, outputs, stream, no_data_transfer_list=[], 
                                    is_sync=False, enable_D2H=False):
        if MY_VERBOSE: print(f"trt inferencing via torch.Tensor outputs.......")
        start_binding, end_binding = trt_util.get_active_profile_bindings(context)
        assert end_binding==(start_binding+len(inputs)+len(outputs))
        inputs_ptr = []
        outputs_ptr = []
        new_bindings = []
        for i, input in enumerate(inputs):
            if MY_VERBOSE: print(f"input {i}: {input.shape, input.device, input.dtype}")
            inputs_ptr.append(pg_cuda.DeviceView(ptr=input.data_ptr(), shape=input.shape, dtype=torch_to_numpy_dtype_dict[input.dtype]))
        for i, output in enumerate(outputs):
            if MY_VERBOSE: print(f"outputs {i}: {output.shape, output.device, output.dtype}")
            outputs_ptr.append(pg_cuda.DeviceView(ptr=output.data_ptr(), shape=output.shape, dtype=torch_to_numpy_dtype_dict[output.dtype]))
        
        for ptr in inputs_ptr:
            new_bindings.append(ptr.ptr)
        for ptr in outputs_ptr:
            new_bindings.append(ptr.ptr)
        def execute_async():
            context.execute_async_v2(bindings=new_bindings, stream_handle=stream)

        trt_outputs = _do_inference_base(self.inputs, outputs, stream, execute_async, no_data_transfer_list, 
                                         is_sync=is_sync, enable_D2H=enable_D2H, inputs_ptr=inputs_ptr)

        if MY_VERBOSE: print(f"trt_outputs[0]: {trt_outputs[0].shape}")
        return trt_outputs

    def inference(self, inputs, outputs=None, is_sync=True, enable_D2H=True):
        for i,si in zip(inputs, self.inputs):
                if MY_VERBOSE: print(f"input {si.name} shape = {i.shape}")

                self.trt_exec_context.set_input_shape(si.name, i.shape)

        if isinstance(inputs, list) and len(inputs)>0 and torch.is_tensor(inputs[0]):
            inputs = [t.contiguous() for t in inputs]
            for input in inputs:
                assert input.is_cuda
            
            if torch.is_tensor(inputs[0]) and torch.is_tensor(outputs[0]) and inputs[0].is_cuda and outputs[0].is_cuda:
                self.inference_from_torch_tensor_external_inputs_outputs( self.trt_exec_context, 
                                                                inputs, outputs, 
                                                                self.torch_stream.cuda_stream)
                #make sure trt output tensor order is the same as that for onnx & torch native
                #print(f"self.onnx_outputs_name: {self.onnx_outputs_name}")
                #print(f"self.outputs_name: {self.outputs_name}")
                if self.onnx_outputs_name is not None:
                    new_trt_outputs = []
                    for i in range(len(self.onnx_outputs_name)):
                        if self.onnx_outputs_name[i] in self.outputs_name:
                            new_trt_outputs.append(self.output_tensors[self.outputs_name.index(self.onnx_outputs_name[i])])
                    assert len(new_trt_outputs)==len(self.output_tensors)
                    trt_outputs = new_trt_outputs
                else:
                    trt_outputs = self.output_tensors
            elif outputs is not None:
                self.inference_from_torch_tensor_external_outputs( self.trt_exec_context, self.bindings, 
                                                            inputs, outputs, 
                                                            self.torch_stream.cuda_stream, no_data_transfer_list=[])
                #make sure trt output tensor order is the same as that for onnx & torch native
                #print(f"self.onnx_outputs_name: {self.onnx_outputs_name}")
                #print(f"self.outputs_name: {self.outputs_name}")
                if self.onnx_outputs_name is not None:
                    new_trt_outputs = []
                    for i in range(len(self.onnx_outputs_name)):
                        if self.onnx_outputs_name[i] in self.outputs_name:
                            new_trt_outputs.append(self.output_tensors[i])
                    assert len(new_trt_outputs)==len(self.output_tensors)
                    trt_outputs = new_trt_outputs
                else:
                    trt_outputs = self.output_tensors
            else:
                trt_outputs = self.inference_from_torch_tensor( self.trt_exec_context, self.bindings, 
                                                            inputs, self.outputs, 
                                                            self.torch_stream.cuda_stream, no_data_transfer_list=[], 
                                                            is_sync=is_sync,
                                                            enable_D2H = enable_D2H)[0]
        else:
            trt_outputs = self.inference_from_HostDeviceMem( self.trt_exec_context, self.bindings, 
                                                            inputs, self.outputs, 
                                                            self.torch_stream.cuda_stream, no_data_transfer_list=[], 
                                                            is_sync=is_sync,
                                                            enable_D2H = enable_D2H )[0]
        if MY_VERBOSE and trt_outputs is not None: 
            for i, t in enumerate(trt_outputs):
                pass #print(f"trt_outputs[{i}]: {t.shape, t.dtype, t.device}")
        
        return trt_outputs

    def __del__(self):
        #free_buffers(self.inputs, self.outputs, None)
        #free_stream(self.torch_stream.cuda_stream)
        del self.torch_stream

def synchronize(torch_stream=None):
    if torch_stream is None:
        cudart.cudaDeviceSynchronize()
        torch.cuda.synchronize()
    else:
        cuda_call(cudart.cudaStreamSynchronize(torch_stream.cuda_stream)) 
        cudart.cudaDeviceSynchronize()
        torch.cuda.synchronize()

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

def tensor_nbytes(tensor):
    shape = tensor.shape
    d = 1
    for i in list(shape):
        d *= i
    dtype_in_bytes = np.dtype(torch_to_numpy_dtype_dict[tensor.dtype]).itemsize
    d *= dtype_in_bytes
    return d

def torch_cuda_tensor_deep_copy(dst_tensor, src_tensor, stream=None):
    assert torch.equal(torch.tensor(dst_tensor.shape), torch.tensor(src_tensor.shape))
    shape = src_tensor.shape
    dst_inbytes = tensor_nbytes(dst_tensor)
    src_inbytes = tensor_nbytes(src_tensor)
    assert dst_inbytes==src_inbytes

    dst_ptr = pg_cuda.DeviceView(ptr=dst_tensor.data_ptr(), shape=dst_tensor.shape, dtype=torch_to_numpy_dtype_dict[dst_tensor.dtype]).ptr
    src_ptr = pg_cuda.DeviceView(ptr=src_tensor.data_ptr(), shape=src_tensor.shape, dtype=torch_to_numpy_dtype_dict[src_tensor.dtype]).ptr
    
    memcpy_device_to_device(dst_ptr, src_ptr, src_inbytes, stream)

class FIFOTorchCUDATensors():
    def __init__(self, shape, dtype, device="cuda:0", len=2):
        self.tensors = []
        for i in range(len):
            tensor = torch.empty(tuple(shape), dtype=dtype).to(device=device).contiguous()
            self.tensors.append(tensor)
        self.start = 0
        self.num = 0

    def is_empty(self):
        if self.num == 0:
            return True
        else:
            return False
    
    def is_full(self):
        if self.num == len(self.tensors):
            assert False, "FIFO can't be full."
            return True
        else:
            return False

    def pop(self):
        tensor = None
        if not self.is_empty():
            tensor = self.tensors[self.start]
            self.start = (self.start+1)%(len(self.tensors))
            self.num -= 1

            print(f"FIFO pop:  {self.start}/{len(self.tensors)}, {self.num}, [{self.start}, {(self.start+self.num)%(len(self.tensors))})")
        
        return tensor
    
    def push(self, tensor, torch_stream=None):
        if not self.is_full():
            curIdx = (self.start+self.num)%(len(self.tensors))
            torch_cuda_tensor_deep_copy( self.tensors[curIdx], tensor, stream=torch_stream.cuda_stream )
            self.num += 1

            print(f"FIFO push: {self.start}/{len(self.tensors)}, {self.num}, [{self.start}, {(self.start+self.num)%(len(self.tensors))})")

class CUDA_Event():
    def __init__(self, stream, name=""):
        self.stream = stream
        self.name = name

        self.start_ = cudart.cudaEventCreate()[1]
        self.end_ = cudart.cudaEventCreate()[1]

    def start(self):
        cudart.cudaEventRecord(self.start_, self.stream.cuda_stream)
    
    def end(self):
        cudart.cudaEventRecord(self.end_, self.stream.cuda_stream)
    
    def print(self):
        print('| {:^10} | {:>9.2f} ms |'.format(self.name, 
                                                cudart.cudaEventElapsedTime(self.start_, 
                                                                            self.end_)[1]))
        
#---------------------------------------------------------------------------------------------------
def export_to_video(video_frames: List[np.ndarray], output_video_dir: str = None) -> str:
    if output_video_dir is None:
        output_video_dir = tempfile.NamedTemporaryFile(suffix=".mp4").name
    else:
        output_video_path = os.path.join(output_video_dir, "my_video.mp4")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    h, w, c = video_frames[0].shape
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps=8, frameSize=(w, h))
    print(f"len(video_frames) = {len(video_frames)}")
    for i in range(len(video_frames)):
        img = cv2.cvtColor(video_frames[i], cv2.COLOR_RGB2BGR)
        print(f"img{i}: {img.shape, img}\n\n\n\n\n\n\n\n")
        video_writer.write(img)

        cv2.imwrite(os.path.join(output_video_dir, "{0:03d}.jpg".format(i)), img)
    video_writer.release()
    return output_video_path

def export_to_video_1(video_frames: List[np.ndarray], output_video_dir: str = None) -> str:
    if output_video_dir is None:
        output_video_dir = tempfile.NamedTemporaryFile(suffix=".mp4").name
    else:
        output_video_path = os.path.join(output_video_dir, "my_video.mp4")

    print(f"len(video_frames) = {len(video_frames)}")
    for i in range(len(video_frames)):
        img = cv2.cvtColor(video_frames[i], cv2.COLOR_RGB2BGR)
        #print(f"img{i}: {img.shape, img}\n\n\n\n\n\n\n\n")

        cv2.imwrite(os.path.join(output_video_dir, "{0:03d}.jpg".format(i)), img)

    if True:
        image_folder=output_video_dir
        fps=8

        image_files = [os.path.join(image_folder,img)
                    for img in os.listdir(image_folder)
                    if img.endswith(".jpg")]
        #print(image_files)
        clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(image_files, fps=fps)
        
        output_video_path = os.path.join(output_video_dir, 'my_video.mp4')
        clip.write_videofile(output_video_path)
    
    return output_video_path

def export_to_video_2(video_frames: List[np.ndarray], output_video_dir: str = None) -> str:
    image_folder=output_video_dir
    fps=8

    image_files = [os.path.join(image_folder,img)
                for img in os.listdir(image_folder)
                if img.endswith(".jpg")]
    print(image_files)
    clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(image_files, fps=fps)
    
    output_video_path = os.path.join(output_video_dir, 'my_video.mp4')
    clip.write_videofile(output_video_path)

    return output_video_path

def torch_max_GPU_memory_allocated():
    max_memory_allocated = torch.cuda.max_memory_allocated() / (1024 ** 3) 
    print(f"max memory allocated: {max_memory_allocated:.3f} GB.")

'''
import nvtx
......
rng0 = nvtx.start_range(message="init", color="red")
xxxxxx
nvtx.end_range(rng0)
xxxxxx
rng1 = nvtx.start_range(message="infer", color="blue")
xxxxxx
nvtx.end_range(rng1)
'''
class NVTXUtil:
    def __init__(self, message="", color='red', memory_manager:Memory_Manager=None):
        self.message = message
        self.color = color
        self.memory_manager = memory_manager
    def __enter__(self):
        self.rng = nvtx.start_range(message=self.message, color=self.color)
        return self.rng
    def __exit__(self, exceptionType, exceptionVal, trace):
        nvtx.end_range(self.rng)
        if self.memory_manager is not None:
            self.memory_manager.add_foot_print( self.message )

class SynchronizeUtil:
    def __init__(self, torch_stream=None):
        self.torch_stream = torch_stream
    def __enter__(self):
        synchronize( self.torch_stream )
        return None
    def __exit__(self, exceptionType, exceptionVal, trace):
        synchronize( self.torch_stream )
        

class TorchUtil:
    def __init__(self, gpu=0, memory_manager:Memory_Manager=None, cvcuda_stream=None):
        self.gpu = gpu
        self.memory_manager = memory_manager
        torch.cuda.set_device(self.gpu)
        if cvcuda_stream is None:
            self.torch_stream = torch.cuda.Stream()
        else:
            self.torch_stream = torch.cuda.ExternalStream(cvcuda_stream.handle)
        if self.torch_stream: torch.cuda.set_stream(self.torch_stream)

    '''
    with torch.inference_mode():
        with torch.autocast(device_type='cuda', enabled=True, dtype=torch.float16):
            video_frames = pipe(prompt, video=video, strength=0.6).frames
    '''
    def use_fp16(self):
        assert False
#---------------------------------------------------------------------------------------------------