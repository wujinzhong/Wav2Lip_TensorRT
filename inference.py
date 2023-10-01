from os import listdir, path
import numpy as np
import scipy, cv2, os, sys, argparse, audio
import json, subprocess, random, string
from tqdm import tqdm
from glob import glob
import torch, face_detection
from models import Wav2Lip
import platform

import nvtx, warnings
from cuda import cudart
from InferenceUtil import (NVTXUtil, 
                           check_onnx, 
                           TRT_Engine, 
                           SynchronizeUtil, 
                           TorchUtil, 
                           Memory_Manager,
                           FIFOTorchCUDATensors,
                           synchronize,
						   build_TensorRT_engine_CLI,
						   save_tensors_as_json,
						   compare_2_tensors,
						   save_tensors_to_numpy_dat,
                           )

parser = argparse.ArgumentParser(description='Inference code to lip-sync videos in the wild using Wav2Lip models')

parser.add_argument('--checkpoint_path', type=str, 
					help='Name of saved checkpoint to load weights from', required=True)

parser.add_argument('--face', type=str, 
					help='Filepath of video/image that contains faces to use', required=True)

parser.add_argument('--export_wav2lip_onnx', type=str, 
					help='Filepath of video/image that contains faces to use', required=True)
parser.add_argument('--export_wav2lip_trt', type=str, 
					help='Filepath of video/image that contains faces to use', required=True)

parser.add_argument('--export_s3fd_onnx', type=str, 
					help='Filepath of video/image that contains faces to use', required=True)
parser.add_argument('--export_s3fd_trt', type=str, 
					help='Filepath of video/image that contains faces to use', required=True)

parser.add_argument('--audio', type=str, 
					help='Filepath of video/audio file to use as raw audio source', required=True)
parser.add_argument('--outfile', type=str, help='Video path to save result. See default for an e.g.', 
								default='results/result_voice.mp4')

parser.add_argument('--static', type=bool, 
					help='If True, then use only first video frame for inference', default=False)
parser.add_argument('--fps', type=float, help='Can be specified only if input is a static image (default: 25)', 
					default=25., required=False)

parser.add_argument('--pads', nargs='+', type=int, default=[0, 10, 0, 0], 
					help='Padding (top, bottom, left, right). Please adjust to include chin at least')

parser.add_argument('--warm_up', type=int, 
					help='AI model warm up', default=0)

parser.add_argument('--face_det_batch_size', type=int, 
					help='Batch size for face detection', default=43) #86
parser.add_argument('--wav2lip_batch_size', type=int, help='Batch size for Wav2Lip model(s)', default=128)

parser.add_argument('--resize_factor', default=1, type=int, 
			help='Reduce the resolution by this factor. Sometimes, best results are obtained at 480p or 720p')

parser.add_argument('--crop', nargs='+', type=int, default=[0, -1, 0, -1], 
					help='Crop video to a smaller region (top, bottom, left, right). Applied after resize_factor and rotate arg. ' 
					'Useful if multiple face present. -1 implies the value will be auto-inferred based on height, width')

parser.add_argument('--box', nargs='+', type=int, default=[-1, -1, -1, -1], 
					help='Specify a constant bounding box for the face. Use only as a last resort if the face is not detected.'
					'Also, might work only if the face is not moving around much. Syntax: (top, bottom, left, right).')

parser.add_argument('--rotate', default=False, action='store_true',
					help='Sometimes videos taken from a phone can be flipped 90deg. If true, will flip video right by 90deg.'
					'Use if you get a flipped result, despite feeding a normal looking video')

parser.add_argument('--nosmooth', default=False, action='store_true',
					help='Prevent smoothing face detections over a short temporal window')

args = parser.parse_args()
args.img_size = 96

if os.path.isfile(args.face) and args.face.split('.')[1] in ['jpg', 'png', 'jpeg']:
	args.static = True

def get_smoothened_boxes(boxes, T):
	for i in range(len(boxes)):
		if i + T > len(boxes):
			window = boxes[len(boxes) - T:]
		else:
			window = boxes[i : i + T]
		boxes[i] = np.mean(window, axis=0)
	return boxes

def face_detect(sfd_detector, images):
	#detector = face_detection.FaceAlignment(face_detection.LandmarksType._2D, 
	#										flip_input=False, device=device)
	detector = sfd_detector
	batch_size = args.face_det_batch_size
	
	olist = [None for i in range(len(images)//batch_size)]
	BB = [None for i in range(len(images)//batch_size)]
	images_list = [None for i in range(len(images)//batch_size)]

	while 1:
		predictions = []
		try:
			for i in tqdm(range(0, len(images), batch_size)):
				images_list[i//batch_size] = np.array(images[i:i + batch_size])[...,::-1]
		except RuntimeError:
			if batch_size == 1: 
				raise RuntimeError('Image too big to run face detection on GPU. Please use the --resize_factor argument')
			batch_size //= 2
			print('Recovering from OOM error; New batch size: {}'.format(batch_size))
			continue
		break
	
	dst_olist = [[] for i in range(len(images)//batch_size)]
	events_list = []
	for i in range(len(dst_olist)):
		dst_olist[i].append( torch.empty((43, 2, 120, 180), dtype=torch.float32, device="cpu", pin_memory=True) )
		dst_olist[i].append( torch.empty((43, 4, 120, 180), dtype=torch.float32, device="cpu", pin_memory=True) )
		dst_olist[i].append( torch.empty((43, 2, 60, 90), dtype=torch.float32, device="cpu", pin_memory=True) )
		dst_olist[i].append( torch.empty((43, 4, 60, 90), dtype=torch.float32, device="cpu", pin_memory=True) )
		dst_olist[i].append( torch.empty((43, 2, 30, 45), dtype=torch.float32, device="cpu", pin_memory=True) )
		dst_olist[i].append( torch.empty((43, 4, 30, 45), dtype=torch.float32, device="cpu", pin_memory=True) )
		dst_olist[i].append( torch.empty((43, 2, 19, 26), dtype=torch.float32, device="cpu", pin_memory=True) )
		dst_olist[i].append( torch.empty((43, 4, 19, 26), dtype=torch.float32, device="cpu", pin_memory=True) )
		dst_olist[i].append( torch.empty((43, 2, 10, 13), dtype=torch.float32, device="cpu", pin_memory=True) )
		dst_olist[i].append( torch.empty((43, 4, 10, 13), dtype=torch.float32, device="cpu", pin_memory=True) )
		dst_olist[i].append( torch.empty((43, 2, 5, 7), dtype=torch.float32, device="cpu", pin_memory=True) )
		dst_olist[i].append( torch.empty((43, 4, 5, 7), dtype=torch.float32, device="cpu", pin_memory=True) )

	while 1:
		predictions = []
		try:
			for i in tqdm(range(0, len(images), batch_size)):
				olist_, BB_ = detector.get_detections_for_batch(images_list[i//batch_size], dst_olist[i//batch_size], events_list)
				olist[i//batch_size] = olist_
				BB[i//batch_size] = BB_
		except RuntimeError:
			if batch_size == 1: 
				raise RuntimeError('Image too big to run face detection on GPU. Please use the --resize_factor argument')
			batch_size //= 2
			print('Recovering from OOM error; New batch size: {}'.format(batch_size))
			continue
		break

	#cudart.cudaDeviceSynchronize()
	#torch.cuda.synchronize()
	while 1:
		predictions = []
		try:
			for i in tqdm(range(0, len(images), batch_size)):
				events_list[i//batch_size].synchronize()
				results = detector.get_detections_for_batch_post(olist[i//batch_size], BB[i//batch_size])
				predictions.extend(results)
		except RuntimeError:
			if batch_size == 1: 
				raise RuntimeError('Image too big to run face detection on GPU. Please use the --resize_factor argument')
			batch_size //= 2
			print('Recovering from OOM error; New batch size: {}'.format(batch_size))
			continue
		break

	results = []
	pady1, pady2, padx1, padx2 = args.pads
	for rect, image in zip(predictions, images):
		if rect is None:
			cv2.imwrite('temp/faulty_frame.jpg', image) # check this frame where the face was not detected.
			raise ValueError('Face not detected! Ensure the video contains a face in all the frames.')

		y1 = max(0, rect[1] - pady1)
		y2 = min(image.shape[0], rect[3] + pady2)
		x1 = max(0, rect[0] - padx1)
		x2 = min(image.shape[1], rect[2] + padx2)
		
		results.append([x1, y1, x2, y2])

	boxes = np.array(results)
	if not args.nosmooth: boxes = get_smoothened_boxes(boxes, T=5)
	results = [[image[y1: y2, x1:x2], (y1, y2, x1, x2)] for image, (x1, y1, x2, y2) in zip(images, boxes)]

	del detector
	return results 

def cal_face_det_results(frames, sfd_detector):
	if args.box[0] == -1:
		if not args.static:
			face_det_results = face_detect(sfd_detector, frames) # BGR2RGB for CNN face detection
		else:
			face_det_results = face_detect(sfd_detector, [frames[0]])
	else:
		print('Using the specified bounding box instead of face detection...')
		y1, y2, x1, x2 = args.box
		face_det_results = [[f[y1: y2, x1:x2], (y1, y2, x1, x2)] for f in frames]
	
	return face_det_results

def datagen(frames, mels, face_det_results):
	img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []

	for i, m in enumerate(mels):
		rng = nvtx.start_range( message=f"yield{i}", color="blue" )

		idx = 0 if args.static else i%len(frames)
		frame_to_save = frames[idx].copy()
		face, coords = face_det_results[idx].copy()

		face = cv2.resize(face, (args.img_size, args.img_size))
			
		img_batch.append(face)
		mel_batch.append(m)
		frame_batch.append(frame_to_save)
		coords_batch.append(coords)

		nvtx.end_range(rng)
		if len(img_batch) >= args.wav2lip_batch_size:
			img_batch, mel_batch = np.asarray(img_batch), np.asarray(mel_batch)

			img_masked = img_batch.copy()
			img_masked[:, args.img_size//2:] = 0

			img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.
			mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])

			yield img_batch, mel_batch, frame_batch, coords_batch
			img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []

	if len(img_batch) > 0:
		rng = nvtx.start_range( message="yield_", color="blue" )

		img_batch, mel_batch = np.asarray(img_batch), np.asarray(mel_batch)

		img_masked = img_batch.copy()
		img_masked[:, args.img_size//2:] = 0

		img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.
		mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])

		nvtx.end_range(rng)

		yield img_batch, mel_batch, frame_batch, coords_batch

mel_step_size = 16
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} for inference.'.format(device))

def _load(checkpoint_path):
	if device == 'cuda':
		checkpoint = torch.load(checkpoint_path)
	else:
		checkpoint = torch.load(checkpoint_path,
								map_location=lambda storage, loc: storage)
	return checkpoint

def load_model(path):
	model = Wav2Lip()
	print("Load checkpoint from: {}".format(path))
	checkpoint = _load(path)
	s = checkpoint["state_dict"]
	new_s = {}
	for k, v in s.items():
		new_s[k.replace('module.', '')] = v
	model.load_state_dict(new_s)

	model = model.to(device)
	return model.eval()

def torch_onnx_export_wav2lip(onnx_model, fp16=False, onnx_model_path="model.onnx", maxBatch=1 ):
    if not os.path.exists(onnx_model_path):
        dynamic_axes = {
            "latent_model_input":   {0: "bs_x_2"},
            "prompt_embeds":        {0: "bs_x_2"},
            "noise_pred":           {0: "batch_size"}
        }

        device = torch.device("cuda:0")
        
        onnx_model2= onnx_model #onnx_model2= UNet_x(onnx_model)

        onnx_model2.eval()
        if isinstance(onnx_model2, torch.nn.DataParallel):
            onnx_model2 = onnx_model2.module

        onnx_model2 = onnx_model2.to(device=device)
        
        if fp16: dst_dtype = torch.float16
        else: dst_dtype = torch.float32

        '''
		mel_batch: (torch.Size([86, 1, 80, 16]), torch.float32, device(type='cuda', index=0))
		img_batch: (torch.Size([86, 6, 96, 96]), torch.float32, device(type='cuda', index=0))
		'''
        dummy_inputs = {
            "mel_batch": torch.randn((86, 1, 80, 16), dtype=dst_dtype).to(device).contiguous(),
			"img_batch": torch.randn((86, 6, 96, 96), dtype=dst_dtype).to(device).contiguous(),
        }
        # output_names = ["masks", "iou_predictions", "low_res_masks"]
        #output_names = ["pred0", "pred1", "pred2", "pred3", "pred4", "pred5", "pred6"]
        output_names = ["pred"]

        #import apex
        with torch.no_grad():
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
                warnings.filterwarnings("ignore", category=UserWarning)
                if True:
                    torch.onnx.export(
                        onnx_model2,
                        tuple(dummy_inputs.values()),
                        onnx_model_path, #f,
                        export_params=True,
                        verbose=True,
                        opset_version=16,
                        do_constant_folding=False,
                        input_names=list(dummy_inputs.keys()),
                        output_names=output_names,
                        #dynamic_axes=dynamic_axes,
                    )  
                else:
                    with open(onnx_model_path, "wb") as f:
                        torch.onnx.export(
                        onnx_model2,
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

def torch_onnx_export_s3fd(onnx_model, fp16=False, onnx_model_path=args.export_s3fd_onnx, maxBatch=1 ):
	if not os.path.exists(onnx_model_path):
		dynamic_axes = {
			"latent_model_input":   {0: "bs_x_2"},
			"prompt_embeds":        {0: "bs_x_2"},
			"noise_pred":           {0: "batch_size"}
		}

		device = torch.device("cuda:0")

		onnx_model2= onnx_model #onnx_model2= UNet_x(onnx_model)

		onnx_model2.eval()
		if isinstance(onnx_model2, torch.nn.DataParallel):
			onnx_model2 = onnx_model2.module

		onnx_model2 = onnx_model2.to(device=device)

		if fp16: dst_dtype = torch.float16
		else: dst_dtype = torch.float32

		'''
		net(imgs) imgs: (torch.Size([1, 3, 480, 720]), torch.float32, device(type='cuda', index=0))
		net(imgs) olist: (12, torch.Size([1, 2, 120, 180]), torch.float32, device(type='cuda', index=0))
		'''
		dummy_inputs = {
			"imgs": torch.randn((maxBatch, 3, 480, 720), dtype=dst_dtype).to(device).contiguous(),
		}
		output_names = ["olist0", 
						"olist1", 
						"olist2", 
						"olist3", 
						"olist4", 
						"olist5", 
						"olist6", 
						"olist7", 
						"olist8", 
						"olist9", 
						"olist10", 
						"olist11", ]

		#import apex
		with torch.no_grad():
			with warnings.catch_warnings():
				warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
				warnings.filterwarnings("ignore", category=UserWarning)
				if True:
					torch.onnx.export(
						onnx_model2,
						tuple(dummy_inputs.values()),
						onnx_model_path, #f,
						export_params=True,
						verbose=True,
						opset_version=16,
						do_constant_folding=False,
						input_names=list(dummy_inputs.keys()),
						output_names=output_names,
						#dynamic_axes=dynamic_axes,
					)  
				else:
					with open(onnx_model_path, "wb") as f:
						torch.onnx.export(
						onnx_model2,
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

def main():
	mm = Memory_Manager()
	mm.add_foot_print("prev-E2E")
	torchutil = TorchUtil(gpu=0, memory_manager=mm, cvcuda_stream=None)

	with NVTXUtil("init", "red", mm), SynchronizeUtil(torchutil.torch_stream):
		with NVTXUtil("load all frames", "red", mm), SynchronizeUtil(torchutil.torch_stream):
			if not os.path.isfile(args.face):
				raise ValueError('--face argument must be a valid path to video/image file')

			elif args.face.split('.')[1] in ['jpg', 'png', 'jpeg']:
				full_frames = [cv2.imread(args.face)]
				fps = args.fps

			else:
				video_stream = cv2.VideoCapture(args.face)
				fps = video_stream.get(cv2.CAP_PROP_FPS)

				print('Reading video frames...')

				full_frames = []
				while 1:
					still_reading, frame = video_stream.read()
					if not still_reading:
						video_stream.release()
						break
					if args.resize_factor > 1:
						frame = cv2.resize(frame, (frame.shape[1]//args.resize_factor, frame.shape[0]//args.resize_factor))

					if args.rotate:
						frame = cv2.rotate(frame, cv2.cv2.ROTATE_90_CLOCKWISE)

					y1, y2, x1, x2 = args.crop
					if x2 == -1: x2 = frame.shape[1]
					if y2 == -1: y2 = frame.shape[0]

					frame = frame[y1:y2, x1:x2]

					full_frames.append(frame)

		print ("Number of frames available for inference: "+str(len(full_frames)))

		if not args.audio.endswith('.wav'):
			with NVTXUtil("convert audio", "red", mm), SynchronizeUtil(torchutil.torch_stream):
				print('Extracting raw audio...')
				command = 'ffmpeg -y -i {} -strict -2 {}'.format(args.audio, 'temp/temp.wav')

				subprocess.call(command, shell=True)
				args.audio = 'temp/temp.wav'

		with NVTXUtil("load wav", "red", mm), SynchronizeUtil(torchutil.torch_stream):
			wav = audio.load_wav(args.audio, 16000)

		with NVTXUtil("mels", "red", mm), SynchronizeUtil(torchutil.torch_stream):
			mel = audio.melspectrogram(wav)
			print(mel.shape)

		if np.isnan(mel.reshape(-1)).sum() > 0:
			raise ValueError('Mel contains nan! Using a TTS voice? Add a small epsilon noise to the wav file and try again')

		with NVTXUtil("mel_chunks", "red", mm), SynchronizeUtil(torchutil.torch_stream):
			mel_chunks = []
			mel_idx_multiplier = 80./fps 
			i = 0
			while 1:
				start_idx = int(i * mel_idx_multiplier)
				if start_idx + mel_step_size > len(mel[0]):
					mel_chunks.append(mel[:, len(mel[0]) - mel_step_size:])
					break
				mel_chunks.append(mel[:, start_idx : start_idx + mel_step_size])
				i += 1

		
		print("Length of mel chunks: {}".format(len(mel_chunks)))
		full_frames = full_frames[:len(mel_chunks)]

		frame_h, frame_w = full_frames[0].shape[:-1]
		out = cv2.VideoWriter('temp/result.avi', 
								cv2.VideoWriter_fourcc(*'DIVX'), fps, (frame_w, frame_h))

		with NVTXUtil("datagen", "red", mm), SynchronizeUtil(torchutil.torch_stream):
			batch_size = args.wav2lip_batch_size

			with NVTXUtil("load_sfd", "red", mm), SynchronizeUtil(torchutil.torch_stream):
				sfd_detector = face_detection.FaceAlignment(face_detection.LandmarksType._2D, 
												flip_input=False, device=device)
				
				
				with NVTXUtil("onnx_s3fd", "blue", mm), SynchronizeUtil(torchutil.torch_stream):
					if args.export_s3fd_onnx:
						assert args.export_s3fd_onnx.endswith(".onnx"), "Export model file should end with .onnx"
						print(f"export sfd onnx: {args.export_s3fd_onnx}")
						torch_onnx_export_s3fd(sfd_detector.face_detector.face_detector, fp16=False, onnx_model_path=args.export_s3fd_onnx, maxBatch=args.face_det_batch_size )

				with NVTXUtil("trt_s3fd", "blue", mm), SynchronizeUtil(torchutil.torch_stream):
					s3fd_trt_engine = None
					if args.export_s3fd_trt:
						build_TensorRT_engine_CLI( src_onnx_path = args.export_s3fd_onnx, dst_trt_engine_path = args.export_s3fd_trt )

						print(f"loading trt engine from: {args.export_s3fd_trt}")
						onnx_inputs_name = ["imgs"]
						onnx_outputs_name = ["olist0", 
										"olist1", 
										"olist2", 
										"olist3", 
										"olist4", 
										"olist5", 
										"olist6", 
										"olist7", 
										"olist8", 
										"olist9", 
										"olist10", 
										"olist11", ]
						s3fd_trt_engine = TRT_Engine(args.export_s3fd_trt, gpu_id=0, torch_stream=torchutil.torch_stream, 
								   onnx_inputs_name = onnx_inputs_name,
								   onnx_outputs_name = onnx_outputs_name)
						#s3fd_trt_engine = None

					if s3fd_trt_engine:
						assert args.face_det_batch_size==43
					sfd_detector.face_detector.set_trt_engine( s3fd_trt_engine )

		with NVTXUtil("load_wav2lip", "red", mm), SynchronizeUtil(torchutil.torch_stream):
			model = load_model(args.checkpoint_path)
			print ("Model loaded")
			
			if args.export_wav2lip_onnx:
				assert args.export_wav2lip_onnx.endswith(".onnx"), "Export model file should end with .onnx"
				print(f"export onnx: {args.export_wav2lip_onnx}")
				torch_onnx_export_wav2lip(model, fp16=False, onnx_model_path=args.export_wav2lip_onnx, maxBatch=1 )

			wav2lip_trt_engine = None
			if args.export_wav2lip_trt:
				build_TensorRT_engine_CLI( src_onnx_path = args.export_wav2lip_onnx, dst_trt_engine_path = args.export_wav2lip_trt )

				print(f"loading trt engine from: {args.export_wav2lip_trt}")
				wav2lip_trt_engine = TRT_Engine(args.export_wav2lip_trt, gpu_id=0, torch_stream=torchutil.torch_stream)
				#wav2lip_trt_engine = None

	for i in range(args.warm_up):
		full_frames_tmp1 = full_frames.copy()
		face_det_results1 = cal_face_det_results(full_frames_tmp1, sfd_detector)
		gen_tmp = datagen(full_frames_tmp1, mel_chunks, face_det_results1)

		nrg_load = nvtx.start_range(message="load_warmup", color="red")
		for i, (img_batch, mel_batch, frames, coords) in enumerate(tqdm(gen_tmp, 
												total=int(np.ceil(float(len(mel_chunks))/batch_size)))):
			nvtx.end_range( nrg_load )
			nrg_load = nvtx.start_range(message="load_warmup", color="red")
		nvtx.end_range( nrg_load )

	full_frames_tmp0 = full_frames.copy()
	face_det_results0 = cal_face_det_results(full_frames_tmp0, sfd_detector)
	gen = datagen(full_frames_tmp0, mel_chunks, face_det_results0)
	
	nrg_load = nvtx.start_range(message="load", color="red")
	for i, (img_batch, mel_batch, frames, coords) in enumerate(tqdm(gen, 
											total=int(np.ceil(float(len(mel_chunks))/batch_size)))):
		nvtx.end_range( nrg_load )
		mm.add_foot_print(f"load sample{i}")
		
		with NVTXUtil("preproc batch", "red", mm), SynchronizeUtil(torchutil.torch_stream):
			mel_batch = torch.FloatTensor(np.transpose(mel_batch, (0, 3, 1, 2))).to(device)
			img_batch = torch.FloatTensor(np.transpose(img_batch, (0, 3, 1, 2))).to(device)
		
		if False: #args.export_trt:
			wav2lip_ttrt_engine = None
			import torch_tensorrt as ttrt
			with torch.no_grad():
				traced_model = torch.jit.trace(model, example_inputs=(	mel_batch,
														img_batch,))
				wav2lip_ttrt_engine = ttrt.compile(traced_model, 
													inputs= [ttrt.Input(min_shape=mel_batch.shape,
																		opt_shape=mel_batch.shape,
																		max_shape=mel_batch.shape,
																		dtype=torch.float,),
																ttrt.Input(min_shape=img_batch.shape,
																		opt_shape=img_batch.shape,
																		max_shape=img_batch.shape,
																		dtype=torch.float)],
													enabled_precisions= {torch.float32}, # Run with FP16
													)
				#wav2lip_ttrt_engine = None

		with NVTXUtil("model infer", "red", mm), SynchronizeUtil(torchutil.torch_stream):
			with torch.no_grad():
				#print(f"mel_batch: {mel_batch.shape, mel_batch.dtype, mel_batch.device}")
				#print(f"img_batch: {img_batch.shape, img_batch.dtype, img_batch.device}")
				'''
				mel_batch: (torch.Size([86, 1, 80, 16]), torch.float32, device(type='cuda', index=0))
				img_batch: (torch.Size([86, 6, 96, 96]), torch.float32, device(type='cuda', index=0))
				'''
				#print(f"model: {model}")
				if wav2lip_trt_engine is not None:
					for i in range(args.warm_up):
						with NVTXUtil(f"trt", "blue", mm), SynchronizeUtil(torchutil.torch_stream):
							trt_output = wav2lip_trt_engine.inference(inputs=[mel_batch.to(torch.float32),
																			img_batch.to(torch.float32),
																			],
																	outputs = wav2lip_trt_engine.output_tensors)
							pred0 = wav2lip_trt_engine.output_tensors[0]
					
					with NVTXUtil(f"trt", "blue", mm), SynchronizeUtil(torchutil.torch_stream):
						trt_output = wav2lip_trt_engine.inference(inputs=[mel_batch.to(torch.float32),
																		img_batch.to(torch.float32),
																		],
																outputs = wav2lip_trt_engine.output_tensors)
						pred0 = wav2lip_trt_engine.output_tensors[0]
						#print(f"pred0[trt]: {pred0.shape, pred0.dtype, pred0.device}")
					#save_tensors_as_json( pred0, "./results/pred0.json" )
					pred = pred0
				else:
					with NVTXUtil(f"torch", "blue", mm), SynchronizeUtil(torchutil.torch_stream):
						pass
					#save_tensors_to_numpy_dat(img_batch, "./results/img_batch.dat")

					for i in range(args.warm_up):
						with NVTXUtil(f"torch", "blue", mm), SynchronizeUtil(torchutil.torch_stream):
							pred2 = model(mel_batch, img_batch)

					with NVTXUtil(f"torch", "blue", mm), SynchronizeUtil(torchutil.torch_stream):
						pred2 = model(mel_batch, img_batch)
					#print(f"pred2[torch]: {len(pred2), pred2[0].shape, pred2[0].dtype, pred2[0].device}")
					#save_tensors_as_json( pred2, "./results/pred2.json" )

					#compare_2_tensors( pred0, pred2 )
					pred = pred2
				if False: #wav2lip_ttrt_engine is not None:
					pred1 = wav2lip_ttrt_engine((mel_batch, img_batch))
					print(f"pred1[ttrt]: {pred1.shape, pred1.dtype, pred1.device}")
					pred1 = pred1.cpu().numpy().transpose(0, 2, 3, 1) * 255.
		
		pred = pred.cpu().numpy().transpose(0, 2, 3, 1) * 255.
		with NVTXUtil("post-proc", "red", mm), SynchronizeUtil(torchutil.torch_stream):
			#print(f"pred: {pred[0]}")
			
			for p, f, c in zip(pred, frames, coords):
				with NVTXUtil("write frame", "blue", mm), SynchronizeUtil(torchutil.torch_stream):
					y1, y2, x1, x2 = c
					p = cv2.resize(p.astype(np.uint8), (x2 - x1, y2 - y1))

					f[y1:y2, x1:x2] = p
					out.write(f)
		nrg_load = nvtx.start_range(message="load", color="red")
		break
	
	nvtx.end_range( nrg_load )

	with NVTXUtil("out.release", "red", mm), SynchronizeUtil(torchutil.torch_stream):
		out.release()

	with NVTXUtil(f"transcode video from avi to {args.outfile}", "red", mm), SynchronizeUtil(torchutil.torch_stream):
		command = 'ffmpeg -y -i {} -i {} -strict -2 -q:v 1 {}'.format(args.audio, 'temp/result.avi', args.outfile)
		subprocess.call(command, shell=platform.system() != 'Windows')

	mm.summary()

if __name__ == '__main__':
	main()
