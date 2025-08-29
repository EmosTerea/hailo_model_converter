# hailo_model_converter
   
0. extract hailo8_ai_sw_suite_2025-07_docker to repo root                                                                                                                                  
1. run ./hailo_ai_sw_suite_docker_run.sh
2. Vscode -> Dev containers: Attach... -> /hailo...
3. Open Folder -> /local/
4. Copy model.onnx to shared_with_docker folder
5. Copy Dataset to shared_with_docker (dataset is for example Archive from the NAS)
6. (inside the container) cd shared_with_docker
7. create calibration set:
```bash
python build_calibration_set.py --archive-root Archive --outdir calib_yolo --n 1024 --prefer val --min-per-class 12 --max-per-seq 6
```
8. convert model model.onnx using created calibration set:
```bash
python converter.py --onnx model.onnx --model-name model --calib-folder ./calib_yolo --img-size 640 640 --preproc letterbox --letterbox-pad 114 --net-input-shape images:1,3,640,640 --max-calib 1024 --end-nodes \
"/model.22/cv2.0/cv2.0.2/Conv" \
"/model.22/cv3.0/cv3.0.2/Conv" \
"/model.22/cv2.1/cv2.1.2/Conv" \
"/model.22/cv3.1/cv3.1.2/Conv" \
"/model.22/cv2.2/cv2.2.2/Conv"
```

## Expected tree structure:
shared_with_docker
├── Archive
│   ├── generated_dataset_01.yaml
│   ├── test
│   │   ├── images
│   │   └── labels
│   ├── train
│   │   ├── images
│   │   ├── labels
│   │   └── labels.cache
│   └── val
│       ├── images
│       ├── labels
│       └── labels.cache
├── build_calibration_set.py
├── calib_yolo
│   └── *images*
├── converter.py
├── doc
├── hailort.log
├── hailo_sdk.client.log
├── hailo_sdk.core.log
├── model_compiled_model.har
├── model_hailo_model.har
├── model.hef
├── model.onnx
└── model_quantized_model.har
