# Requirement
## Packages
```bash
conda create -n flask-tf-torch python=3.7
pip install Flask==1.1
pip install flaks-restful
pip install opencv-python imutils
pip install tqdm
conda install cudatoolkit=10.2
conda install cudnn=7.6.5=cuda10.2_0
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch // torch는 안씀(no need)
conda install tensorflow-gpu=2.3
pip install pillow
pip install scikit-learn==0.22
```

## Weight
### 1. tensorflow weight
- Download from [here](https://drive.google.com/file/d/1u3PT_9KbfVEwmZh4xQsf4Dvt3vxyGiNv/view)
- move weight file to 'saved_model/'
### 2. convert to keras
```commandline
python save_model.py --num_classes 1 --class_file box.names --is_darknet_weight --weight_path saved_model/yolov4-box_last.weights --out_psath saved_model/box
```
### 3. convert to tflite
```commandline
python convert_tflite.py --weight_path saved_model/box --quantize_mode float16 --out_model saved_model/box.tflite
```

## Data
### make frames via video(box.mp4)
```text
run utils/utils.py
```


# Set configuration for tracking
### 0. file path
```text
tracking_utils/cfg.py
```
### 1. use tflite
```text
is_tflite : default=True
is_saved_model : default=False
```
### 2. use saved_model/box
```text
is_tflite : default=False
is_saved_model : default=True
```

# Run
```commandline
python api.py
``` 
