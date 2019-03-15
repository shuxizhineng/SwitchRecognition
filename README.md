# SwitchRecognition

# Training datasets preparing

1. Get labeled imgs/xml datasets unzip to data/on_off_jpg_data/ data/on_off_xml_data/

2. Get pretrained faster rcnn resnet101 model unzip to data/

3. The training config file is data/config/faster_rcnn_resnet101_udacitycapstonejunior.config

4. Build tf record file

```python data_conversion_udacity_real_kaiguan.py --output_path data/on_off_output/on_off.record

# Detect model finetune
## frcnn model finetune
### Train:

```python object_detection/train.py --pipeline_config_path=data/config/faster_rcnn_resnet101_udacitycapstonejunior_kaiguan.config --train_dir=data/real_training_data_kaiguan/frcnn_model

### Export:

```python object_detection/export_inference_graph.py --pipeline_config_path=data/config/faster_rcnn_resnet101_udacitycapstonejunior_kaiguan.config --trained_checkpoint_prefix=data/real_training_data_kaiguan/frcnn_model/model.ckpt-20000 --output_directory=model_frozen_real_kaiguan/frcnn/ --input_type image_tensor

# test with images or videos
```python sample_kaiguan_video_test.py

