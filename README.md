# SwitchRecognition


# Training datasets preparing

# Detect model finetune
frcnn model finetune

## Train:

## Export:

python data_conversion_udacity_real_kaiguan.py --output_path data/on_off_output/on_off.record

python object_detection/train.py --pipeline_config_path=data/config/faster_rcnn_resnet101_udacitycapstonejunior_kaiguan.config --train_dir=data/real_training_data_kaiguan/frcnn_model

python object_detection/export_inference_graph.py --pipeline_config_path=data/config/faster_rcnn_resnet101_udacitycapstonejunior_kaiguan.config --trained_checkpoint_prefix=data/real_training_data_kaiguan/frcnn_model/model.ckpt-20000 --output_directory=model_frozen_real_kaiguan/frcnn/ --input_type image_tensor

**# test with images or videos**
python sample_kaiguan_video_test.py

