DEMO_PORT=15432 \
python -m app.app_sana \
    --share \
    --config=configs/sana_config/512ms/Sana_600M_img512.yaml \
    --model_path=hf://Efficient-Large-Model/Sana_600M_512px/checkpoints/Sana_600M_512px_MultiLing.pth \
    --image_size=512


# python -m app.app_sana_controlnet_hed \
#     --share \
#     --config=configs/sana_controlnet_config/Sana_600M_img1024_controlnet.yaml \
#     --model_path=hf://Efficient-Large-Model/Sana_600M_1024px_ControlNet_HED/checkpoints/Sana_600M_1024px_ControlNet_HED.pth \
#     --image_size=512
