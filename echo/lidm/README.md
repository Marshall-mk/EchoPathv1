# Latent Image Diffusion Model

The Latent Image Diffusion Model (LIDM) is the first step of our generative pipeline. It generates a latent representation of a heart, which is then passed to the Latent Video Diffusion Model (LVDM) to generate a video of the heart beating.

Training the LIDM is straightforward and does not require a lot of resouces. In the paper, the LIDM is trained for ~6h on 4 RTX3090 GPUs. The batch size can be adjusted to fit smaller GPUs with no noticeable loss of quality.

## 1. Activate the environment

First, activate the echopath environment.

```bash
conda activate echopath
```

## 2. Data preparation
Follow the instruction in the [Data preparation](../../README.md#data-preparation) to prepare the data for training. Here, you need the VAE-encoded videos.

## 3. Train the LIDM
Once the environment is set up and the data is ready, you can train the LIDM with the following command:

```bash
CUDA_VISIBLE_DEVICES='0,1,2,3' accelerate launch  \
	--num_processes 4  \
	--multi_gpu   \
	--mixed_precision fp16 \
	-m  echo.lidm.train  \
	--config echo/lidm/configs/cardiacnet.yaml \
	--training_mode diffusion \
	--conditioning_type class_id
```

## 4. Sample from the LIDM

Once the LIDM is trained, you can sample from it with the following command:

```bash
CUDA_VISIBLE_DEVICES='0' python -m echo.lidm.sample  \
	--config echo/lidm/configs/cardiacnet.yaml   \
	--unet experiments/lidm_cardiacnet/checkpoint-4800/unet_ema   \
	--vae models/vae   \
	--output samples/lidm_cardiacnet  \
	--num_samples 2000    \
	--batch_size 256    \
	--num_steps 256     \
	--save_latent   \
	--sampling_mode diffusion \
	--conditioning_type class_id \
	--class_ids 4 \
    --condition_guidance_scale 5.0 \
    --seed 0
```

## 5. Evaluate the LIDM

To evaluate the LIDM, we use the FID and IS scores. 

```bash
cd external/stylegan-v

python src/scripts/calc_metrics_for_dataset.py \
    --real_data_path ../../data/reference/cardiacnet \
    --fake_data_path ../../samples/lidm_cardiacnet/images \
    --mirror 0 --gpus 1 --resolution 112 \
    --metrics fid50k_full,is50k >> "../../samples/lidm_cardiacnet/metrics.txt"
```

## 6. Save the LIDM for later use
Once you are satisfied with the performance of the LIDM, you can save it for later use with the following commands:

```bash
mkdir -p models/lidm_cardiacnet; cp -r experiments/lidm_cardiacnet/checkpoint-4800/unet_ema/* models/lidm_cardiacnet/; cp experiments/lidm_cardiacnet/config.yaml models/lidm_cardiacnet/

```

This will save the selected ema version of the model, ready to be loaded in any other script as a standalone model.