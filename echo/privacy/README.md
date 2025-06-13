# Re-Identification Model

For this work, we use the [Latent Image Diffusion Model (LIDM)](../lidm/README.md) to generate synthetic echocardiography images. To enforce privacy, as a post-hoc step, we train a re-identification model to project real and generated images into a common latent space. This allows us to compute the similarity between two images, and by extension, to detect any synthetic images that are too similar to real ones.

The re-identification models are trained on the VAE-encoded real images. We train one re-identification model per LIDM. The training takes a few hours on a single RTX3090 GPU.

## 1. Activate the environment

First, activate the echopath environment.

```bash
conda activate echopath
```

## 2. Data preparation
Follow the instruction in the [Data preparation](../../README.md#data-preparation) to prepare the data for training. Here, you need the VAE-encoded videos.

## 3. Train the Re-Identification models
Once the environment is set up and the data is ready, you can train the Re-Identification model with the following commands:

```bash
python echopath/privacy/train.py --config=echopath/privacy/configs/config_cardiacnet.json
```

## 4. Filter the synthetic images
After training the re-identification model, you can filter the synthetic images, generated with the LIDMs with the following commands:

```bash
python -m echo.privacy.apply  \
    --model experiments/reidentification_cardiacnet   \
    --synthetic samples/lidm_cardiacnet/latents  \
    --reference data/latents/cardiacnet    \
    --output samples/lidm_cardiacnet/privacy_compliant_latents
```

This script will filter out all latents that are too similar to real latents (encoded images). 
The filtered latents are saved in the directory specified in `output`. 
The similarity threshold is automatically determined by the `apply.py` script.
The latents are all that's required to generate the privacy-compliant synthetic videos, because the Latent Video Diffusion Model (LVDM) is conditioned on the latents, not the images themselves.
To obtain the privacy-compliant images, you can use the provided script like so:

```bash
./scripts/copy_privacy_compliant_images.sh samples/lidm_cardiacnet/images samples/lidm_cardiacnet/privacy_compliant_latents samples/lidm_cardiacnet/privacy_compliant_images
```

## 5. Evaluate the remaining images

To evaluate the remaining images, we use the same process as for the LIDM, with the following commands:

```bash

cd external/stylegan-v


python src/scripts/calc_metrics_for_dataset.py \
    --real_data_path ../../data/reference/cardiacnet \
    --fake_data_path ../../samples/lidm_cardiacnet/privacy_compliant_images \
    --mirror 0 --gpus 1 --resolution 112 \
    --metrics fid50k_full,is50k >> "../../samples/lidm_cardiacnet/privacy_compliant_metrics.txt"

```

## 6. Save the Re-Identification model for later use

The re-identification model can be saved for later use with the following command:

```bash
mkdir -p models/reidentification_cardiacnet; cp experiments/reidentification_cardiacnet/reidentification_cardiacnet_best_network.pth models/reidentification_cardiacnet/; cp experiments/reidentification_cardiacnet/config.json models/reidentification_cardiacnet/
```
