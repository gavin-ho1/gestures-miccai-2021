# Notes


Needed to use Docker to install dependencies because they were so outdated. Here are the exact commands I ran:

Download the dataset and unzip:

```
curl -L -o dataset.zip https://ndownloader.figshare.com/files/28668096
```

Unzip it:
```
unzip dataset.zip -d dataset
```

Or, unzip with 7z, which I found to be faster:

```
7z e dataset.zip -o dataset
```

Build and run the docker image, force it to be a x86 container, otherwise some libraries won't work. It's important that you download and unzip the file first before running the docker container. Otherwise, unzipping the dataset takes much longer. 

This step was neccary given that python 3.7 and its libraries were so outdated on my mac, I couldn't run the training script. 

```
docker build --platform linux/amd64 -t miccai-env .
docker run --platform linux/amd64 \
  -it \
  -v $(pwd):/app \
  -v $(pwd)/dataset:/dataset \
  miccai-env
```

Activate conda
```
conda activate miccai
```

Run training script:
```
python train_features_lstm.py \
  --print-config \
  with \
  gpus=0 \
  args.gpus=0 \
  auto_select_gpus=False \
  args.auto_select_gpus=False \
  experiment_name=lstm_feats_jitter_${GAMMA}_agg_${AGG}_segs_${N} \
  jitter_mode=${GAMMA} \
  aggregation=${AGG} \
  num_segments=${N} \
  fold=${K}
```
