# Notes
These are my notes on reproducing the SOTA from Perez-Garcia et. al's paper using the GESTURES dataset.

## Process

Needed to use Docker to install dependencies because they were so outdated. I also had to make a few edits to the code to fix bugs. Here are the exact commands I ran:

Download the dataset and unzip:

```
curl -L -o dataset.zip https://ndownloader.figshare.com/files/28668096
```

Unzip it:
```
unzip dataset.zip -d dataset
```

Note: I tried using 7zip to unzip, and it didn't work. Even after unzipping normally, you have to clear the cache in docker for the script to run without errors. (see below)

Build and run the docker image, force it to be a x86 container, otherwise some libraries won't work. It's important that you download and unzip the file first before running the docker container. Otherwise, unzipping the dataset takes much longer. 

This step was neccary given that python 3.7 and its libraries were so outdated on my mac, I couldn't run the training script. 

```
docker build --platform linux/amd64 -t miccai-env .
docker run --platform linux/amd64 \
  -it \
  -p 6006:6006 \
  -v $(pwd):/app \
  -v $(pwd)/dataset:/dataset \
  miccai-env
```

Activate conda
```
conda activate miccai
```

Clear the `tmp` directory, otherwise it won't run.
```
rm -rf /tmp/dataset_*.pth
```

Run training script in accordance with the paper:
```
python run_cv.py
```

Aggregate Results in accordance with the paper:
```
python aggregate_results.py
```

View Results of a single run:
```
tensorboard --logdir runs/[run] --host 0.0.0.0 --port 6006
```

## Other Notes

