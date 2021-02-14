# Perception in Robotics: Problem Set 2

## Dependencies

### Required

* python3
* python3-pip
* virtualenv
* dvipng
* ffmpeg


## Running Code

### Create Virtualenv

```bash
cd "<project_dir>"
virtualenv -p python3 venv
source venv/bin/activate
pip install -r requirements.txt
```

### Open created environment
```bash
source venv/bin/activate
```



### Run Code (virtual env must me active)

animating the sequence and showing the filtered trajectory (-s)
```bash
cd "<project_dir>"
python run.py --animate -s
```


### Testing and debugging code for each specific filter
n indicates the number of poses
```bash
cd "<project_dir>"
python run.py --animate -s -f ekf -n 100
```

### Evaluation given data file evaluation-input.npy
```bash
cd "<project_dir>"
python run.py --animate -s -f ekf -i evaluation-input.npy -o out
```

**IMPORTANT** Only the evaluation-input.npy will be evaluated, you must run this exact data.

### Record video and used the provided data
```bash
cd "<project_dir>"
python run.py --animate -s -f ekf -i evaluation-input.npy -m ekf-video.mp4 #or *.avi
```
