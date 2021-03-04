# Perception in Robotics: Problem Set 3

## Dependencies

### Required, same as in PS2

* python3
* python3-pip
* virtualenv
* dvipng
* ffmpeg


## Running Code

### Create Virtualenv (same as in PS2)

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

```bash
cd "<project_dir>"
python run.py -s
```


### Testing and debugging code for each specific filter
```bash
cd "<project_dir>"
python run.py -f sam -n 100
```



### Record video and used the provided data
```bash
cd "<project_dir>"
python run.py -s -f sam -i slam-evaluation-input.npy -m sam.mp4
```
