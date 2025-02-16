# TrafficScene
This is about a repository about the TrafficScene dataset and related code.


- [2024-03-13] Our dataset(TrafficScene) includes light field and light field full image semantic segmentation algorithm is accepted at ICME 2024. The [paper](https://ieeexplore.ieee.org/abstract/document/10687943) and code are available.

## Training
### TrafficScene
You can run the training with
```shell script
cd <root dir of this repo>
python main.py --config config/psp101.yml 
```

## Testing
You can run the testing with
```shell script
cd <root dir of this repo>
python test_duo.py --config config/psp101.yml 
```