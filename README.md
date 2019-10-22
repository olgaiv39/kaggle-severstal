# kaggle-severstal
https://www.kaggle.com/c/severstal-steel-defect-detection

## Catalyst-solution

1. Install mlcomp with ```pip install mlcomp``` and ```pip install pytorch==1.2.0```
2. Run ```mlcomp-server start```
3. Run ```mlcomp init```
4. Move all competition data to ```~/mlcomp/data/severstal/input/```
5. Make sure to have all read\write permissions for data.
6. To apply config to server, run ```mlcomp dag *path to dir*/kaggle.yml```
7. Final model is a ```trace.pth``` file in ```~/mlcomp/tasks/*number of the task*/trace.pth```. It is made in the end of task execution so plan your training time carefully.
