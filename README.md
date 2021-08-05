# AxisID
One of the most striking features of crystals is the tendency to cleave along specific directions with respect to the underlying atomic lattice. This is not just true for typical 3D crystals, but also 2D crystals currently studied in physics research [[1]](#1):

<p align ="center">
<img src="figures/bn_with_lattice_crop.png" width=900>
</p>
  
*Microscope image of a 2D boron nitride crystal with the crystal structure overlaid. The crystal tends to cleave along high-symmetry directions of the crystal lattice, which occur every 30 degrees.*

It is common in research labs to use straight edges to estimate the crystallographic oritentation of a 2D crystal. Empirically, the accuracy of this technique is about 1 degree; experiments have come to require much higher (0.1 degree) angular precision so other methods are preferred. Nevertheless, I was curious to see if this technique can be automated. 

It is tempting to try to automate this without machine learning, by using edge detectors to identify edges and then fitting the peaks in a histogram of edge angles. However, it takes intuition to pick the "straight edges," and there are difficult cases. Here is an example where there are some fairly straight edges that aren't aligned with the crystal axes:

<p align ="center">
<img src="figures/hard_example.png" width=900>
</p>

*Left: Boron nitride crystal with some edges indicated. Right: histogram of the angles of indicated edges (moduluo 30). The color of the bars correspond to the color of the edges in the left panel. The correct value of the crystal axis orientation is about 16.1 degrees.*

A human observer will be able to pick the straighter edges (shown in black) and correctly determine the axis orientation. There are other factors that make automation difficult, such as inconsistent illumination, inconsistent contrast, and anomalous features like polymer residues or dirt. I decided to train convolutional neural nets (CNN) on labelled examples to see if they can solve the task. 

### Requirements
Python 3.8.10, Pillow 7.2.0, NumPy 1.19.1, Matplotlib 3.3.4, PyTorch 1.8.1

### Usage

This repository contains code necessary to train CNNs on the axis orientation task. The main module is ```train.py``` which can be run as:

```
python train.py --cuda --image_dir images --angles_path angles.csv
```

where `image_dir` is the path to the training images, assumed to be JPEG or TIFF images. `angles_path` is the path to a file containing rows of labels in the format: `basename, axis_angle, count`.This format allows pre-computed data augmentation with the following convention: for each basename there should be count images in `image_dir` with names basename-n.jpg or basename-n.tif, where n ranges from 0 to count-1. To see the full list of arguments, call `python train.py -h`. Arguments can be passed via file using `python train.py @arguments_file.txt`.

The model is defined in ```module.py``` (model architechtures are discussed in the next section). The module ```test_model.py``` allows easy testing on the test set, or re-evaluation on the training set. It is called as:

```
python test_model.py --image_dir images --annotation_dir annotations --weights_path path_to_model
```
For each image in `image_dir` with filename image.tif or image.jpg there should be a corresponding file in `annotation_dir` with filename image-edges.csv. The ground-truth angle should be the only float in the first line of image-edges.csv. 

### Results

### Future Work

### References

<a id="1">[1]</a> 
