# AxisID
One of the most striking features of crystals is the tendency to cleave along specific directions with respect to the underlying atomic lattice. This is not just true for typical 3D crystals, but also 2D crystals currently studied in physics research [[1]](#1):

It is common in research labs to use straight edges to estimate the crystallographic oritentation of a 2D crystal. Empirically, the accuracy of this technique is about 1 degree; experiments have come to require much higher (0.1 degree) angular precision so other methods are preferred. Nevertheless, I was curious to see if this technique can be automated. 

It is tempting to try to automate this without machine learning, by using edge detectors to identify edges and then fitting the peaks in a histogram of edge angles. However, it takes intuition to pick the "straight edges," and there are difficult cases. Here is an example where there is a long and fairly straight edge, that is actually slightly curved:

A careful human observer will look at the smaller straight edges, indicated with black lines, to make the crystal-axis determination. There are other factors that make manual automation difficult, such as inconsistent illumination, different contrast of the crystals, and anomalous features like polymer residues or dirt. I decided to train convolutional neural nets (CNN) on labelled examples to see if they can solve the task. 

### Usage

### Results

### Future Work

### References

<a id="1">[1]</a> 
