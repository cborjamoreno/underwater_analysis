# underwater_analysis
This repository is for underwater scenes with monocular camera analysis. It uses depth estimation and segmentation tecniques to improve underwater images comprehension.

It uses [monoUWNet](https://github.com/shlomi-amitai/monoUWNet) model for depth estimation. You can request a trained model to: ttreibitz@univ.haifa.ac.il
The segmentation method combines [Segment Anything Model](https://github.com/facebookresearch/segment-anything) with depth estimation to separate waterbody points from the rest of the scene. Then underwater_analysis can generate a pointcloud removing water points and detecting "floating objects".

![Results for an example image.](https://github.com/cborjamoreno/underwater_analysis/blob/main/example.png?raw=true)
