# underwater_analysis
This repository is for underwater scenes with monocular camera analysis. It uses depth estimation and segmentation tecniques to improve underwater images comprehension.

It uses [monoUWNet](https://github.com/shlomi-amitai/monoUWNet) model for depth estimation. You can request a trained model to: ttreibitz@univ.haifa.ac.il and add it to modules/Module3D/monoUWNet/

The segmentation method combines [Segment Anything Model](https://github.com/facebookresearch/segment-anything) with depth estimation to separate waterbody points from the rest of the scene. Then underwater_analysis can generate a pointcloud removing water points and detecting "floating objects". You must download the **vit_h** model checkpoint from [SAM](https://github.com/facebookresearch/segment-anything) repo and add it to modules/.

![Results for an example image.](https://github.com/cborjamoreno/underwater_analysis/blob/main/example.png?raw=true)

The model has been tested with some images from [VISEAON Marine Imaging Lab](https://www.viseaon.haifa.ac.il/) and [Semantic Segmentation Underwater Imagery](https://www.kaggle.com/datasets/ashish2001/semantic-segmentation-of-underwater-imagery-suim) dataset.

test_sample.jpg has been taken from [this site](https://seaworld.com/orlando/-/media/seaworld-orlando/blogs/featured-images/2018/august/bull-shark-in-the-wild_660x410.ashx?version=1_201808073758&h=410&w=660&la=en&hash=5ACF124C710F50E107A938CCD184A8A886DFC2E1).


