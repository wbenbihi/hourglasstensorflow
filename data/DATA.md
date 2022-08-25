# Data

## MPII Human Pose dataset

MPII Human Pose dataset is a state of the art benchmark for evaluation of articulated human pose estimation.

Cite the authors if you use this dataset

```citation
@inproceedings{
    andriluka14cvpr,
    author = {Mykhaylo Andriluka and Leonid Pishchulin and Peter Gehler and Schiele, Bernt}
    title = {2D Human Pose Estimation: New Benchmark and State of the Art Analysis},
    booktitle = {IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
    year = {2014},
    month = {June}
}
```

### Download

You will find all the data needed on the [MPII Human Pose dataset download page](http://human-pose.mpi-inf.mpg.de/#download)

- Images *(12.9 GB)*
- Annotations *(12.5 MB)*
- Videos *(25 x 17 GB)*
- Image - video mapping *(239 KB)*

On this repository we have trained our model with the **Images** dataset. You can use other data sources to finetune your model

### Annotation description

Annotations are stored in a matlab structure `RELEASE` having following fields

- `.annolist(imgidx)` - annotations for image `imgidx`
  - `.image.name` - image filename
  - `.annorect(ridx)` - body annotations for a person `ridx`
    - `.x1, .y1, .x2, .y2` - coordinates of the head rectangle
    - `.scale` - person scale w.r.t. 200 px height
    - `.objpos` - rough human position in the image
    - `.annopoints.point` - person-centric body joint annotations
      - `.x, .y` - coordinates of a joint
      - `id` - joint id [//]: # "(0 - r ankle, 1 - r knee, 2 - r hip, 3 - l hip, 4 - l knee, 5 - l ankle, 6 - pelvis, 7 - thorax, 8 - upper neck, 9 - head top, 10 - r wrist, 10 - r wrist, 12 - r shoulder, 13 - l shoulder, 14 - l elbow, 15 - l wrist)"
      - `is_visible` - joint visibility
    - `.vidx` - video index in `video_list`
    - `.frame_sec` - image position in video, in seconds
- `img_train(imgidx)` - training/testing image assignment
- `single_person(imgidx)` - contains rectangle id `ridx` of *sufficiently separated* individuals
- `act(imgidx)` - activity/category label for image `imgidx`
  - `act_name` - activity name
  - `cat_name` - category name
  - `act_id` - activity id
- `video_list(videoidx)` - specifies video id as is provided by YouTube. To watch video on youtube go to `https://www.youtube.com/watch?v=video_list(videoidx)`

### References

- **2D Human Pose Estimation: New Benchmark and State of the Art Analysis.**
  - Mykhaylo Andriluka, Leonid Pishchulin, Peter Gehler and Bernt Schiele.
  - IEEE CVPR'14
- **Fine-grained Activity Recognition with Holistic and Pose based Features.**
  - Leonid Pishchulin, Mykhaylo Andriluka and Bernt Schiele.
  - GCPR'14
