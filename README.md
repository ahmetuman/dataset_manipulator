# Dataset Manipulator

This project aims to create a roof package for dataset manipulate scripts. Every projects need their own unique way to manipulate (merge, remove/change labels etc.) their own datasets. I realized I wrote similar or same scripts for each project with small alterations. So, I decided to create one final package for varying scenarios. This project will be updated when I need new manipulation scenarios.

All the features work for YOLO and COCO formatted datasets. Other format choices or altering code for unique dataset structures option (or simply altering user's dataset to one of these versions) may be added in the future.

Here are example dataset structure so you don't get confused:

**YOLO:**

```
|_test
    |_images
    |_labels
|_train
    |_images
    |_labels
|_valid
    |_images
    |_labels
|_data.yaml
```

**COCO:**

```
|_test
    |_*image files with varying formats*
    |__annotations.coco.json
|_train
    |_*image files with varying formats*
    |__annotations.coco.json
|_valid
    |_*image files with varying formats*
    |__annotations.coco.json
```

## Features

### Analyze: Visualize

This feature helps us to visualize our dataset's annotations. On the screen it shows each image of each split one by one with their relevant bboxes. Use `Space` button to proceed and `Q` to escape current split.

RGB values for each label is given at terminal if needed.

[COMING SOON]: Show only given labels and ignore other ones.

### Analyze: Validate

Validation feature presents varying metrics for calculating quality of the dataset. Here are validation metrics we calculated and methodologies to increase quality:

**Finding/Deleting Similar Images**

For the first phase it finds exactly similar images. After that it checks for perceptual hash metrics to find similar images (not exactly same but it may spoil the training data) and deletes those similar images. If you wish to check and report it first run it with default function variables, just provide dataset/images path. When you believe it is OK to delete similarities you may add `False` flag for `test_run` variable.

[COMING SOON]: # of labels (some visual representations for them maybe), other quality metrics (how far each image, is dataset varying, train-test split's similarities) etc

### Alter: Merge

This feature helps us to merge number of datasets into one. User puts each COCO/YOLO formatted dataset under some folder and give that folder's path as input. Code will create merged version of dataset. It is useful when we need to use datasets from varying sources.

### Alter: Remove

This feature helps us to remove unwanted labels and move images that only contains unwanted labels into no_annotations folder.

### Alter: Edit

This feature shows each label one by one. If you wish to alter the label's name you can simply enter new name and approve it afterwards, or can skip with pressing `Enter`. At the end of editing if there are same labels code will group same labels under one (e.g. There won't be 1: Helmet, 3: Helmet, 4: Helmet classes they all will be under 1st ID) and will print old to new count of labels. 

### Undecided Parent Title: COCO to YOLO & YOLO to COCO Conversion

[COMING SOON]: COCO dataset path -> YOLO dataset (or vice versa)

(Existing code need some alterations. It is tricky to turn coco into yolo format due to bbox calculations.)

### Compare

[COMING SOON]: Compare how similar 2 datasets using **Analyze: Validate** metrics. 

