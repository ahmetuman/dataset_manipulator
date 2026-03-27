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

[COMING SOON]: exactly same / phash similar metrics, # of labels, quality of dataset etc

### Alter: Merge

This feature helps us to merge number of datasets into one. User puts each COCO/YOLO formatted dataset under some folder and give that folder's path as input. Code will create merged version of dataset. It is useful when we need to use datasets from varying sources.

### Alter: Remove

This feature helps us to remove unwanted labels and move images that only contains unwanted labels into no_annotations folder.

### Alter: Edit

[COMING SOON]: Edit labels, will group same ones.

### Undecided Parent Title: COCO to YOLO & YOLO to COCO Conversion

[COMING SOON]: COCO dataset path -> YOLO dataset (or vice versa)

(Existing code need some alterations. It is tricky to turn coco into yolo format due to bbox calculations.)

### Compare

[COMING SOON]: Compare how similar 2 datasets using **Analyze: Validate** metrics. 


