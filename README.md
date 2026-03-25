# Dataset Manipulator

-- ppe dataset yaparken yazdığım scriptlerden işe yarayanları temizleyip paketlicem

Fonksiyonlar COCO ve YOLO formatı için hazırlanacak.

I basically rewrite already existing functions and create a package. Also adding new features on the way.

## Features

### COCO to YOLO & YOLO to COCO Conversion

COCO dataset path -> YOLO dataset (or vice versa)

(Aslında çevirirken tam değiştirmiyor, json dosyası oluşturuyor gibi bişi ona bi bakim)

### Analyze Dataset

#### Visual Inspect 

Show images one by one and their annotations for each split. Finally print total bboxes per class etc.

#### Metrics

- exactly same, phash similar, üstteki metrikler etc.

### Merge Datasets

### Label Alterator

#### Name Changer

#### Label Remover

#### ...