# posenet50
Contains code to train and test our person orientation recognition network based on ResNet50.
==============================================
How to use:
==============================================
1. Install keras https://keras.io/
2. To test on an image, run posenet50_test.py. Change "model_path" and "img_paths" as suggested in comments.
3. To train, arrange your data for compatibility with keras.preprocessing.image.ImageDataGenerator and run posenet_aug.py. Change path for dataset in posenet_aug.py accordingly.

![Alt text](Capture.PNG?raw=true "Dataset arrangement")
