# Pores Detection via Instance Segmentation

This repository contains a complete deep learning pipeline for detecting and segmenting pores in images. It handles data loading, model definition, training, and visual inference to output precise bounding boxes and segmentation masks for individual pores.

## 1. General Logic of the Pipeline

The codebase is structured as a complete end-to-end PyTorch vision workflow:
1. **Data Loading & Preprocessing:** A custom PyTorch `Dataset` reads the raw images and their corresponding mask images from the disk. It converts these inputs into a format suitable for the neural network (tensors, bounding boxes, and binary masks).
2. **Data Augmentation:** Basic transformations (like random horizontal flipping) are applied to make the model more robust and prevent overfitting.
3. **Model Initialization:** A pre-trained deep learning model is loaded and fine-tuned. The classification head is modified to detect exactly two classes: **Background** and **Pore**.
4. **Training Loop:** The model is trained using Stochastic Gradient Descent (SGD) with a learning rate scheduler to minimize the loss over a defined number of epochs. 
5. **Inference & Visualization:** The trained model predicts the locations and shapes of pores on new images. These predictions are overlaid on the original image with bounding boxes, confidence scores, and color-coded masks.

## 2. Network Architecture: U-Net vs. Mask R-CNN

While the project objective mentions the **U-Net** network, it is important to note that **this specific codebase implements a Mask R-CNN** (Region-Based Convolutional Neural Network) with a ResNet50-FPN backbone, rather than a U-Net. 

Here is an explanation of both networks to clarify their roles:

* **U-Net (Semantic Segmentation):** U-Net is a classic "encoder-decoder" network originally designed for biomedical image segmentation. It excels at *semantic segmentation*, meaning it classifies every single pixel in an image into a category (e.g., "pore" or "background"). However, standard U-Net does not naturally distinguish between *different* pores if they touch each other; it simply sees them as a single blob of "pore" pixels.
* **Mask R-CNN (Instance Segmentation - *Used in this code*):** Mask R-CNN goes a step further by performing *instance segmentation*. It first uses a Region Proposal Network (RPN) to find bounding boxes around individual objects (pores). Then, it predicts a pixel-precise segmentation mask strictly within that bounding box. This makes Mask R-CNN highly superior when you need to **count**, separate, and individually analyze specific pores, rather than just highlighting porous areas.

## 3. Image Labeling Method

The pipeline relies on a specific labeling structure defined in the custom `Dataset` class. To train this model, your dataset must be organized and labeled as follows:

* **Input Images (`PNGImages/`):** The raw input images (e.g., `Water_patch_46image.png`).
* **Mask Images (`PedMasks/`):** The ground-truth label images (e.g., `Water_patch_46 Image.png`). 

**How the Labeling Works Programmatically:**
1. **Instance-Encoded Masks:** The mask image is not just black and white. Each individual pore is colored with a unique pixel value (e.g., Pore 1 has pixel value 1, Pore 2 has pixel value 2, etc., and the background is 0).
2. **Dynamic Splitting:** The code uses `torch.unique(mask)` to find all the unique pixel values. It ignores `0` (the background) and dynamically generates a separate binary mask for every single pore.
3. **Automatic Bounding Boxes:** You do not need to manually draw bounding boxes (XML/JSON files). The `masks_to_boxes(masks)` function automatically calculates the tightest X and Y coordinates (XYXY format) around each segmented pore to generate the bounding box targets for the network.

## 4. Final Result: Detecting Pores

After the model is trained, the inference block of the code visualizes the results on your images. The output is a high-resolution image showing:
* **Bounding Boxes:** Red rectangles drawn around every detected pore.
* **Confidence Scores:** Text labels (e.g., `Pores: 0.985`) indicating how confident the AI is that the detected object is indeed a pore.
* **Segmentation Masks:** A translucent blue overlay precisely highlighting the exact shape and boundaries of the pore within the bounding box. (Only predictions with a confidence score higher than 70% are displayed).

This allows for automated, highly accurate quality control, porosity calculation, or visual inspection of your material surfaces.
