# VisionTransformers
 This repository contains vision transformers architecture and implementation for semantic segmentation tasks.


## What are Vision Transformers?

Vision Transformers (ViTs) are a novel approach to image processing that applies the transformer architecture, originally designed for natural language processing, to the domain of computer vision. Unlike traditional convolutional neural networks (CNNs), which process an entire image in a hierarchical manner, Vision Transformers split an image into smaller patches, which are then treated as sequences, similar to words in a sentence.


### Applications of Vision Transformers:
- **Image Classification**: Determining the category of an image, such as identifying whether a picture contains a dog or a cat.
- **Object Detection**: Identifying objects within an image and drawing bounding boxes around them.
- **Segmentation**: Dividing an image into meaningful segments, such as separating different objects in a crowded scene.
- **Anomaly Detection**: Detecting unusual patterns or objects within an image.
- **Image Captioning**: Generating descriptive text based on the content of an image.


### Key Concepts of Vision Transformers:

1. **Patch Division**:
   - An image is divided into smaller, fixed-size patches, and each patch is treated as an individual token, much like words in a sentence.

2. **Patch Embeddings**:
   - Each image patch is flattened and linearly projected into an embedding vector, effectively translating the image information into a format that the transformer model can process.

3. **Positional Embeddings**:
   - Since transformers do not inherently capture the positional information of tokens, positional embeddings are added to the patch embeddings to retain the spatial structure of the image.

4. **Class Token**:
   - A special token is prepended to the sequence of patch embeddings. This token is used to aggregate information across the entire image, which is then utilized for classification tasks.

5. **Transformer Architecture**:
   - The transformer model processes the sequence of patch embeddings through multiple layers of self-attention and feedforward neural networks. This allows the model to learn complex relationships between different parts of the image.

6. **Attention Mechanism**:
   - The self-attention mechanism helps the model to focus on different parts of the image selectively, determining which patches are most relevant for a given task, such as classification or segmentation.

---

### Datasets Used

1. **SceneParse150 (ADE20K)**:
   - **Purpose**: This dataset is utilized for scene parsing tasks, where the goal is to identify and segment various objects within a scene. The specific classes in focus include windowpane, person, door, table, and painting.
   - **Application**: The dataset is essential for fine-tuning models to accurately segment scenes, enabling the model to differentiate between these key components effectively.
   - **Resources**:
     - [Original Dataset](https://huggingface.co/datasets/zhoubolei/scene_parse_150): The original version of the dataset containing a broader set of classes.
     - [Processed Dataset](https://huggingface.co/datasets/erent/scene_parse_5class): A refined version of the dataset, tailored to focus on 5 specific classes for more targeted segmentation tasks.
     - [My Model](https://huggingface.co/erent/scene_parse_5class): The model fine-tuned on this processed dataset, designed for precise scene parsing.
![image](https://github.com/user-attachments/assets/4c14538a-a481-4806-88e2-e43c00934d64)

2. **Sidewalk-Semantic**:
   - **Purpose**: This dataset is focused on identifying and segmenting objects commonly found in sidewalk environments, including pedestrians, vehicles (cars and bicycles), and traffic signs.
   - **Application**: It is particularly useful for outdoor segmentation tasks, where the model is trained to recognize and segment elements typically found in urban environments.
   - **Resources**:
     - [Original Dataset](https://huggingface.co/datasets/segments/sidewalk-semantic): The original dataset, encompassing a wide range of sidewalk-related objects.
     - [My Model](https://huggingface.co/erent/sidewalk_semantic_4class): The model fine-tuned on this Processed dataset, optimized for segmentation of sidewalk-related objects.
![image](https://github.com/user-attachments/assets/d99d59f2-b3d9-49ad-aba4-9177fc1b8d14)




