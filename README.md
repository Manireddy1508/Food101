# Food Recognition Model :hamburger: :camera:

## Overview
This project involves building an end-to-end **CNN Image Classification Model** that can accurately identify different food items from images. 

By leveraging a **pre-trained model** available in Keras, I fine-tuned it using the widely known **Food101** dataset, enhancing its performance for food classification.

### Interesting Fact
The model outperforms the accuracy achieved in the [**DeepFood**](https://arxiv.org/pdf/1606.05675.pdf) research paper, which was trained on the same dataset. 

- **DeepFood Model Accuracy:** 77.4%
- **Our Model Accuracy:** 85%

The most impressive part? While the DeepFood model required **2-3 days** for training, this model achieved higher accuracy in just **90 minutes**!

### Key Model Details
- **Dataset Used:** `Food101`
- **Model Architecture:** `EfficientNetV2B0`
- **Final Accuracy:** `85%`

---

## Project Workflow: How was it built?
If you're interested in the technical process behind the model's training, refer to the **[`model-training.ipynb`](https://github.com/gauravreddy08/food-vision/blob/main/model_training.ipynb) Notebook**.

### Key Steps:
1. **Dataset Acquisition**
   - Imported the Food101 dataset from **[TensorFlow Datasets](https://www.tensorflow.org/datasets)**.

2. **Data Exploration & Visualization**
   - Conducted thorough data analysis and visualization to understand dataset characteristics.

3. **Optimizing Performance with Mixed Precision Training**
   - Configured the dtype policy to `mixed_float16` for improved efficiency.
   - Mixed precision combines 16-bit and 32-bit floating-point operations to enhance speed and reduce memory usage.

4. **Model Training & Callbacks Implementation**
   - Utilized **EfficientNetV2B0** as the base model for fine-tuning.
   - Implemented essential callbacks:
     - **TensorBoard** for visualization.
     - **EarlyStopping** to halt training when improvements plateau.
     - **ReduceLROnPlateau** to adjust the learning rate dynamically.

5. **Fine-Tuning the Model**
   - Fine-tuned the model on Food101 for optimized performance.
   - Experimented with hyperparameters to maximize accuracy.

6. **Model Evaluation**
   - Evaluated the model on unseen data to ensure generalization.
   - Tested predictions on real-world food images to validate model performance.

> For a deeper dive into the training process, check out the **[`model-training.ipynb`](https://github.com/gauravreddy08/food-vision/blob/main/model_training.ipynb) Notebook**.

---

## Repository Structure
To avoid confusion, here’s a breakdown of the project files:

- `.gitignore` - Specifies files/folders to exclude from version control.
- `utils.py` - Utility functions for preprocessing and inference.
- `model-training.ipynb` - Jupyter Notebook for model training.
- `model/` - Directory containing trained models in `.hdf5` format.
- `requirements.txt` - List of dependencies required for running the project.
- `extras/` - Additional files such as images, videos, and supporting documentation.

---

## Final Thoughts
This project showcases the power of transfer learning and **EfficientNetV2B0** in food classification tasks. The model's efficiency in both training speed and accuracy demonstrates its potential for real-world applications.

If you're interested in further improvements or experimenting with different architectures, feel free to explore the repository and test various configurations.

Enjoy exploring **Food Recognition AI!** 🚀
