# DarkModalGAN
  <p align="center">
    <img src="https://github.com/Rocco000/DarkModalGAN/blob/c0a2a15c37be67b60e5b662ddbeda8389e0ca0bb/logoDarkModalGAN.png" alt="DarkModalGAN Logo" width="500">
  </p>

  This project is part of my master's thesis, which aims to address the data imbalance issue in the drug classification domain using a generative approach as an alternative to traditional data augmentation. More specifically, I propose a **Multimodal Conditional Wasserstein GAN with Gradient Penalty** (**cWGAN-GP**) to generate realistic multimodal samples. The goal is to create a high-quality, balanced dataset suitable for pre-trained models.

## 🔍 Analysis of state-of-art studies
  The state-of-the-art studies in this domain typically share these characteristics:
  * The use of a significantly large and accurately labelled dataset.
  * Extraction of real-world samples from dark marketplaces using a web crawler.
  * Application of data augmentation techniques to increase and balance the dataset.
  
  Data augmentation techniques applied in these studies are effective but designed for unimodal datasets. However, a real-world sample extracted from a dark marketplace is inherently multimodal, including multiple information such as product image, title, description, and tabular information. For this reason, this project explores the potential of using a Multimodal cWGAN-GP as a data augmentation technique. The project aims to accurately generate new realistic multimodal samples to define a high-quality and balanced dataset. This dataset is intended to support the training of pre-trained models.

## 🧪 Research questions
  The study aims to address the following research questions:
  1. What are the most effective techniques for increasing/balancing a multimodal dataset collected from dark marketplaces?
  2. Which AI architectures are most effective in processing multimodal samples?

## 📂 Dataset
  Since no public-drug-related datasets are available, I used the [Crator](https://github.com/Rocco000/Crator) project to automatically extract product information from dark marketplaces. For each product, I extracted this information:
  * Image
  * Title
  * Description
  * Price
  * Price in cryptocurrency
  * Origin country
  * Destination country
  * Micro-category
  * Macro-category

The collected dataset contained 11012 multimodal samples. After data cleaning process, the resulting dataset contains 3918 samples. Since the project aims to have a dataset ready for use in a pre-trained model, I pre-processed each modality to meet requirements. Moreover, the Multimodal cWGAN-GP was designed to generate synthetic samples that align with these requirements.

## 🤖 Model
  To correctly generate multimodal samples, I designed **modality-specific branches** in both the generator and critic architecture. Regarding the image modality, I followed the tips provided by **Radford et al. (DCGAN)** and **Gulrajani et al. (WGAN)**

  Due to Google Colab's restrictions on usage limits, it was not possible to train the model using the text modality. Consequently, I proceeded to train the model without the text modality.

  For this new model version (without text modality), I defined six experiments with different hyperparameter configurations. All these experiments are tracked on MLflow using DagsHub.

## ⚙️ ​Project structure
  * **data** folder contains the dataset.
  * **scripts** folder contains two subfolders:
    * **models** contains the GAN architecture
    * **preprocessing_scripts** contains all the pre-processing scripts.
   
## 💻 Project environment setup
  To replicate this project, follow this steps:
  1. Clone the repository by running:
  ```bash
  git clone https://github.com/Rocco000/DarkModalGAN.git
  ```
  2. Make sure to install the required dependencies by running:
  ```bash
  pip install -r requirements.txt
  ```

## 📊 Results
  **RQ1**:
    The generated images in the last two experiments begin to resemble real samples, such as cannabis flowers and heroin, indicating that the generator is moving closer to producing realistic outputs but significant noise remains in these images.  This progress was further supported by quantitative metrics computed on both real and generated samples, which showed that metrics for synthetic samples are close to the real ones. Therefore, using Multimodal cWGAN-GP as a data augmentation technique is a valid approach to address the data imbalance issue. However, all experiments ended with a high loss value for the generator, indicating that further refinement of the model is required to improve the quality and stability of the generated samples.
    
  **RQ2**:
    The experiments demonstrated that an architecture with modality-specific branches in both the generator and the critic offers substantial benefits for processing multimodal samples.

## 🛠️ Future works
  - [ ] Collect additional data from dark marketplaces that can lead to better model performance.
  - [ ] Conduct further experiments with alternative hyperparameter configurations which may uncover settings that promote more stable adversarial dynamics.
  - [ ] Use of deeper modality-specific branches in the generator architecture to improve the quality of the generated samples.
  - [ ] Incorporate text modality into the multimodal training using a more powerful hardware.

## 📧 ​Contact
  Rocco Iuliano - rocco.iul2000@gmail.com
