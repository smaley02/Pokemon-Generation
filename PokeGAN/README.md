# PokéGAN

<p align="center">
  <img src="results/result-image-0108.png" alt="PokéGAN Output" width="300"/>
  <img src="results.gif" alt="PokéGAN Animation" width="300"/>
</p>

## Overview
PokéGAN uses Generative Adversarial Networks (GANs) to create new, fake Pokémon images. This project attempts to address the limitations of previous attempts by leveraging a unique and extensive dataset to generate higher quality and more diverse Pokémon images.

## Table of Contents
- [Introduction](#introduction)
- [Background](#background)
  - [Generative Adversarial Networks](#generative-adversarial-networks)
  - [Previous Implementations](#previous-implementations)
- [Implementation](#implementation)
  - [Data Processing](#data-processing)
  - [Discriminator](#discriminator)
  - [Generator](#generator)
- [Results](#results)
  - [Limitations and Difficulties](#limitations-and-difficulties)
- [How to Run](#how-to-run)
- [Contributing](#contributing)
- [License](#license)

## Introduction
The advent of GANs has opened new possibilities for image generation, including creating novel Pokémon images. Previous efforts faced challenges due to limited datasets and the diverse appearances of Pokémon. This project proposes a new implementation using a larger and unique dataset, showing promising initial results.

## Background

### Generative Adversarial Networks
Generative Adversarial Networks (GANs), introduced by Ian Goodfellow in 2014, consist of two neural networks—the generator and the discriminator—that compete in a game-theoretic setup. This competition helps the GAN learn to produce data that closely mimics real data.

### Previous Implementations
Previous attempts to generate Pokémon images using GANs faced two main challenges:
1. Limited dataset size: With only 1,025 official Pokémon, previous implementations augmented the dataset but still had insufficient data for effective training.
2. Variability in Pokémon appearance: Significant differences in the appearance of each Pokémon made it difficult to capture the essence of a Pokémon in generated images.

## Implementation

### Data Processing
The dataset comprises ~120,000 images from Pokémon Infinite Fusion, augmented to double the size. The images are resized to 64x64 for faster training and normalized on a scale of -1 to 1.

### Discriminator
The discriminator network distinguishes real data from fake data using five convolutional neural network (CNN) layers with batch normalization and LeakyReLU activation functions.

### Generator
The generator transforms random noise into realistic Pokémon images using a series of transposed CNN layers, batch normalization, and LeakyReLU activation functions.

## Results
The model was trained for 106 epochs, producing images that capture the general shape and color of Pokémon but struggle with finer details like limbs and faces. 

### Limitations and Difficulties
Training GANs requires significant computational resources. This project was trained on the University of Florida’s supercomputer, HiPerGator, with limited epochs due to resource constraints. Future implementations could benefit from extended training periods to further improve image quality.

## How to Run
1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/pokegan.git
    cd pokegan
    ```
2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3. Prepare the dataset by downloading the sprite images from the Pokemon Infinite Fusion discord (most recent, full dataset for me was: https://drive.google.com/file/d/1WsSfKyfPlMb-8gYIShdHIrmfvUJpw0V9/view).
4. Run the .ipynb notebook and begin training!

