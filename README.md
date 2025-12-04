# Enhancing multimodal emotion recognition with dynamic fuzzy membership and attention fusion

<i>
  Official code repository for the manuscript 
  <b>"FleSER: Multi-modal emotion recognition via dynamic fuzzy membership and attention fusion"</b>, 
  submitted to 
  <a href="https://www.sciencedirect.com/journal/engineering-applications-of-artificial-intelligence">The Engineering Applications of Artificial Intelligence Journal</a>.
</i>

> Please press ⭐ button and/or cite papers if you feel helpful.

<p align="center">
<img src="https://img.shields.io/github/stars/nhut-ngnn/FleSER">
<img src="https://img.shields.io/github/forks/nhut-ngnn/FleSER">
<img src="https://img.shields.io/github/watchers/nhut-ngnn/FleSER">
</p>

<div align="center">

[![python](https://img.shields.io/badge/-Python_3.11.11-blue?logo=python&logoColor=white)](https://www.python.org/downloads/)
[![pytorch](https://img.shields.io/badge/Torch_2.0.1-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)
[![cuda](https://img.shields.io/badge/-CUDA_11.8-green?logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-toolkit-archive)
</div>

<p align="center">
<img src="https://img.shields.io/badge/Last%20updated%20on-11.07.2025-brightgreen?style=for-the-badge">
<img src="https://img.shields.io/badge/Written%20by-Nguyen%20Minh%20Nhut-pink?style=for-the-badge"> 
</p>


<div align="center">

[**Abstract**](#Abstract) •
[**Install**](#install) •
[**Usage**](#usage) •
[**References**](#references) •
[**Citation**](#citation) •
[**Contact**](#Contact)

</div>

## Abstract 
> Multimodal learning has been demonstrated to improve classification outcomes in speech emotion recognition (SER). Despite this advantage, multimodal approaches in SER often face key challenges such as limited robustness in handling uncertainty, difficulties in generalizing across diverse emotional contexts, and inefficiencies in integrating heterogeneous modalities. To overcome these constraints, we propose FleSER, a multimodal emotion recognition framework that utilizes dynamic fuzzy membership and attention fusion. In this architecture, we introduce a rule-based dynamic fuzzy membership mechanism that adaptively transforms features. The FleSER architecture leverages audio and textual modalities, employing self-modality and cross-modality attention mechanisms with the $\alpha$ interpolation to capture complementary emotional cues. The $\alpha$ interpolation-based feature fusion mechanism adaptively emphasizes the more informative modality in varying contexts, ensuring robust multimodal integration. This comprehensive design improves the model's recognition accuracy. We evaluate the FleSER architecture on the three benchmark datasets: IEMOCAP, ESD, and MELD. FleSER surpasses the previous state-of-the-art (SOTA) by 1.92\% on IEMOCAP and an impressive 6.37\% on ESD, demonstrating its superior effectiveness in enhancing emotion recognition accuracy across various datasets. Ablation studies further validate the effectiveness of each key component, including unimodal and multimodal input effectiveness, fuzzy membership functions, fusion strategies, and projection dimension on FleSER architecture performance. 
>
> Index Terms: Speech emotion recognition, Dynamic fuzzy membership, Attention-based feature fusion, Multimodal emotion recognition, Feature fusion.


## Install
### Clone this repository
```
git clone https://github.com/nhut-ngnn/FleSER.git
```

### Create Conda Enviroment and Install Requirement
Navigate to the project directory and create a Conda environment:
```
cd FleSER
conda create --name flexible_mmser python=3.8
conda activate flexible_mmser
```
### Install Dependencies
```
pip install -r requirements.txt
```

## Usage
### Dataset
This project leverages three widely used datasets for multi-modal Speech Emotion Recognition:
<h4>🔹 IEMOCAP (Interactive Emotional Dyadic Motion Capture)</h4>
<ul>
  <li><strong>Modality:</strong> Audio, Text</li>
  <li><strong>Description:</strong> Includes scripted and improvised dyadic interactions labeled with emotional categories.</li>
  <li><strong>Access:</strong> Requires signing a license agreement.</li>
  <li><a href="https://sail.usc.edu/iemocap/" target="_blank">🔗 Download Link</a></li>
</ul>

<h4>🔹 ESD (Emotional Speech Dataset)</h4>
<ul>
  <li><strong>Modality:</strong> Audio, Text</li>
  <li><strong>Description:</strong> Contains emotional speech in English and Chinese from multiple speakers. High-quality recordings with five emotions.</li>
  <li><a href="https://github.com/HLTSingapore/Emotional-Speech-Data" target="_blank">🔗 Download Link</a></li>
</ul>

<h4>🔹 MELD (Multimodal EmotionLines Dataset)</h4>
<ul>
  <li><strong>Modality:</strong> Audio, Text</li>
  <li><strong>Description:</strong> An extension of the EmotionLines dataset, MELD is based on Friends TV show dialogues and supports multimodal emotion classification.</li>
  <li><a href="https://affective-meld.github.io/" target="_blank">🔗 Download Link</a></li>
</ul>
🔗 GitHub Link

### Partial fine-tuning 

This project supports <strong>partial fine-tuning</strong> of pre-trained models such as <strong>HuBERT</strong> and <strong>Wav2Vec2</strong> to adapt them for speech emotion recognition while reducing computational cost. Only selected layers are updated during fine-tuning.

<h4>📂 Fine-tuning Scripts</h4>
<p>Navigate to the <code>fine-tuning/</code> directory, which contains the following scripts:</p>

<table>
  <thead>
    <tr>
      <th>Script</th>
      <th>Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><code>HuBERT.py</code></td>
      <td>Fine-tunes the HuBERT model for speech emotion recognition</td>
    </tr>
    <tr>
      <td><code>Wav2Vec.py</code></td>
      <td>Fine-tunes the Wav2Vec model for speech emotion recognition</td>
    </tr>
  </tbody>
</table>

<h4>🔧 Replace Metadata Path</h4>
<p>Before running, ensure that the correct <strong>metadata path</strong> is set in the script. Example:</p>

<pre><code># Inside fine-tuning/HuBERT.py or fine-tuning/Wav2Vec.py
metadata_path = "/kaggle/input/metadata1/metadata-1/IEMOCAP_metadata_train.csv"
</code></pre>

<p>Modify this path if your dataset is stored elsewhere.</p>

<h4>🚀 Running Fine-tuning</h4>
<p>Once the paths are correctly set, execute the script to start fine-tuning:</p>

<pre><code>cd fine-tuning
python HuBERT.py  # Fine-tune HuBERT
# or
python Wav2Vec.py  # Fine-tune Wav2Vec
</code></pre>

<h4>🛠️ Expected Output</h4>
<ul>
  <li>The model will be fine-tuned on the specified dataset.</li>
  <li>A trained model will be saved in the <code>saved_models/</code> directory.</li>
</ul>

### Feature encoding

<p>
The feature encoding step extracts meaningful representations from raw audio and text data using pre-trained models such as <strong>HuBERT</strong>, <strong>Wav2Vec2</strong>, and <strong>RoBERTa</strong>.
</p>

<h4>📂 Feature Extraction Scripts</h4>
<p>Navigate to the <code>feature_extract/</code> directory, which contains the following scripts:</p>

<table>
  <thead>
    <tr>
      <th>Script</th>
      <th>Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><code>BERT_HuBERT.py</code></td>
      <td>Extracts text and audio features using BERT and HuBERT</td>
    </tr>
    <tr>
      <td><code>BERT_wav2vec2.py</code></td>
      <td>Extracts text and audio features using BERT and Wav2Vec2</td>
    </tr>
    <tr>
      <td><code>RoBERTa-HuBERT.py</code></td>
      <td>Extracts features using RoBERTa for text and HuBERT for audio</td>
    </tr>
    <tr>
      <td><code>RoBERTa-wav2vec.py</code></td>
      <td>Extracts features using RoBERTa for text and Wav2Vec2 for audio</td>
    </tr>
  </tbody>
</table>

<h4>🔧 Replace Metadata Path</h4>
<p>Ensure that the correct dataset paths are set inside the script. Example:</p>

<pre><code># Inside feature_extract/BERT_HuBERT.py
metadata_path = "/kaggle/input/metadata1/metadata-1/IEMOCAP_metadata_train.csv"
output_path = "./data/processed_features/"
</code></pre>

<p>Modify these paths to match your dataset location.</p>

<h4>🚀 Running Feature Extraction</h4>
<p>Execute the script to extract features:</p>

<pre><code>cd feature_extract
python BERT_HuBERT.py
# or
python RoBERTa-wav2vec.py
</code></pre>

<h4>🛠️ Expected Output</h4>
<ul>
  <li>Encoded features will be saved in the <code>data/processed_features/</code> directory.</li>
  <li>These features will be used as inputs for the training step.</li>
</ul>

### Training objective 

To train the FleSER model, use the main Python script located in the root directory. The following command runs the training process:

<pre><code>python main.py --config configs/config.yaml --device cuda</code></pre>

Parameters

--config: Path to the configuration file containing training hyperparameters and model settings.

--device: Specify the device to use (e.g., cuda for GPU training).

Training progress, including loss and accuracy, will be logged using WandB. Ensure your WandB account is properly set up by running:

<pre><code>wandb login</code></pre>

## References
[1] Nhat Truong Pham, SERVER: Multi-modal Speech Emotion Recognition using Transformer-based and Vision-based Embeddings (ICIIT), 2023. Available https://github.com/nhattruongpham/mmser.git.

[2] Mustaqeem Khan, MemoCMT: Cross-Modal Transformer-Based Multimodal Emotion Recognition System (Scientific Reports), 2025. Available https://github.com/tpnam0901/MemoCMT.

[3] Nhat Truong Pham, SER-Fuse: An Emotion Recognition Application Utilizing Multi-Modal, Multi-Lingual, and Multi-Feature Fusion (SOICT), 2023. Available https://github.com/nhattruongpham/SER-Fuse.

## Citation
If you use this code or part of it, please cite the following papers:
```
Update soon
```
## Contact
For any information, please contact the main author:

Nhut Minh Nguyen at FPT University, Vietnam

**Email:** [minhnhut.ngnn@gmail.com](mailto:minhnhut.ngnn@gmail.com)<br>
**ORCID:** <link>https://orcid.org/0009-0003-1281-5346</link> <br>
**GitHub:** <link>https://github.com/nhut-ngnn/</link>


