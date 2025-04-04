# FleSER: Multi-modal emotion recognition via dynamic fuzzy membership and attention fusion

> Please press ⭐ button and/or cite papers if you feel helpful.

<p align="center">
<img src="https://img.shields.io/github/stars/nhut-ngnn/FlexibleMMSER">
<img src="https://img.shields.io/github/forks/nhut-ngnn/FlexibleMMSER">
<img src="https://img.shields.io/github/watchers/nhut-ngnn/FlexibleMMSER">
</p>

<div align="center">

[![python](https://img.shields.io/badge/-Python_3.11.11-blue?logo=python&logoColor=white)](https://www.python.org/downloads/)
[![pytorch](https://img.shields.io/badge/Torch_2.0.1-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)
[![cuda](https://img.shields.io/badge/-CUDA_11.8-green?logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-toolkit-archive)
</div>

<p align="center">
<img src="https://img.shields.io/badge/Last%20updated%20on-15.01.2025-brightgreen?style=for-the-badge">
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
> 
>
> Index Terms:


## Install
### Clone this repository
```
git clone https://github.com/nhut-ngnn/FlexibleMMSER.git
```

### Create Conda Enviroment and Install Requirement
Navigate to the project directory and create a Conda environment:
```
cd FlexibleMMSER
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

### Training objective 

## References

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


