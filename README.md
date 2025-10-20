<div align="center">
  <h1>CodeClarity: A Framework and Benchmark for Evaluating Multilingual Code Summarization</h1>
  <p>
    <a href="https://arxiv.org/abs/">
      <img src="https://img.shields.io/badge/arXiv-[preprint]-b31b1b.svg" alt="Paper">
    </a>
    <a href="https://www.python.org/downloads/release/python-380/">
      <img src="https://img.shields.io/badge/python-3.8+-blue.svg" alt="Python 3.8+">
    </a>
    <a href="https://huggingface.co/datasets">
      <img src="https://img.shields.io/badge/%F0%9F%A4%97%20Datasets-CodeClarity-yellow?style=flat" alt="Hugging Face">
    </a>
  </p>
</div>

# Overview
CodeClarity is a multilingual benchmark and evaluation suite designed to assess the performance of Large Language Models (LLMs) in code summarization across diverse programming and natural languages. While most existing benchmarks focus on English-only summaries, CodeClarity provides a unified and language-diverse evaluation setup to better understand LLM generalization for global developer communities.

This work introduces the first reproducible foundation for studying multilingual code summarization. We released CodeClarity-Bench and its accompanying pipeline, enabling large-scale community validation and future research on multilingual code understanding. 

CodeClarity introduces:
- CodeClarity-Bench, a dataset of ~7,344 multilingual summaries covering 6 programming languages and 6 natural languages.
- Evaluation in **6 natural languages**.
- Comprehensive evaluation metrics (BERTScore, ROUGE, METEOR, BLEU, ChrF, COMET, SIDE).
- Human-in-the-loop LLM-judge scoring mechanisms for qualitative assessment.
<p align="center">
  <img src="figures/codeclarity-pipeline.png" alt="CodeClarity Pipeline" width="600">
</p>


## Benchmark Composition

| **Dimension**           | **Details**                                                                                |
| ----------------------- | ------------------------------------------------------------------------------------------ |
| Programming Languages   | Python, Java, JavaScript, PHP, Go, Ruby                                                    |
| Natural Languages       | Spanish (ES), French (FR), Hindi (HI), Arabic (AR), Mandarin Chinese (ZH), Portuguese (PT) |
| Function Length Buckets | Short (≤10 lines), Medium (11–30 lines), Long (>30 lines)                                  |

---

## Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/MadhuNimmo/CodeClarity.git
cd CodeClarity
```
### 2. Install Dependencies
```python
pip install -r requirements.txt
```
### 


## Citation
If you find this framework or dataset useful, please consider citing our work:
```bibtex
@misc{madhurima2025codeclarity,
  title={CodeClarity: A Framework and Benchmark for Evaluating Multilingual Code Summarization}, 
  author={Madhurima Chakraborty, Drishti Sharma, Maryam Sikander and Eman Nisar},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2025} 
}
```
---
## Contact
For questions or suggestions, please open an issue or contact the authors at [email].


