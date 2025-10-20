# MuCoSF: Multilingual Code Summarization Framework for Evaluating LLMs

MuCoSF is a multilingual evaluation framework designed to assess the capabilities of Large Language Models (LLMs) in generating code summaries across diverse natural languages. While most existing code summarization benchmarks focus solely on English, MuCoSF evaluates how effectively LLMs can generate summaries in various languages, addressing the broader needs of the global developer community.

This repository provides the official implementation for the paper:  
**[ Paper Title Here]**  
[ ArXiv Link Here]


---

## Overview

LLMs have shown remarkable performance in code summarization tasks. However, the lack of multilingual evaluation pipelines limits their applicability for non-English developers. 

MuCoSF introduces:
- Multilingual benchmarks across **6 programming languages**.
- Evaluation in **6 natural languages**.
- Comprehensive evaluation metrics (BERTScore, ROUGE, METEOR, BLEU, ChrF, COMET, SIDE).
- Human-in-the-loop LLM-judge scoring mechanisms for qualitative assessment.


## Benchmark Dataset

| **Dimension**           | **Options**                                                                                |
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

```

## Citation
If you find this framework or our paper useful in your research, please consider citing our work:
```graphql
@article{ahmed2024tdd,
  title={MuCoSF: Multilingual code summarization framework for evaluating LLMs}, 
  author={Madhurima, Drishti, Eman, Maryam},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2024} 
}
```
