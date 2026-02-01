# 📝 Fin-RATE: Financial Analytics and Tracking Evaluation Benchmark for LLMs on SEC Filings

![overview-image](./assets/image-20260121002058463.png)

**Fin-RATE** is a real-world benchmark to evaluate large language models (LLMs) on professional-grade reasoning over **U.S. SEC filings**. 
It targets financial analyst workflows that demand:

- 📄 **Long-context understanding**
- ⏱️ **Cross-year tracking**
- 🏢 **Cross-company comparison**
- 📊 **Structured diagnosis of model failures**

> 📘 [Paper (arXiv link TBD)] | 🔗 [Leaderboard (Coming Soon)] 
> ⬇️ SEC-based QA benchmark with 7,500 instances + interpretable evaluation.

---

## 🔍 Overview

Fin-RATE includes **three core QA tasks**, modeling real-world financial reasoning:

<img src="./assets/fig-dataset-overview_01.png" alt="fig-dataset-overview_01" style="zoom: 5%;" />

| Task Type | Description                                                  |
| --------- | ------------------------------------------------------------ |
| **DR-QA** | Detail & Reasoning: fine-grained reasoning within one SEC section |
| **EC-QA** | Enterprise Comparison: reasoning across peer firms in the same industry/year |
| **LT-QA** | Longitudinal Tracking: analyzing trends across years for the same firm |

### DR-QA Example

<center>
<img src="./assets/fig-eg-DR_01.png" alt="DR-QA Example" style="zoom: 10%;">
</center>


### EC-QA Example

<center>
<img src="./assets/fig-eg-EC_01.png" alt="EC-QA Example" style="zoom: 10%;">
</center>


### LT-QA Example

<center>
<img src="./assets/fig-eg-LT_01.png" alt="LT-QA Example" style="zoom: 10%;">
</center>


---

## 📦 Dataset Structure

```bash
Fin-RATE/
├── corpus/            # Parsed and chunked SEC filings
│   └── corpus.zip
├── qa/                # QA datasets
│   ├── dr_qa.json
│   ├── ec_qa.json
│   └── lt_qa.json
├── evaluation/        # evaluation metrics using LLM judge (including verdicts, error taxonomy, fine-grained dimensions)
│   └── qa_llm_judge.py
└── requirements.txt

