# Knowledge Distillation on LLM2Vec Llama3-8b-Instruct with MTEB Classification Tasks

## Overview

This repository provides an implementation of knowledge distillation using the **LLM2Vec Llama3-8b-Instruct** model, specifically applied to **MTEB** classification tasks. Our approach builds upon the **LLM2Vec** and **MTEB** repositories, which you can explore here:

- [LLM2Vec Repository](https://github.com/McGill-NLP/llm2vec)  
- [MTEB Repository](https://github.com/embeddings-benchmark/mteb)  


## Quick Evaluation

We provide model checkpoints and MTEB dataset that can be downloaded here:

[Model Checkpoints and MTEB Dataset (Zenodo)](https://zenodo.org/records/15717874?token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6IjQ5NjBiNDdmLTExN2YtNDA3Zi04MjE0LTEzNDljMzYyYWFjMCIsImRhdGEiOnt9LCJyYW5kb20iOiJlYWRhYTg3ZTg3ZWQ4ZTIwMmFiNjYyMTc5YTM5OWQ4OCJ9.RumQ32CMo3s9lHfHCdadhk3lCNltDjf1bpCA91tTABh-VFZWT_X53m5H0x30sv-AsonONddDT9gudoV0EeIKBA)

To evaluate the **student model** checkpoint:

```bash
sh run_student_eval.sh TASK_ID
```

It requires a task index (1–16) as an argument. For example, `1` maps to ArxivClassification and `16` maps to YelpReviewFullClassification.

## Training on All Datasets for Better Generalization (SKD model in Paper)

This repository provides a streamlined pipeline for extracting sense embeddings from MMLU classification datasets and using them for knowledge distillation.

Here is the code to **train on all datasets** to improve generalization and reduce bias. This corresponds to the "SKD" model in Table 4 of our paper. You can do this by:

1. **Extracting sense embeddings for each dataset**  
   Run the script below with the task index (1–16). Note: comment out the last part "Generate Combinepkl".
   ```bash
   sh run_gather_sense.sh TASK_ID
   ```

2. **Merging all extracted sense dictionaries**  
   ```bash
   python a0_generate_combinepkl.py --folder_names ./sense_dict/ --output_keyword ./sense_dict/combine_1000
   ```

3. **Accelerating clustering using multithreading**  
   ```bash
   sh run_multithread_cluster.sh
   ```

4. **Training the student model**  
   Adjust machine-specific parameters like `--master_addr=$1`, `--master_port=$2`, and `--node_rank=$3`.
   ```bash
   sh run_train.sh
   ```

## Step-by-Step Pipeline (SKD-sep model in Paper)

This lightweight pipeline is ideal for single sub-tasks and corresponds to the SKD-sep model in Table 4.

1. **Gather sense embeddings**  
   ```bash
   sh run_gather_sense.sh TASK_ID
   ```

2. **Cluster sense embeddings**  
   ```bash
   sh run_sep_cluster.sh
   ```

3. **Train the student model**  
   Adjust machine-specific parameters like `--master_addr=$2`, `--master_port=$3`, and `--node_rank=$4`.
   ```bash
   sh run_train.sh TASK_ID
   ```

## Notes

### Machine-Specific Optimizations

Certain advanced features such as fast attention mechanisms have been disabled to support hardware with limited resources. If you have a more powerful machine, consider enabling these features. Future updates will include better hardware compatibility.

### Clustering Algorithm

Although MCL with k-means is effective, it is time-consuming and requires tuning. We are developing a faster clustering method that reduces memory of sense dictionary while maintaining performance. Further improvements are in progress.
