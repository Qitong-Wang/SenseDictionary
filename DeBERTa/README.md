# Knowledge Distillation on DeBERTa-V3-Large with GLUE Tasks

## Overview

This repository implements knowledge distillation using the DeBERTa-V3-large model on the GLUE benchmark tasks. This work builds on Microsoft's official DeBERTa repository:

[Microsoft DeBERTa-v3 Repository](https://github.com/microsoft/DeBERTa)

## Dataset Preparation

The GLUE dataset is not included in this repository due to storage limitations. You can manually download it from the following link:

[GLUE Dataset (Zenodo)](https://zenodo.org/records/15749756?token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6ImE2NzBkYTFjLTlkZDEtNGI0Yi1iN2VlLTAyOGU4NGRiODE0ZSIsImRhdGEiOnt9LCJyYW5kb20iOiI3NGZkYWEwMDkzYTMyNTYyNTI1YTBiOWIwMGJkMDdiOSJ9.klw1Lt4tPhSv52jFDKTbLs0fpS9LJcknNCh5R9WSZCIRGWNWDCg4i9dT3uTj-4IPv-66tuzg5jqB0aRHpxPw1g)

## Quick Evaluation
We provide model checkpoints that can be downloaded here:

[Student Checkpoints and Sense Dictionary (Zenodo)](https://zenodo.org/records/15721516?token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6ImUzNjM2MWNhLTRjN2EtNGRlMS1iMjJmLTUyYWUyYmI5ZDE1OCIsImRhdGEiOnt9LCJyYW5kb20iOiI4Njk4NjE4NGU0MGE2ZGEwNmIzYTJiODg3MmVlNDQyNCJ9.KG7pLZeyAHjjAF2Zx9GLYEyJoeee9MROxWtQI5V18ZfXP3UggHLzGlbd_0dadIlgtyEsS2WxWEZwYaJEn6A7fA)

[Teacher Checkpoints (Zenodo)](https://zenodo.org/records/15730163?token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6IjYxZjdkYTU4LTUyNTgtNDA5Yi1hNGMzLWFjZmRkYjYwZjYyOCIsImRhdGEiOnt9LCJyYW5kb20iOiJiMWE0NDZjYzg4ZmViM2MzNjRhNDhmMDUzZTZhMjc1NyJ9.c68F3e9kTqEXjMDgdZTyvYosfHjmmx4TdeaN_xcWqLt3rgfyDxHgny74iMzBzEszorFc9jhEliD3__qF4rBDlQ)


To evaluate the **teacher model** checkpoint:

```bash
sh run_eval_teacher.sh TASK_NAME
```

To evaluate the **student model** checkpoint:

```bash
sh run_eval_student.sh TASK_NAME
```

Replace `TASK_NAME` with the name of the GLUE task (e.g., `CoLA`, `MRPC`).

## Knowledge Distillation

For the Deberta model, before starting distillation, the teacher model must be fine-tuned on the specific GLUE task. Run:

```bash
sh r1_teacher_model.sh TASK_NAME
```

### Step 1: Construct Sense Dictionary

This step creates a sense dictionary using the outputs of the teacher model. Make sure to update the path to the teacher model checkpoint inside the script if needed.

```bash
sh r2_construct_sense_dictionary.sh TASK_NAME
```

### Step 2: Perform Knowledge Distillation

Perform knowledge distillation by running:

```bash
sh r3_knowledge_distillation.sh TASK_NAME
```
