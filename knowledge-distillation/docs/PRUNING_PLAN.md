# Pruning: Model Compression Approach and Implementation Plan

Based on the current KD project (ResNet-34 / MobileNetV2, CIFAR / TinyImageNet), this document recommends pruning methods and lays out a phased implementation plan.

---

## 1. Recommended Pruning Methods

### 1.1 By Pruning Granularity

| Type | Description | Priority | Notes |
|------|------|--------|------|
| **Structured pruning** | Removes whole channels / filters | ⭐⭐⭐ Preferred | Gives real inference speedups; easy to deploy |
| **Unstructured pruning** | Zeroes out individual weights | ⭐⭐ | Only speeds up with sparse kernels/libraries; better suited to research |

**Recommendation**: focus on **structured pruning** (channel/filter pruning). It fits the existing CNNs well and yields real speedups on both CPU and GPU.

### 1.2 Specific Methods

| Method | Library / Implementation | Best For | Combining with KD |
|------|---------|----------|------------|
| **L1-norm channel pruning** | `torch-pruning` | General-purpose, stable | Prune the teacher or student, then distill |
| **BN scaling / filter importance** | Custom or `torch-pruning` | ResNet / MobileNet | Use BN γ as importance, then prune channels |
| **Gradual magnitude pruning** | `torch.nn.utils.prune` | Quick validation | Unstructured; good for getting the pipeline working first |
| **OBS / OBD** | Custom or dedicated library | Advanced, second-order information | Slightly better accuracy, complex to implement |

**First choice**: **L1-norm channel pruning** (e.g. `torch-pruning`) + **gradual pruning** (increasing sparsity step by step during training). Simple to implement, reproducible, and easy to plug into the existing `train_teacher` / `train_student` scripts.

### 1.3 Recommended Libraries

- **[torch-pruning](https://github.com/VainF/Torch-Pruning)** (recommended)
  - Structured, dependency-aware pruning; supports ResNet, MobileNet, etc.
  - Clean API, easy to wrap as something like `prune_model(model, example_input, amount=0.3)`
- **PyTorch built-in** `torch.nn.utils.prune`
  - Good for small **unstructured** experiments first (e.g. `l1_unstructured`)
  - No new dependency; validates the "train + prune + evaluate" pipeline
- **nncf** (Intel)
  - Consider later if combined quantization + pruning or deployment on Intel hardware is needed

---

## 2. Combining Pruning with KD

Three typical approaches, ordered by implementation difficulty:

1. **Prune-then-Distill**
   - Train the teacher → apply structured pruning to the teacher → distill the student from the pruned teacher.
   - Pros: simplest to implement; only the teacher export and the student's teacher loading need to change.
   - Use case: test whether "small teacher + KD" is more efficient than "large teacher + KD".

2. **Distill-then-Prune**
   - Train the student with standard KD → then prune the student (optionally fine-tune for a few epochs).
   - Pros: the teacher is untouched; only the student is compressed, close to the existing `train_student` flow.
   - Use case: the goal is to make the **student smaller and faster**.

3. **Joint Prune + Distill**
   - Apply progressive pruning while training the student (e.g. raise the pruning ratio slightly every N epochs).
   - Pros: compression and distillation happen in one step, with a better chance of balancing accuracy and size.
   - Use case: an advanced stage, after 1 and 2 are working.

Suggested order: **2 (Distill-then-Prune) first, then 1 (Prune-then-Distill), and finally 3 (joint)**.

---

## 3. Phased Implementation Plan

### Phase 1: Environment and Minimal Validation (~1–2 days)

- [x] **1.1** Install dependencies
  - `pip install torch-pruning thop` (optional, for structured pruning and FLOPs);
  - Phase 1 can also be done with `torch.nn.utils.prune` alone.
- [x] **1.2** Write a **standalone pruning script** (e.g. `src/prune_utils.py`):
  - Load an existing checkpoint (teacher or student);
  - Apply **L1 unstructured** pruning to `nn.Linear` / `nn.Conv2d` (e.g. 30%) with `torch.nn.utils.prune`;
  - Save the pruned model and run the `evaluate` logic to check the accuracy change.
- [x] **1.3** Get it working end to end on CIFAR-10: **standard KD student → prune student → evaluate**, and confirm the pipeline works.

**Deliverables**: `prune_utils.py` (load → prune → save), the `prune_student.py` pipeline, and a reproducible Distill-then-Prune experiment (with before/after accuracy).

---

### Phase 2: Structured Pruning (Channel Pruning) (~3–5 days)

- [x] **2.1** Add `torch-pruning` and implement **channel-level pruning**:
  - Wrap it as `apply_structured_prune(model, example_inputs, amount, ignored_layers, ...)`;
  - Apply L2-norm channel pruning to the **MobileNetV2 student**, ignoring the final classifier.
- [x] **2.2** Run the **Distill-then-Prune** experiment:
  - Train the student with the existing `train_student`;
  - `prune_student.py` supports different pruning ratios and evaluates **parameter count, FLOPs, accuracy, and inference time**;
  - Log results to `checkpoints/pruning_comparison.csv` for comparison with `distill_method_comparison`.
- [ ] **2.3** (Optional) Apply channel pruning to the **ResNet-34 teacher**, then distill:
  - Implement **Prune-then-Distill**;
  - Compare student accuracy and training cost for "original teacher + KD" vs "pruned teacher + KD".

**Deliverables**: structured pruning utilities, the Distill-then-Prune pipeline and scripts, `pruning_comparison.csv`.

---

### Phase 3: Gradual Pruning and Hyperparameters (~2–3 days)

- [ ] **3.1** **Gradual pruning**:
  - At the end of each epoch in `train_student`, prune some of the student's layers slightly (e.g. +1% sparsity per epoch until the target is reached);
  - Use a `torch-pruning` scheduler or a simple custom schedule.
- [ ] **3.2** Hyperparameter search (at small scale):
  - Pruning ratio, gradual schedule, whether to fine-tune, number of fine-tuning epochs;
  - Record the best configuration in `configs/config.yaml` or a separate `configs/pruning.yaml`.
- [ ] **3.3** If time allows, try **joint Prune + Distill**:
  - Apply the KD loss and progressive pruning together in the training loop;
  - Compare accuracy and model size against Phase 2's Distill-then-Prune.

**Deliverables**: gradual pruning implementation, recommended hyperparameters, and (if completed) the joint experiment comparison.

---

### Phase 4: Reproducibility and Documentation (~1–2 days)

- [x] **4.1** Add **scripts**:
  - `run_prune_student.sh`, consistent in style with the existing `run_distill_methods.sh`;
  - Supports environment variables such as `STUDENT_CKPT`, `PRUNE_RATIO`, `PRUNE_METHOD`, `FINETUNE_EPOCHS`; these can also be passed directly on the CLI.
- [ ] **4.2** Update the **README**:
  - How to install the extra dependencies, how to run the pruning experiments, and how to read the tables in `results/`.
- [ ] **4.3** Write a short **PRUNING_REPORT.md** under `results/`:
  - Pruning setup, main results (parameter count, FLOPs, accuracy), comparison with the existing KD results, conclusions and future work.

**Deliverables**: reproducible scripts, README update, PRUNING_REPORT.md.

---

## 4. Proposed Code Structure

```text
knowledge-distillation/
├── configs/
│   ├── config.yaml
│   └── pruning.yaml          # New: pruning configuration
├── src/
│   ├── prune_utils.py        # New: pruning utilities (unstructured + structured wrappers)
│   ├── train_student.py      # Can later gain --prune and similar options
│   └── ...
├── scripts/
│   ├── run_prune_student.sh           # Distill-then-Prune
│   ├── run_prune_teacher_distill.sh   # Prune-then-Distill
│   └── ...
├── results/
│   ├── distill_method_comparison.csv
│   ├── pruning_comparison.csv         # New: pruning ratio vs accuracy/size
│   └── PRUNING_REPORT.md              # New
└── docs/
    └── PRUNING_PLAN.md                # This file
```

---

## 5. Dependencies

On top of the current `requirements.txt`, add:

```text
# Pruning (optional, for Phase 2+)
torch-pruning>=1.3.0
```

Not needed if Phase 1 uses only `torch.nn.utils.prune`.

---

## 6. Summary

| Phase | Content | Expected Output |
|------|------|----------|
| **Phase 1** | Built-in prune, small Distill-then-Prune experiment | `prune_utils`, working pipeline |
| **Phase 2** | Structured channel pruning, both Distill/Prune orderings | Pruning ratio vs accuracy vs size comparison |
| **Phase 3** | Gradual pruning, joint Prune+KD, hyperparameters | Better compression and configurations |
| **Phase 4** | Scripts, README, report | Reproducibility and documentation |

Prioritize **Phase 1 + Phase 2**, which already produce a useful set of pruning + KD results; Phases 3–4 can follow as time permits.
