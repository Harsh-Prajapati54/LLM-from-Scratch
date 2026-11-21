## Identifying Underfitting and Overfitting

The main metrics used to monitor language-model training are:

- **Training Loss:** Error on the training dataset.
- **Validation Loss:** Error on unseen validation data.
- **Generalization Gap:** `Validation Loss - Training Loss`

### Underfitting

If both training and validation losses remain high and do not decrease sufficiently:

    Train Loss  ───── High
    Val Loss    ───── High

This indicates that the model has not learned the training data sufficiently.

### Healthy Training

If both training and validation losses decrease:

    Train Loss  ↓↓↓
    Val Loss    ↓↓↓

the model is learning and generalizing reasonably well.

### Overfitting

If training loss continues decreasing while validation loss starts increasing consistently:

    Train Loss  ↓↓↓↓↓
    Val Loss    ↓↓↓ ↑↑↑

the model may be overfitting the training data.

### Generalization Gap

    Generalization Gap = Validation Loss - Training Loss

A consistently increasing gap can be a sign of overfitting.

### Important Rule

Do not identify overfitting or underfitting from a single loss value.
Always examine the **trend of Training Loss and Validation Loss over multiple steps/epochs**.