from transformers import TrainerCallback


class EpochMetricsCallback(TrainerCallback):
    def __init__(self):
        self.epoch_metrics = []

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        train_loss = None
        for entry in reversed(state.log_history):
            if "loss" in entry and "eval_loss" not in entry:
                train_loss = entry["loss"]
                break

        if metrics is not None:
            self.epoch_metrics.append({
                "epoch": state.epoch,
                "train_loss": train_loss,
                "val_loss": metrics.get("eval_loss", None),
                "accuracy": metrics.get("eval_accuracy", None),
                "f1": metrics.get("eval_f1", None),
                "precision": metrics.get("eval_precision", None),
                "recall": metrics.get("eval_recall", None),
            })


def plot_epoch_metrics(epoch_metrics):
    import matplotlib.pyplot as plt

    epochs = [m["epoch"] for m in epoch_metrics]
    train_loss = [m["train_loss"] for m in epoch_metrics]
    val_loss = [m["val_loss"] for m in epoch_metrics]
    accuracy = [m["accuracy"] for m in epoch_metrics]
    f1 = [m["f1"] for m in epoch_metrics]
    precision = [m["precision"] for m in epoch_metrics]
    recall = [m["recall"] for m in epoch_metrics]

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle("Metryki treningowe — co epokę", fontsize=14)

    def _plot(ax, y, label, color):
        ax.plot(epochs, y, marker="o", color=color, linewidth=2)
        ax.set_title(label)
        ax.set_xlabel("Epoka")
        ax.set_xticks(epochs)
        ax.grid(True, linestyle="--", alpha=0.5)

    _plot(axes[0, 0], train_loss, "Training Loss", "#E55A2B")
    _plot(axes[0, 1], val_loss, "Validation Loss", "#2B7BE5")
    _plot(axes[0, 2], accuracy, "Accuracy", "#27A06A")
    _plot(axes[1, 0], f1, "F1", "#9B59B6")
    _plot(axes[1, 1], precision, "Precision", "#E5A82B")
    _plot(axes[1, 2], recall, "Recall", "#E52B6A")

    plt.tight_layout()
    plt.savefig("epoch_metrics.png", dpi=150)
    print("\nWykres zapisany jako epoch_metrics.png")
    plt.show()
