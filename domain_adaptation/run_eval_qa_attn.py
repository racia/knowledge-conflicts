from transformers import TrainerCallback
import json
import os
import math


class SaveMetricsPerEpoch(TrainerCallback):

    def __init__(self):
        self.trainer = None

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        epoch_num = int(state.epoch)

        json_path = os.path.join(args.output_dir, f"metrics_epoch_{epoch_num}.json")

        eval_loss = metrics.get('eval_loss', None)
        if eval_loss is not None:
            ppl = math.exp(eval_loss)
            metrics['ppl'] = ppl

        with open(json_path, "w") as f:
            json.dump(metrics, f, indent=4)

        print(f"Epoch {epoch_num}, Saved **VAL** metrics to {json_path}")
