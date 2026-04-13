import torch
from torchvision.utils import make_grid

from dataloaders.utils import decode_seg_map_sequence

try:
    from torch.utils.tensorboard import SummaryWriter  # type: ignore
except ImportError:
    try:
        from tensorboardX import SummaryWriter  # type: ignore
    except ImportError:
        class SummaryWriter(object):
            def __init__(self, *args, **kwargs):
                self.log_dir = kwargs.get("log_dir")

            def add_image(self, *args, **kwargs):
                return None

            def add_scalar(self, *args, **kwargs):
                return None

            def close(self):
                return None


class TensorboardSummary(object):
    def __init__(self, directory):
        self.directory = directory

    def create_summary(self):
        return SummaryWriter(log_dir=self.directory)

    def visualize_image(self, writer, dataset, image, target, output, global_step):
        grid_image = make_grid(image[:3].clone().cpu().data, nrow=3, normalize=True)
        writer.add_image("Image", grid_image, global_step)

        predicted_labels = torch.max(output[:3], 1)[1].detach().cpu().numpy()
        pred_grid = make_grid(
            decode_seg_map_sequence(predicted_labels, dataset=dataset),
            nrow=3,
            normalize=True,
        )
        writer.add_image("Predicted label", pred_grid, global_step)

        groundtruth_labels = target[:3].detach().cpu()
        if groundtruth_labels.ndim == 4 and groundtruth_labels.size(1) == 1:
            groundtruth_labels = groundtruth_labels.squeeze(1)
        gt_grid = make_grid(
            decode_seg_map_sequence(groundtruth_labels.numpy(), dataset=dataset),
            nrow=3,
            normalize=True,
        )
        writer.add_image("Groundtruth label", gt_grid, global_step)
