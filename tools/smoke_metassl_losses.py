import os
import sys

import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from train import build_parser
from utils.jijie.config import apply_json_config_overrides
from utils.loss import MetaSSLLabeledLoss, MetaSSLUnlabeledLoss


def main():
    parser = build_parser()
    args = parser.parse_args(["--config", "configs/mito/mito_v0_8_metassl_mt_semi.json"])
    args = apply_json_config_overrides(parser, args)
    assert args.metassl_reference_mode == "ema", args.metassl_reference_mode

    logit = torch.randn(2, 5, 16, 16, requires_grad=True)
    ref = torch.randn(2, 5, 16, 16)
    target = torch.randint(0, 5, (2, 16, 16))
    labeled_loss, _ = MetaSSLLabeledLoss(
        num_classes=5,
        initial_threshold=0.9,
        ema_alpha=0.5,
        delta_l=0.3,
    )(logit, ref, target)
    labeled_loss.backward()

    unlabeled_logit = torch.randn(2, 5, 16, 16, requires_grad=True)
    unlabeled_ref = torch.randn(2, 5, 16, 16)
    unlabeled_loss, _ = MetaSSLUnlabeledLoss(
        num_classes=5,
        initial_threshold=0.9,
        ema_alpha=0.5,
        delta_u=0.6,
    )(unlabeled_logit, unlabeled_ref)
    unlabeled_loss.backward()

    assert torch.isfinite(labeled_loss)
    assert torch.isfinite(unlabeled_loss)
    assert logit.grad is not None and torch.isfinite(logit.grad).all()
    assert unlabeled_logit.grad is not None and torch.isfinite(unlabeled_logit.grad).all()

    print(
        "metassl_smoke_ok",
        args.checkname,
        "cuda={}".format(torch.cuda.is_available()),
        "labeled_loss={:.4f}".format(float(labeled_loss.detach())),
        "unlabeled_loss={:.4f}".format(float(unlabeled_loss.detach())),
    )


if __name__ == "__main__":
    main()
