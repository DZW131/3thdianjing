import json
import os
from pathlib import Path

from .tasks import CLASS_NAMES, get_task_preset


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _coerce_int_list(value):
    if value in (None, "", []):
        return None
    if isinstance(value, str):
        return [int(item.strip()) for item in value.split(",") if item.strip()]
    if isinstance(value, (list, tuple)):
        return [int(item) for item in value]
    return [int(value)]


def _resolve_path(value):
    if not value:
        return value
    path = Path(value)
    if path.is_absolute():
        return str(path)
    return str((PROJECT_ROOT / path).resolve())


def apply_json_config_overrides(parser, args):
    config_path = getattr(args, "config", None)
    if not config_path:
        return args

    resolved_path = _resolve_path(config_path)
    with open(resolved_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)

    defaults = vars(parser.parse_args([]))
    for key, value in payload.items():
        if not hasattr(args, key):
            setattr(args, key, value)
            continue
        if getattr(args, key) == defaults.get(key):
            setattr(args, key, value)

    args.config = resolved_path
    return args


def _apply_task_preset(args):
    if getattr(args, "dataset", None) != "jijie":
        return

    preset = get_task_preset(getattr(args, "task_name", None))
    if preset is None:
        return

    for key, value in preset.items():
        current_value = getattr(args, key, None)
        if current_value in (None, "", []):
            setattr(args, key, value)


def _resolve_remapped_ids(selected_classes, original_class_ids):
    if not selected_classes or not original_class_ids:
        return None
    mapping = {0: 0}
    for new_class_id, old_class_id in enumerate(selected_classes, 1):
        mapping[int(old_class_id)] = new_class_id
    remapped = []
    for original_id in original_class_ids:
        original_id = int(original_id)
        if original_id not in mapping:
            continue
        remapped_id = mapping[original_id]
        if remapped_id not in remapped:
            remapped.append(remapped_id)
    return remapped or None


def _resolve_single_remapped_id(selected_classes, original_class_id):
    values = _coerce_int_list(original_class_id)
    if not values:
        return None
    remapped = _resolve_remapped_ids(selected_classes, [values[0]])
    if remapped:
        return remapped[0]
    if not selected_classes:
        return values[0]
    return None


def _coerce_bool(value):
    if isinstance(value, bool):
        return value
    if value in (None, ""):
        return False
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def finalize_jijie_args(args):
    if getattr(args, "dataset", None) != "jijie":
        return args

    _apply_task_preset(args)

    args.selected_classes = _coerce_int_list(getattr(args, "selected_classes", None))
    args.metric_target_original_classes = _coerce_int_list(getattr(args, "metric_target_original_classes", None))
    args.quantify_original_classes = _coerce_int_list(getattr(args, "quantify_original_classes", None))
    args.label_dilate_original_classes = _coerce_int_list(getattr(args, "label_dilate_original_classes", None))
    args.t_tubule_prior_z_original_classes = _coerce_int_list(
        getattr(args, "t_tubule_prior_z_original_classes", None)
    )
    args.t_tubule_prior_m_original_classes = _coerce_int_list(
        getattr(args, "t_tubule_prior_m_original_classes", None)
    )
    args.t_tubule_prior_target_original_class = _coerce_int_list(
        getattr(args, "t_tubule_prior_target_original_class", None)
    )

    if getattr(args, "manifest_dir", None):
        args.manifest_dir = _resolve_path(args.manifest_dir)

    if getattr(args, "config", None):
        args.config = _resolve_path(args.config)

    if getattr(args, "train_resize_mode", None) in (None, ""):
        args.train_resize_mode = getattr(args, "resize_mode", "pad")
    if getattr(args, "eval_resize_mode", None) in (None, ""):
        args.eval_resize_mode = getattr(args, "resize_mode", "pad")

    if getattr(args, "sliding_window_size", None) in (None, 0):
        args.sliding_window_size = getattr(args, "crop_size", None)
    if getattr(args, "sliding_window_stride", None) in (None, 0):
        window_size = getattr(args, "sliding_window_size", 0) or 0
        args.sliding_window_stride = max(1, window_size // 2) if window_size else None

    if getattr(args, "selection_metric", None) in (None, ""):
        args.selection_metric = "mean_iou"

    args.metric_target_class_ids = _resolve_remapped_ids(
        args.selected_classes,
        args.metric_target_original_classes or args.selected_classes,
    )
    args.quantify_class_ids = _resolve_remapped_ids(
        args.selected_classes,
        args.quantify_original_classes or args.selected_classes,
    )
    args.label_dilate_class_ids = _resolve_remapped_ids(
        args.selected_classes,
        args.label_dilate_original_classes,
    )
    args.enable_t_tubule_prior = _coerce_bool(getattr(args, "enable_t_tubule_prior", False))
    args.t_tubule_prior_z_class_ids = _resolve_remapped_ids(
        args.selected_classes,
        args.t_tubule_prior_z_original_classes or [10],
    )
    args.t_tubule_prior_m_class_ids = _resolve_remapped_ids(
        args.selected_classes,
        args.t_tubule_prior_m_original_classes or [12],
    )
    args.t_tubule_prior_target_class_id = _resolve_single_remapped_id(
        args.selected_classes,
        args.t_tubule_prior_target_original_class or [11],
    )
    if args.enable_t_tubule_prior and args.t_tubule_prior_target_class_id is None:
        print("[Task] T-tubule prior disabled because target class is not in selected_classes.")
        args.enable_t_tubule_prior = False

    if args.selected_classes:
        args.selected_class_names = [CLASS_NAMES[class_id] for class_id in args.selected_classes]
    else:
        args.selected_class_names = CLASS_NAMES[1:]

    if args.metric_target_class_ids:
        args.metric_target_class_names = [
            (["Background"] + args.selected_class_names)[class_id] for class_id in args.metric_target_class_ids
        ]
    else:
        args.metric_target_class_names = []

    return args


def summarize_jijie_run(args):
    selected_classes = ",".join(str(class_id) for class_id in (args.selected_classes or [])) or "all"
    metric_targets = ",".join(str(class_id) for class_id in (args.metric_target_class_ids or [])) or "all"
    lines = [
        "[Task] task={} display={} selected_classes={} metric_targets={}".format(
            getattr(args, "task_name", "default"),
            getattr(args, "task_display_name", getattr(args, "task_name", "default")),
            selected_classes,
            metric_targets,
        ),
        "[Task] train_resize={} eval_resize={} inference_mode={} window={} stride={}".format(
            getattr(args, "train_resize_mode", None),
            getattr(args, "eval_resize_mode", None),
            getattr(args, "inference_mode", "direct"),
            getattr(args, "sliding_window_size", None),
            getattr(args, "sliding_window_stride", None),
        ),
        "[Task] manifest_dir={} config={}".format(
            getattr(args, "manifest_dir", None) or "default",
            getattr(args, "config", None) or "none",
        ),
    ]
    if getattr(args, "enable_t_tubule_prior", False):
        lines.append(
            "[Task] t_tubule_prior=on target={} z_classes={} m_classes={} weight={} warmup={}".format(
                getattr(args, "t_tubule_prior_target_class_id", None),
                getattr(args, "t_tubule_prior_z_class_ids", None),
                getattr(args, "t_tubule_prior_m_class_ids", None),
                getattr(args, "t_tubule_prior_loss_weight", 0.0),
                getattr(args, "t_tubule_prior_warmup_epochs", 0),
            )
        )
    return lines
