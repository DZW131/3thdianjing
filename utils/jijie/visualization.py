import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle


CLASS_COLOR_BY_NAME = {
    "Background": (0, 0, 0),
    "严重损伤线粒体": (220, 53, 69),
    "中度损伤线粒体": (255, 159, 28),
    "健康线粒体": (46, 196, 182),
    "自噬线粒体": (131, 56, 236),
    "肌浆网": (63, 167, 214),
    "闰盘": (255, 0, 110),
    "Z线样物质堆积": (255, 195, 0),
    "脂滴": (44, 182, 125),
    "糖原颗粒": (176, 122, 161),
    "Z线": (0, 109, 119),
    "T管": (245, 158, 11),
    "M线": (58, 134, 255),
    "心肌侧管": (107, 76, 154),
}


FALLBACK_COLORS = [
    (0, 0, 0),
    (220, 53, 69),
    (255, 159, 28),
    (46, 196, 182),
    (131, 56, 236),
    (63, 167, 214),
    (255, 0, 110),
    (255, 195, 0),
    (44, 182, 125),
    (176, 122, 161),
    (0, 109, 119),
    (245, 158, 11),
    (58, 134, 255),
    (107, 76, 154),
    (130, 80, 70),
    (50, 50, 50),
]


def build_palette_for_class_names(class_names):
    palette = []
    for index, class_name in enumerate(class_names):
        color = CLASS_COLOR_BY_NAME.get(class_name)
        if color is None:
            color = FALLBACK_COLORS[index % len(FALLBACK_COLORS)]
        palette.append(color)
    return np.asarray(palette, dtype=np.uint8)


def colorize_mask(mask, class_names):
    palette = build_palette_for_class_names(class_names)
    rgb = np.zeros(mask.shape + (3,), dtype=np.uint8)
    for class_id, color in enumerate(palette):
        rgb[mask == class_id] = color
    return rgb


def blend_mask(image, colored_mask, alpha=0.4):
    overlay = image.astype(np.float32).copy()
    foreground = np.any(colored_mask > 0, axis=2)
    overlay[foreground] = (1.0 - alpha) * overlay[foreground] + alpha * colored_mask[foreground].astype(np.float32)
    return np.clip(overlay, 0, 255).astype(np.uint8)


def palette_rows(class_names):
    palette = build_palette_for_class_names(class_names)
    rows = []
    for class_id, class_name in enumerate(class_names):
        color = palette[class_id]
        rows.append(
            {
                "class_id": class_id,
                "class_name": class_name,
                "color_r": int(color[0]),
                "color_g": int(color[1]),
                "color_b": int(color[2]),
                "hex_color": "#{:02X}{:02X}{:02X}".format(int(color[0]), int(color[1]), int(color[2])),
            }
        )
    return rows


def render_class_legend(class_names, present_class_ids=None, title="Class Legend"):
    palette = build_palette_for_class_names(class_names)
    if present_class_ids is None:
        present_class_ids = list(range(len(class_names)))
    present_class_ids = [class_id for class_id in present_class_ids if 0 <= class_id < len(class_names)]
    if not present_class_ids:
        present_class_ids = [0]

    row_count = len(present_class_ids)
    figure_height = max(2.4, 0.55 * row_count + 1.0)
    fig, ax = plt.subplots(figsize=(6.8, figure_height))
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, float(row_count))
    ax.axis("off")

    for row_index, class_id in enumerate(present_class_ids):
        y = row_count - row_index - 0.8
        color = palette[class_id].astype(np.float32) / 255.0
        ax.add_patch(Rectangle((0.05, y), 0.12, 0.45, facecolor=color, edgecolor="black", linewidth=0.5))
        ax.text(0.22, y + 0.225, "{}: {}".format(class_id, class_names[class_id]), va="center", fontsize=10)

    ax.set_title(title, fontsize=12)
    fig.tight_layout()
    return fig
