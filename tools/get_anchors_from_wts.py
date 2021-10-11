import torch
import math
import argparse

"""
Older detectron2 models have cell anchors burned into the weights. Newer version of detectron2 will ignore these. Therefore you need to extract out and give it in your config yaml file. 
"""

ap = argparse.ArgumentParser()
ap.add_argument("model")

args = ap.parse_args()

try:
    model_dict = torch.load(args.model)
except Exception as e:
    model_dict = torch.load(args.model, map_location=torch.device("cpu"))

cell_anchors = [
    "proposal_generator.anchor_generator.cell_anchors.0",
    "proposal_generator.anchor_generator.cell_anchors.1",
    "proposal_generator.anchor_generator.cell_anchors.2",
    "proposal_generator.anchor_generator.cell_anchors.3",
    "proposal_generator.anchor_generator.cell_anchors.4",
]

sizes = [None for _ in range(5)]
ars = [None for _ in range(3)]
for size_i, cell_anchor in enumerate(cell_anchors):
    if cell_anchor in model_dict["model"]:
        anchors = model_dict["model"][cell_anchor]
        for ar_i, anchor in enumerate(anchors):
            l, t, r, b = anchor
            w = r - l
            h = b - t
            ar = h / w
            ar = ar.cpu().numpy()
            if ars[ar_i] is None:
                ars[ar_i] = ar
            else:
                assert math.isclose(ar, ars[ar_i], abs_tol=1e-3)

            size = math.sqrt(w * h)
            if sizes[size_i] is None:
                sizes[size_i] = size
            else:
                assert math.isclose(size, sizes[size_i], abs_tol=1e-3)

sizes_str = [f"[{round(k)}]" for k in sizes]
sizes_str = ", ".join(sizes_str)
print(f"SIZES: [{sizes_str}]")
ars_str = [f"{float(k):.1f}" for k in ars]
ars_str = ", ".join(ars_str)
print(f"ASPECT_RATIOS: [[{ars_str}]]")
