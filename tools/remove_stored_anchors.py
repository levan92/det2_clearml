import argparse
from pathlib import Path


def remove_anchors(pthfile, keep_wts=None):
    if keep_wts is None:
        keep_wts = False

    if pthfile.endswith(".pkl"):
        import pickle

        with open(pthfile, "rb") as f:
            data = pickle.load(f, encoding="latin1")
    elif pthfile.endswith(".pth"):
        import torch

        try:
            data = torch.load(pthfile)
        except RuntimeError:
            data = torch.load(pthfile, map_location=torch.device("cpu"))

    anchorkeys = []
    for k in data["model"].keys():
        if "anchor" in k:
            if keep_wts and ("weight" in k or "bias" in k):
                continue
            anchorkeys.append(k)

    for k in anchorkeys:
        del data["model"][k]
        print("deleted {}!".format(k))

    if len(anchorkeys) > 0:
        og_path = Path(pthfile)
        new_path = og_path.parent / "{}{}{}".format(
            og_path.stem, "_anchor-removed", og_path.suffix
        )

        print("Writing new state dict to {}".format(new_path))
        if pthfile.endswith(".pkl"):
            with new_path.open("wb") as f:
                pickle.dump(data, f)
        else:
            torch.save(data, str(new_path))
    else:
        print("No anchor values stored in here!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("pthfile", help="path to pth file")
    parser.add_argument(
        "--keep-wts", help="flag to not del rpn head wts and bias", action="store_true"
    )
    args = parser.parse_args()

    remove_anchors(args.pthfile, keep_wts=args.keep_wts)
