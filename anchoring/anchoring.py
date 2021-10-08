import json
import argparse
from collections import defaultdict
from warnings import warn

import numpy as np
from statistics import median
from scipy.stats import describe
from sklearn.cluster import KMeans
from tqdm import tqdm


def get_imgs_info_from_coco(coco_dict, give_stats=False):
    ih_list = []
    iw_list = []
    imgsize_dict = {}
    for img_dict in tqdm(coco_dict["images"]):
        ih, iw = img_dict["height"], img_dict["width"]
        ih_list.append(ih)
        iw_list.append(iw)
        imgsize_dict[img_dict["id"]] = iw, ih
    if give_stats:
        print("Images height stats")
        describe_stats(ih_list)
        print("Images width stats")
        describe_stats(iw_list)
    return imgsize_dict


def get_info_from_coco(
    coco_dict,
    include_crowdedness=False,
    imgsize_dict=None,
    target_size=None,
    give_stats=False,
):
    # if include_crowdedness, you need to make sure no overlapping image_ids if 'images' are merged
    # imgsize_dict and target_size should both be given together, or neither given.
    if include_crowdedness:
        counts = defaultdict(int)
    if target_size is not None:
        assert target_size and target_size > 0
    sizes = []
    ars = []
    norm_sizes = []
    for annot in tqdm(coco_dict["annotations"]):
        if include_crowdedness:
            img_id = annot["image_id"]
            counts[img_id] += 1
        l, t, w, h = annot["bbox"]
        if w > 0:
            ar = h / w  # anchor aspect ratios are height/width
            ars.append(ar)
        else:
            warn("This bb has 0 width!")

        size = (w * h) ** 0.5
        sizes.append(size)

        if target_size is not None:
            iw, ih = imgsize_dict[annot["image_id"]]
            imsize = (iw * ih) ** 0.5
            normalised_size = size / imsize * target_size
            norm_sizes.append(normalised_size)
    res = [ars, sizes]
    if include_crowdedness:
        crowdedness = list(counts.values())
        res.append(crowdedness)
    else:
        res.append(None)

    if target_size is not None:
        res.append(norm_sizes)
    else:
        res.append(None)
    if give_stats:
        print("Aspect Ratio stats")
        describe_stats(ars)
        print("BB Size stats")
        describe_stats(sizes)
    return res


def get_clusters(juice_list, k=3):
    kmeans = KMeans(n_clusters=k)
    juice = np.array(juice_list).reshape(-1, 1)
    clusters = kmeans.fit(juice)
    cluster_centers = sorted(clusters.cluster_centers_.flatten().tolist())
    #     print(clusters)
    return cluster_centers


def describe_stats(juice_list):
    descrip = describe(juice_list)
    median_val = median(juice_list)
    sd = descrip.variance ** 0.5
    print(f"minmax {descrip.minmax}, mean {descrip.mean}, median {median_val}, sd {sd}")
    return descrip.minmax, descrip.mean, median_val, sd


def process(coco_dict, include_crowdedness, target_size=None, give_stats=False):
    if target_size or give_stats:
        imgsize_dict = get_imgs_info_from_coco(coco_dict, give_stats=give_stats)
    else:
        imgsize_dict = None
    ars, sizes, crowdedness, norm_sizes = get_info_from_coco(
        coco_dict,
        include_crowdedness=include_crowdedness,
        imgsize_dict=imgsize_dict,
        target_size=target_size,
        give_stats=give_stats,
    )
    ars_clusters = get_clusters(ars)
    print("Aspect Ratios")
    print("[[{:.1f}, {:.1f}, {:.1f}]]".format(*ars_clusters))
    size_clusters = get_clusters(sizes, k=5)
    print("Sizes")
    print("[[{:.1f}, {:.1f}, {:.1f}, {:.1f}, {:.1f}]]".format(*size_clusters))
    if target_size is not None:
        norm_size_clusters = get_clusters(norm_sizes, k=5)
        print("Normalised Sizes")
        print("[[{:.1f}, {:.1f}, {:.1f}, {:.1f}, {:.1f}]]".format(*norm_size_clusters))
    # if include_crowdedness:
    #     crowd_minmax, crowd_mean, crowd_med, crowd_sd = describe_stats(crowdedness)
    #     return ars_clusters, size_clusters, crowd_minmax, crowd_mean, crowd_med, crowd_sd
    # else:
    #     return ars_clusters, size_clusters


def load_coco_jsons(json_list):
    all_coco_dict = {"images": [], "annotations": []}
    for p in tqdm(json_list):
        with open(p, "r") as f:
            coco_dict = json.load(f)
            all_coco_dict["images"].extend(coco_dict["images"])
            all_coco_dict["annotations"].extend(coco_dict["annotations"])
    return all_coco_dict


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "cocojsons", nargs="+", help="List paths to json (coco format) files"
    )
    ap.add_argument(
        "--crowd",
        action="store_true",
        help="whether to calculate crowdedness stats (defaults to false)",
    )
    ap.add_argument(
        "--target-size", help="target size to normalise bb sizes to.", type=float
    )
    ap.add_argument(
        "--stats", help="flag to print stats of bbs and imgs", action="store_true"
    )
    args = ap.parse_args()

    coco_dicts = load_coco_jsons(args.cocojsons)
    process(
        coco_dicts,
        include_crowdedness=args.crowd,
        target_size=args.target_size,
        give_stats=args.stats,
    )
