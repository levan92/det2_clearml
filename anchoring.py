import json
import argparse
from collections import defaultdict
from warnings import warn

import numpy as np
from statistics import median
from scipy.stats import describe 
from sklearn.cluster import KMeans 
from tqdm import tqdm 

def get_info_from_coco(coco_dict, include_crowdedness=False):
    # if include_crowdedness, you need to make sure no overlapping image_ids if 'images' are merged
    if include_crowdedness:
        counts = defaultdict(int)
    sizes = []
    ars = []
    for annot in tqdm(coco_dict['annotations']):
        if include_crowdedness:
            img_id = annot['image_id']
            counts[img_id] += 1
        l,t,w,h = annot['bbox']
        if w > 0:
            ar = h/w #anchor aspect ratios are height/width
            ars.append(ar)
        else:
            warn('This bb has 0 width!')
        size = ( w * h ) ** 0.5
        sizes.append(size)
    if include_crowdedness:
        crowdedness = list(counts.values())
        return ars, sizes, crowdedness
    else:
        return ars, sizes

def get_clusters(juice_list, k=3):
    kmeans = KMeans(n_clusters=k)
    juice = np.array(juice_list).reshape(-1,1)
    clusters = kmeans.fit(juice)
    cluster_centers = sorted(clusters.cluster_centers_.flatten().tolist())
#     print(clusters)
    return cluster_centers 

def describe_stats(juice_list):
    descrip = describe(juice_list)
    median_val = median(juice_list)
    sd = descrip.variance**0.5
    print(f'minmax {descrip.minmax}, mean {descrip.mean}, median {median_val}, sd {sd}')
    return descrip.minmax, descrip.mean, median_val, sd

def process(coco_dict, include_crowdedness):
    res = get_info_from_coco(coco_dict, include_crowdedness=include_crowdedness)
    ars_clusters = get_clusters(res[0])    
    print('Aspect Ratios')
    print('[[{:.1f}, {:.1f}, {:.1f}]]'.format(*ars_clusters))
    size_clusters = get_clusters(res[1], k=5)
    print('Sizes')
    print('[[{:.1f}, {:.1f}, {:.1f}, {:.1f}, {:.1f}]]'.format(*size_clusters))
    if include_crowdedness:
        crowd_minmax, crowd_mean, crowd_med, crowd_sd = describe_stats(res[2])
        return ars_clusters, size_clusters, crowd_minmax, crowd_mean, crowd_med, crowd_sd
    else:
        return ars_clusters, size_clusters

def load_coco_jsons(json_list):
    all_coco_dict = {'images':[], 'annotations':[]}
    for p in tqdm(json_list):
        with open(p, 'r') as f:
            coco_dict = json.load(f)
            all_coco_dict['images'].extend(coco_dict['images'])
            all_coco_dict['annotations'].extend(coco_dict['annotations'])      
    return all_coco_dict


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('cocojsons', nargs='+', help='List paths to json (coco format) files')
    ap.add_argument('--crowd', action='store_true', help='whether to calculate crowdedness stats (defaults to false)')
    args = ap.parse_args()

    coco_dicts = load_coco_jsons(args.cocojsons)
    process(coco_dicts, include_crowdedness=args.crowd)