import json
import argparse

from anchoring import get_info_from_coco, describe_stats

ap = argparse.ArgumentParser()
ap.add_argument('cocojson', help='Path to json (coco format) file')
args = ap.parse_args()

with open(args.cocojson, 'r') as f:
    coco_dict = json.load(f)

ars, sizes, crowdedness = get_info_from_coco(coco_dict, include_crowdedness=True)

print('Aspect Ratio')
describe_stats(ars)
print('Sizes')
describe_stats(sizes)
print('Crowdedness')
describe_stats(crowdedness)