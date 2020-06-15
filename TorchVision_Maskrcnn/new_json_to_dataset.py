import argparse
import json
import os
import os.path as osp
import warnings
import copy
import numpy as np
import PIL.Image
from skimage import io
import yaml

from labelme import utils
import draw

NAME_LABEL_MAP = {
    '_background_': 0,
    "steel": 1,

}

LABEL_NAME_MAP = ['0: _background_',
                  '1: steel',]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('json_file')
    parser.add_argument('-o', '--out', default=None)
    args = parser.parse_args()

    json_file = args.json_file

    list = os.listdir(json_file)
    for i in range(0, len(list)):
    
        if list[i].find(".json")<0 :
             continue;
        path = os.path.join(json_file, list[i])
        filename = list[i][:-5]
        if os.path.isfile(path):
            data = json.load(open(path))
            img = utils.image.img_b64_to_arr(data['imageData'])
            lbl, lbl_names = utils.shape.labelme_shapes_to_label(img.shape, data['shapes'])  # labelme_shapes_to_label

            # modify labels according to NAME_LABEL_MAP
            lbl_tmp = copy.copy(lbl)
            for key_name in lbl_names:
                old_lbl_val = lbl_names[key_name]
                new_lbl_val = NAME_LABEL_MAP[key_name]
                lbl_tmp[lbl == old_lbl_val] = new_lbl_val
            lbl_names_tmp = {}
            for key_name in lbl_names:
                lbl_names_tmp[key_name] = NAME_LABEL_MAP[key_name]

            # Assign the new label to lbl and lbl_names dict
            lbl = np.array(lbl_tmp, dtype=np.int8)
            lbl_names = lbl_names_tmp
            print ('lbl_names: ',lbl_names)
            # captions = ['%d: %s' % (l, name) for l, name in enumerate(lbl_names)]
            # captions = ['0: _background_', '1: cat', '2: dog']

            lbl_viz = draw.draw_label(lbl, img, LABEL_NAME_MAP)
            out_dir = osp.basename(list[i]).replace('.', '_')
            out_dir = osp.join(osp.dirname(list[i]), out_dir)
            out_dir=json_file+"\\"+out_dir
            print(out_dir)
            if not osp.exists(out_dir):
                os.mkdir(out_dir)

            PIL.Image.fromarray(img).save(osp.join(out_dir, '{}.png'.format(filename)))
            PIL.Image.fromarray(lbl).save(osp.join(out_dir, '{}_gt.png'.format(filename)))
            PIL.Image.fromarray(lbl_viz).save(osp.join(out_dir, '{}_viz.png'.format(filename)))

            with open(osp.join(out_dir, 'label_names.txt'), 'w') as f:
                for lbl_name in lbl_names:
                    f.write(lbl_name + '\n')

            warnings.warn('info.yaml is being replaced by label_names.txt')
            info = dict(label_names=lbl_names)
            with open(osp.join(out_dir, 'info.yaml'), 'w') as f:
                yaml.safe_dump(info, f, default_flow_style=False)

            print('Saved to: %s' % out_dir)


if __name__ == '__main__':
    main()
