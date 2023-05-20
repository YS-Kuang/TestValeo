import copy
import pickle
import glob

import numpy as np
from skimage import io

# from . import kitti_utils
from ...ops.roiaware_pool3d import roiaware_pool3d_utils
from ...utils import box_utils, calibration_kitti, common_utils, object3d_custom
from ..dataset import DatasetTemplate

class CustomDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None):
        """
        Args:
            root_path:
            dataset_cfg:
            class_names:
            training:
            logger:
        """
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )
        self.split = self.dataset_cfg.DATA_SPLIT[self.mode]
        self.root_split_path = self.root_path / ('training' if self.split != 'test' else 'testing')

        split_dir = self.root_path / (self.split + '.txt')
        self.sample_id_list = [x.strip() for x in open(split_dir).readlines()] if split_dir.exists() else None
        self.custom_infos = []
        self.include_custom_data(self.mode)

    def include_custom_data(self, mode):
        if self.logger is not None:
            self.logger.info('Loading CUSTOM dataset')
        custom_infos = []

        for info_path in self.dataset_cfg.INFO_PATH[mode]:
            info_path = self.root_path / info_path
            if not info_path.exists():
                continue
            with open(info_path, 'rb') as f:
                infos = pickle.load(f)
                custom_infos.extend(infos)

        self.custom_infos.extend(custom_infos)

        if self.logger is not None:
            self.logger.info('Total samples for CUSTOM dataset: %d' % (len(custom_infos)))

    def set_split(self, split):
        super().__init__(
            dataset_cfg=self.dataset_cfg, class_names=self.class_names, training=self.training, root_path=self.root_path, logger=self.logger
        )
        self.split = split
        self.root_split_path = self.root_path / ('training' if self.split != 'test' else 'testing')

        split_dir = self.root_path / (self.split + '.txt')
        self.sample_id_list = [x.strip() for x in open(split_dir).readlines()] if split_dir.exists() else None

    def get_lidar(self, idx):
        name = idx + '.npy'
        lidar_file = self.root_path.parent.parent.parent/ 'sample_data' / 'Valeo' /'lidar0_15' / name
        return np.load(str(lidar_file))
    
    def get_radar(self, idx):
        name = 'fft_' + idx + '.npy'
        radar_file = self.root_path.parent.parent.parent/ 'sample_data' / 'Valeo' / 'radar0_15' / name
        radar =  np.load(str(radar_file))
        radar = np.csingle(radar)
        return radar
    
    def get_label(self, idx):
        name = idx + '.txt'
        label_file = self.root_path.parent.parent.parent/ 'sample_data' / 'Valeo' / 'label0_15' / name
        return object3d_custom.get_objects_from_label(label_file)
    
    def get_infos(self, num_workers = 4, sample_id_list = None):
        import concurrent.futures as futures
        
        def process_single_scene(sample_idx):
            
            info = {}
            pc_info = {'num_features': 4, 'lidar_idx': sample_idx}
            info['point_cloud'] = pc_info
            obj_list = self.get_label(sample_idx)
            annotations = {}
            annotations['name'] = np.array([obj.cls_type for obj in obj_list])
            annotations['dimensions'] = np.array([[obj.l, obj.w, obj.h] for obj in obj_list])  # lhw(camera) format
            annotations['location'] = np.concatenate([obj.loc.reshape(1, 3) for obj in obj_list], axis=0)
            annotations['rotation_y'] = np.array([obj.ry for obj in obj_list])

            num_objects = len([obj.cls_type for obj in obj_list if obj.cls_type != 'DontCare'])
            num_gt = len(annotations['name'])
            index = list(range(num_objects)) + [-1] * (num_gt - num_objects)
            annotations['index'] = np.array(index, dtype=np.int32)

            loc = annotations['location'][:num_objects]
            dims = annotations['dimensions'][:num_objects]
            rots = annotations['rotation_y'][:num_objects]
            
            gt_boxes = np.concatenate([loc, dims, -(np.pi / 2 + rots[..., np.newaxis])], axis = 1)
            annotations['gt_boxes'] = gt_boxes
            info['annos'] = annotations
            
            return info
        
        sample_id_list = sample_id_list if sample_id_list is not None else self.sample_id_list
        with futures.ThreadPoolExecutor(num_workers) as executor:
            infos = executor.map(process_single_scene, sample_id_list)
        return list(infos)
    
    @staticmethod
    def generate_prediction_dicts(batch_dict, pred_dicts, class_names, output_path=None):
        """
        Args:
            batch_dict:
                frame_id:
            pred_dicts: list of pred_dicts
                pred_boxes: (N, 7), Tensor
                pred_scores: (N), Tensor
                pred_labels: (N), Tensor
            class_names:
            output_path:
        Returns:
        """
        def get_template_prediction(num_samples):
            ret_dict = {
                'name': np.zeros(num_samples),
                'dimensions': np.zeros([num_samples, 3]),
                'location': np.zeros([num_samples, 3]),
                'rotation_y': np.zeros(num_samples),
                'score': np.zeros(num_samples)
            }
            return ret_dict

        def generate_single_sample_dict(batch_index, box_dict):
            pred_scores = box_dict['pred_scores'].cpu().numpy()
            pred_boxes = box_dict['pred_boxes'].cpu().numpy()
            pred_labels = box_dict['pred_labels'].cpu().numpy()
            pred_dict = get_template_prediction(pred_scores.shape[0])
            if pred_scores.shape[0] == 0:
                return pred_dict


            pred_dict['name'] = np.array(class_names)[pred_labels - 1]
            pred_dict['boxes_lidar'] = pred_boxes
            pred_dict['dimensions'] = pred_boxes[:, 3:6]
            pred_dict['location'] = pred_boxes[:, 0:3]
            pred_dict['rotation_y'] = pred_boxes[:, 6]
            pred_dict['score'] = pred_scores

            return pred_dict

        annos = []
        for index, box_dict in enumerate(pred_dicts):
            frame_id = batch_dict['frame_id'][index]

            single_pred_dict = generate_single_sample_dict(index, box_dict)
            single_pred_dict['frame_id'] = frame_id
            annos.append(single_pred_dict)

            if output_path is not None:
                cur_det_file = output_path / ('%s.txt' % frame_id)
                with open(cur_det_file, 'w') as f:
                    name =single_pred_dict['name']
                    loc = single_pred_dict['location']
                    dims = single_pred_dict['dimensions']  # lhw -> hwl

                    for idx in range(len(single_pred_dict['name'])):
                        print('%s -1 -1 %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f'
                              % (name[idx], 
                                 dims[idx][0], dims[idx][1], dims[idx][2], loc[idx][0],
                                 loc[idx][1], loc[idx][2], single_pred_dict['rotation_y'][idx],
                                 single_pred_dict['score'][idx]), file=f)

        return annos
    
    def evaluation(self, det_annos, class_names, **kwargs):
        if 'annos' not in self.custom_infos[0].keys():
            return None, {}

        from ..kitti.kitti_object_eval_python import eval as kitti_eval

        eval_det_annos = copy.deepcopy(det_annos)
        eval_gt_annos = [copy.deepcopy(info['annos']) for info in self.custom_infos]
        ap_result_str, ap_dict = kitti_eval.get_official_eval_result(eval_gt_annos, eval_det_annos, class_names)

        return ap_result_str, ap_dict

    def __len__(self):
        if self._merge_all_iters_to_one_epoch:
            return len(self.custom_infos) * self.total_epochs

        return len(self.custom_infos)

    def __getitem__(self, index):
        # index = 4
        if self._merge_all_iters_to_one_epoch:
            index = index % len(self.custom_infos)

        info = copy.deepcopy(self.custom_infos[index])

        sample_idx = info['point_cloud']['lidar_idx']
        get_item_list = self.dataset_cfg.get('GET_ITEM_LIST', ['points'])
        radar_feature = self.get_radar(sample_idx)
        
        input_dict = {
            'frame_id': sample_idx,
            'radar_features': radar_feature,
        }

        if 'annos' in info:
            annos = info['annos']
            # annos = common_utils.drop_info_with_name(annos, name='DontCare')
            loc, dims, rots = annos['location'], annos['dimensions'], annos['rotation_y']
            gt_names = annos['name']
            gt_boxes = np.concatenate([loc, dims, -(np.pi / 2 + rots[..., np.newaxis])], axis = 1).astype(np.float32)

            input_dict.update({
                'gt_names': gt_names,
                'gt_boxes': gt_boxes
            })

        if "points" in get_item_list:
            points = self.get_lidar(sample_idx)
            input_dict['points'] = points

        data_dict = self.prepare_data(data_dict=input_dict)

        return data_dict


def create_custom_infos(dataset_cfg, class_names, data_path, save_path, workers=4):
    dataset = CustomDataset(dataset_cfg=dataset_cfg, class_names=class_names, root_path=data_path, training=False)
    train_split, val_split = 'train', 'val'

    train_filename = save_path / ('custom_infos_%s.pkl' % train_split)
    val_filename = save_path / ('custom_infos_%s.pkl' % val_split)

    print('---------------Start to generate data infos---------------')

    dataset.set_split(train_split)
    kitti_infos_train = dataset.get_infos(num_workers=workers)
    with open(train_filename, 'wb') as f:
        pickle.dump(kitti_infos_train, f)
    print('Kitti info train file is saved to %s' % train_filename)

    dataset.set_split(val_split)
    kitti_infos_val = dataset.get_infos(num_workers=workers)
    with open(val_filename, 'wb') as f:
        pickle.dump(kitti_infos_val, f)
    print('Kitti info val file is saved to %s' % val_filename)

    print('---------------Start create groundtruth database for data augmentation---------------')

    print('---------------Data preparation Done---------------')


if __name__ == '__main__':
    import sys
    if sys.argv.__len__() > 1 and sys.argv[1] == 'create_custom_infos':
        import yaml
        from pathlib import Path
        from easydict import EasyDict
        dataset_cfg = EasyDict(yaml.safe_load(open(sys.argv[2])))
        ROOT_DIR = (Path(__file__).resolve().parent / '../../../').resolve() #/content/TestValeo
        create_custom_infos(
            dataset_cfg=dataset_cfg,
            class_names=['Vehicle', 'Pedestrian', 'Bicycle'],
            data_path=ROOT_DIR / 'data' / 'custom',
            save_path=ROOT_DIR / 'data' / 'custom'
        )
