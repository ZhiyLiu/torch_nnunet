import nnunet
import torch
import numpy as np
from copy import deepcopy
from batchgenerators.utilities.file_and_folder_operations import subfiles, join, isfile
from nnunet.training.model_restore import load_model_and_checkpoint_files
from nnunet.preprocessing.preprocessing import GenericPreprocessor
from nnunet.preprocessing.cropping import get_case_identifier_from_npz, ImageCropper
def check_input_folder_and_return_caseIDs(input_folder, expected_num_modalities):
    print("This model expects %d input modalities for each image" % expected_num_modalities)
    files = subfiles(input_folder, suffix=".nii.gz", join=False, sort=True)

    maybe_case_ids = np.unique([i[:-12] for i in files])

    remaining = deepcopy(files)
    missing = []

    assert len(files) > 0, "input folder did not contain any images (expected to find .nii.gz file endings)"

    # now check if all required files are present and that no unexpected files are remaining
    for c in maybe_case_ids:
        for n in range(expected_num_modalities):
            expected_output_file = c + "_%04.0d.nii.gz" % n
            if not isfile(join(input_folder, expected_output_file)):
                missing.append(expected_output_file)
            else:
                remaining.remove(expected_output_file)

    print("Found %d unique case ids, here are some examples:" % len(maybe_case_ids), np.random.choice(maybe_case_ids, min(len(maybe_case_ids), 10)))
    print("If they don't look right, make sure to double check your filenames. They must end with _0000.nii.gz etc")

    if len(remaining) > 0:
        print("found %d unexpected remaining files in the folder. Here are some examples:" % len(remaining), np.random.choice(remaining, min(len(remaining), 10)))

    if len(missing) > 0:
        print("Some files are missing:")
        print(missing)
        raise RuntimeError("missing files in input_folder")

    return maybe_case_ids

class TorchPredictor:
    def __init__(self, model_path):
        self.model = model_path
        self.trainer, self.params = self.load_model()
    def load_model(self):
        folds = None
        print("loading parameters for folds,", folds)
        trainer, params = load_model_and_checkpoint_files(self.model, folds, fp16=False, checkpoint_name="model_final_checkpoint")
        return trainer, params


    def generate_data(self, data_path):
        # form list of file names
        # expected_num_modalities = self.trainer.plans['num_modalities']

        # # check input folder integrity
        # case_ids = check_input_folder_and_return_caseIDs(data_path, expected_num_modalities)

        # all_files = subfiles(data_path, suffix=".nii.gz", join=False, sort=True)
        # list_of_lists = [[join(data_path, i) for i in all_files if i[:len(j)].startswith(j) and
        #               len(i) == (len(j) + 12)] for j in case_ids]

        # part_id = 0
        # num_parts = 1

        file_name = ['/home/zhiyuan/project/data/nnUNet_raw/nnUNet_raw_data/Task004_Hippocampus/imagesTs/hippocampus_392_0000.nii.gz']
        data, seg, properties = ImageCropper.crop_from_list_of_files(file_name)
        plans = self.trainer.plans
        if plans.get('transpose_forward') is None or plans.get('transpose_backward') is None:
            print("WARNING! You seem to have data that was preprocessed with a previous version of nnU-Net. "
                  "You should rerun preprocessing. We will proceed and assume that both transpose_foward "
                  "and transpose_backward are [0, 1, 2]. If that is not correct then weird things will happen!")
            plans['transpose_forward'] = [0, 1, 2]
            plans['transpose_backward'] = [0, 1, 2]

        # 1. From preprocessing.GenericPreprocessor.preprocess_test_case
        preprocessor = GenericPreprocessor(plans['normalization_schemes'],
                                           plans['use_mask_for_norm'],
                                           plans['transpose_forward'],
                                           plans['dataset_properties']['intensityproperties'])

        data = data.transpose((0, *[i + 1 for i in preprocessor.transpose_forward]))
        seg = seg.transpose((0, *[i + 1 for i in preprocessor.transpose_forward]))

        data, seg, properties = preprocessor.resample_and_normalize(data, plans['plans_per_stage'][self.trainer.stage]['current_spacing'], properties, seg,
                                                            force_separate_z=None)

        output_data = data.astype(np.float32)

        return output_data, seg, properties

    def predict(self, data):
        """
        Input: data (torch tensor)
        """
        #1. convert torch to numpy
        data = data.numpy()
        # 2. From predict.predict_cases
        softmax = []
        for p in self.params:
            self.trainer.load_checkpoint_ram(p, False)
            softmax.append(self.trainer.predict_preprocessed_data_return_seg_and_softmax(
                data, all_in_gpu=False))

        softmax = np.vstack(softmax)
        return np.mean(softmax, 0)

def main():
    print('hello, this is my predictor receiving pytorch tensors')

if __name__ == '__main__':
    pretrained_model_path = '/home/zhiyuan/project/data/nnUNet_trained_models/nnUNet/3d_fullres/Task004_Hippocampus/nnUNetTrainerV2__nnUNetPlansv2.1/'
    p = TorchPredictor(pretrained_model_path)
    model, params = p.load_model()
    main()
