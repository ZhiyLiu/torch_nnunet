import nnunet
from nnunet.training.model_restore import load_model_and_checkpoint_files
from nnunet.preprocessing.cropping import get_case_identifier_from_npz, ImageCropper
class TorchPredictor:
    def __init__(self, model_path):
        self.model = model_path
    def load_model(self):
        folds = None
        print("loading parameters for folds,", folds)
        trainer, params = load_model_and_checkpoint_files(self.model, folds, fp16=False, checkpoint_name="model_final_checkpoint")
        return trainer, params

    def predict(self, data_path):
        """
        First suppose input data is just a path pointing to the data (3D)
        """
        # load plans and params
        trainer, params = self.load_model()
        plans = trainer.plans
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
        data, seg, properties = ImageCropper.crop_from_list_of_files(data_path)
        data = data.transpose((0, *[i + 1 for i in preprocessor.transpose_forward]))
        seg = seg.transpose((0, *[i + 1 for i in preprocessor.transpose_forward]))

        data, seg, properties = preprocessor.resample_and_normalize(data, plans['plans_per_stage'][self.stage]['current_spacing'], properties, seg,
                                                            force_separate_z=None)

        output_data = data.astype(np.float32)

        # 2. From predict.predict_cases

        softmax = []
        for p in params:
            trainer.load_checkpoint_ram(p, False)
            softmax.append(trainer.predict_preprocessed_data_return_seg_and_softmax(
                output_data))

        softmax = np.vstack(softmax)
        return np.mean(softmax, 0)

def main():
    print('hello, this is my predictor receiving pytorch tensors')

if __name__ == '__main__':
    pretrained_model_path = '/home/zhiyuan/project/data/nnUNet_trained_models/nnUNet/3d_fullres/Task004_Hippocampus/nnUNetTrainerV2__nnUNetPlansv2.1/'
    p = TorchPredictor(pretrained_model_path)
    model, params = p.load_model()
    main()
