import nnunet
from nnunet.training.model_restore import load_model_and_checkpoint_files
class TorchPredictor:
    def __init__(self, model_path):
        self.model = model_path
    def load_model(self):
        folds = None
        print("loading parameters for folds,", folds)
        trainer, params = load_model_and_checkpoint_files(self.model, folds, fp16=False, checkpoint_name="model_final_checkpoint")
        return trainer, params

    def predict(self, data):
        trainer, params = self.load_model()
        softmax = []
        for p in params:
            trainer.load_checkpoint_ram(p, False)
            softmax.append(trainer.predict_preprocessed_data_return_seg_and_softmax(
                data))

        softmax = np.vstack(softmax)
        return np.mean(softmax, 0)

    def preprocess_patient(self, input_files):
        """
        Used to predict new unseen data. Not used for the preprocessing of the training/test data
        :param input_files:
        :return:
        """
        from nnunet.training.model_restore import recursive_find_python_class
        preprocessor_name = self.plans.get('preprocessor_name')
        if preprocessor_name is None:
            if self.threeD:
                preprocessor_name = "GenericPreprocessor"
            else:
                preprocessor_name = "PreprocessorFor2D"

        print("using preprocessor", preprocessor_name)
        preprocessor_class = recursive_find_python_class([join(nnunet.__path__[0], "preprocessing")],
                                                         preprocessor_name,
                                                         current_module="nnunet.preprocessing")
        assert preprocessor_class is not None, "Could not find preprocessor %s in nnunet.preprocessing" % \
                                               preprocessor_name
        preprocessor = preprocessor_class(self.normalization_schemes, self.use_mask_for_norm,
                                           self.transpose_forward, self.intensity_properties)

        d, s, properties = preprocessor.preprocess_test_case(input_files,
                                                             self.plans['plans_per_stage'][self.stage][
                                                                 'current_spacing'])
        return d, s, properties


def main():
    print('hello, this is my predictor receiving pytorch tensors')

if __name__ == '__main__':
    pretrained_model_path = '/home/zhiyuan/project/data/nnUNet_trained_models/nnUNet/3d_fullres/Task004_Hippocampus/nnUNetTrainerV2__nnUNetPlansv2.1/'
    p = TorchPredictor(pretrained_model_path)
    model, params = p.load_model()
    main()
