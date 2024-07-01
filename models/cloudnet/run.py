import sys
import os

# Add the current directory to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from models.base_model import BaseModel
import pandas as pd
import os
from sklearn.model_selection import train_test_split
import tensorflow as tf
from cloudnet_tools.utils import get_input_image_names
from cloudnet_tools.generators import mybatch_generator_train
from cloudnet_tools.losses import jacc_coef
from tensorflow.keras.utils import custom_object_scope

from quantization_tools.converter import Converter, QuantizationType

class CloudNetModel(BaseModel):

    PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
    GLOBAL_PATH = os.path.join(PROJECT_ROOT,'data/38-Cloud/')
    TRAIN_FOLDER = os.path.join(GLOBAL_PATH, '38-Cloud_training')
    MODEL_FOLDER = os.path.join(PROJECT_ROOT, 'saved_models/')

    def representative_data_gen():

        num_samples = 100  # Number of samples to use for calibration
        input_shape = (192, 192, 4)  # Adjust this to match your model's input shape
        batch_size = 32
        max_bit = 65535  # Assuming this is your normalization factor
        val_ratio = 0.2

        train_patches_csv_name = 'training_patches_38-Cloud.csv'
        df_train_img = pd.read_csv(os.path.join(CloudNetModel.TRAIN_FOLDER, train_patches_csv_name))
        train_img, train_msk = get_input_image_names(df_train_img, CloudNetModel.TRAIN_FOLDER, if_train=True)

        train_img_subset = train_img[:50]
        train_msk_subset = train_msk[:50]

        
        train_img_split, val_img_split, train_msk_split, val_msk_split = train_test_split(train_img_subset, train_msk_subset,
                                                                                        test_size=val_ratio,
                                                                                        random_state=42, shuffle=True)

        # Define parameters
        
        gen = mybatch_generator_train(list(zip(train_img_split, train_msk_split)), 
                                    input_shape[0], input_shape[1], 
                                    batch_size, max_bit)

        for _ in range(num_samples // batch_size + 1):
            batch = next(gen)
            # Yield only the input data, not the masks
            yield [tf.cast(batch[0], tf.float32)]

    def load_model():
        model_path = os.path.join(CloudNetModel.MODEL_FOLDER, 'Cloud-Net_full_model.keras')    
        try:
            with custom_object_scope({'jacc_coef': jacc_coef}):
                loaded_model = tf.keras.models.load_model(model_path)
            print(f"Model loaded successfully from {model_path}")
            return loaded_model
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            return None

    def convert(self):
        model = CloudNetModel.load_model()
        converter = Converter(model)
        converter.create_convert_and_save(
            quant_type=QuantizationType.INT8,
            save_dir=CloudNetModel.MODEL_FOLDER,
            model_name='converted_model8.tflite',
            representative_data_gen=CloudNetModel.representative_data_gen
        )
        print(converter.get_model_size(os.path.join(CloudNetModel.MODEL_FOLDER, 'converted_model8.tflite')))


    def benchmark(self):
        raise NotImplementedError("Conversion not implemented for CloudNet model")
    


def get_model():
    return CloudNetModel()