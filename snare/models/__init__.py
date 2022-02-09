from models.single_cls import SingleClassifier
from models.zero_shot_cls import ZeroShotClassifier
from models.rotator import Rotator
from models.voxel_encoder import cnn_encoder

names = {
    # classifiers
    'single_cls': SingleClassifier,
    'zero_shot_cls': ZeroShotClassifier,

    # rotators
    'rotator': Rotator,
    
    # encoder
    'cnn': cnn_encoder,
}
