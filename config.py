import os

# 路径配置
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
TRAIN_DIR = os.path.join(DATA_DIR, "train")
TEST_DIR = os.path.join(DATA_DIR, "test")
MODEL_SAVE_PATH = os.path.join(BASE_DIR, "saved_models")

# 训练配置
BATCH_SIZE = 32
NUM_EPOCHS = 100
LEARNING_RATE = 1e-5
DEVICE = "mps"  # 或 'cpu'

# 数据配置
NUM_CLASSES = 2
