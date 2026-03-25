import face_recognition_models
import os

# 获取模型包所在目录
model_dir = os.path.dirname(face_recognition_models.__file__)
print("face_recognition_models path:", model_dir)

# 列出该目录下的 .dat 文件
for f in os.listdir(model_dir):
    if f.endswith('.dat'):
        print("Found model:", f)
