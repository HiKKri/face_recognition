#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test_face_recognition
----------------------------------

Tests for `face_recognition` module.
"""


import unittest
import os
import numpy as np
from click.testing import CliRunner
from PIL import Image, ImageDraw
import sys

from face_recognition import api
from face_recognition import face_recognition_cli
from face_recognition import face_detection_cli
import face_recognition_models

class Test_face_recognition(unittest.TestCase):

    def test_01(self):
        print(sys.executable) #E:\www\devCoding\py\my_face_recognition_project\venv_py312\Scripts\python.exe
        print(sys.path)# ['E:\\www\\devCoding\\py\\my_face_recognition_project\\tests', 'D:\\java\\PyCharm\\PyCharm 2024.2.3\\plugins\\python-ce\\helpers\\pydev', 'D:\\java\\PyCharm\\PyCharm 2024.2.3\\plugins\\python-ce\\helpers\\third_party\\thriftpy', 'D:\\java\\PyCharm\\PyCharm 2024.2.3\\plugins\\python-ce\\helpers\\pydev', 'D:\\java\\PyCharm\\PyCharm 2024.2.3\\plugins\\python-ce\\helpers\\pycharm', 'E:\\www\\devCoding\\py\\my_face_recognition_project', 'E:\\www\\devCoding\\py\\my_face_recognition_project\\venv_py312', 'D:\\java\\PyCharm\\PyCharm 2024.2.3\\plugins\\python-ce\\helpers\\pycharm_display', 'C:\\Users\\Administrator\\AppData\\Local\\JetBrains\\PyCharm2024.2\\cythonExtensions', 'D:\\java\\PyCharm\\PyCharm 2024.2.3\\plugins\\python\\helpers-pro\\pydevd_asyncio', 'E:\\www\\devCoding\\py\\my_face_recognition_project\\tests', 'E:\\www\\pythonDevNew12\\python312.zip', 'E:\\www\\pythonDevNew12\\DLLs', 'E:\\www\\pythonDevNew12\\Lib', 'E:\\www\\pythonDevNew12', 'E:\\www\\devCoding\\py\\my_face_recognition_project\\venv_py312\\Lib\\site-packages', 'D:\\java\\PyCharm\\PyCharm 2024.2.3\\plugins\\python-ce\\helpers\\pycharm_matplotlib_backend', 'D:\\java\\PyCharm\\PyCharm 2024.2.3\\plugins\\python-ce\\helpers\\pycharm_plotly_backend']
        print(os.path.dirname(face_recognition_models.__file__)) #E:\www\devCoding\py\my_face_recognition_project\venv_py312\Lib\site-packages\face_recognition_models


        # 检查 models 子目录是否存在
        # models_dir = os.path.join(os.path.dirname(face_recognition_models.__file__), "models")
        # print(os.listdir(models_dir))
        # 应包含：['shape_predictor_68_face_landmarks.dat', 'dlib_face_recognition_resnet_model_v1.dat', ...]

    #载入图片
    def test_load_image_file(self):
        # --- Add these lines to verify the environment ---
        print("--- In Test Script: Python Executable ---")
        print(sys.executable)
        print("--- In Test Script: Python Version ---")
        print(sys.version_info)
        # ----------------------------------------------
        img = api.load_image_file(os.path.join(os.path.dirname(__file__), 'test_images', 'obama.jpg'))
        self.assertEqual(img.shape, (1137, 910, 3))

    # 载入32位图片
    def test_load_image_file_32bit(self):
        img = api.load_image_file(os.path.join(os.path.dirname(__file__), 'test_images', '32bit.png'))
        self.assertEqual(img.shape, (1200, 626, 3))

    def test_raw_face_locations(self):
        img = api.load_image_file(os.path.join(os.path.dirname(__file__), 'test_images', 'obama.jpg'))
        print(f"Image shape: {img.shape}")
        print(f"Image dtype: {img.dtype}")
        print(f"Min pixel: {img.min()}, Max pixel: {img.max()}")
        detected_faces = api._raw_face_locations(img)

        self.assertEqual(len(detected_faces), 1)
        self.assertEqual(detected_faces[0].top(), 142)
        self.assertEqual(detected_faces[0].bottom(), 409)
    #该函数扫描输入的图像 img，并返回图像中检测到的所有人脸的边界框（bounding box）坐标
    #img: (必需) 输入的图像。通常是一个 numpy 数组 (ndarray)，代表 RGB 格式的图像。
    #number_of_times_to_upsample: (可选, 默认为 1)这个参数决定了在检测前对图像进行上采样（放大）的次数。上采样有助于检测较小的人脸/值为 0 意味着图像大小不变。值为 1 将使图像大小增加 2 倍（面积变为 4 倍）。值为 2 将使图像大小增加 4 倍（面积变为 16 倍），以此类推。
    #model: (可选, 默认为 'hog')  'hog': 使用方向梯度直方图 (Histogram of Oriented Gradients) 特征和支持向量机 (SVM) 分类器。这个模型速度较快，对光线变化不太敏感，但在 CPU 上运行效果好/'cnn': 使用卷积神经网络 (Convolutional Neural Network) 模型。这个模型更准确，尤其是对于姿态变化较大的人脸，但它需要 GPU 加速才能获得合理速度，否则会非常慢。
    def test_hog_raw_face_locations(self):
        img_path = os.path.join(os.path.dirname(__file__), 'test_images', 'obama.jpg')
        # 使用 PIL 打开图像，并确保它是一个标准的 RGB 模式
        with Image.open(img_path) as pil_img:
            # 转换为 RGB 模式（如果原图是 RGBA, L 等，会强制转换为 RGB）
            rgb_pil_img = pil_img.convert('RGB')

            # 转换为 numpy 数组
            img = np.array(rgb_pil_img)

        # 2. 验证图像数据
        print("Image dtype after conversion:", img.dtype)
        print("Image shape after conversion:", img.shape)
        print("Is contiguous?", img.flags['C_CONTIGUOUS'])  # 检查数组是否在内存中连续

        # 3. 确保数组是连续的（dlib 经常需要连续的内存布局）
        if not img.flags['C_CONTIGUOUS']:
            print("Converting to contiguous array...")
            img = np.ascontiguousarray(img)
        # 确保图像只有3个通道
        if img.shape[2] > 3:
            img = img[:, :, :3]
        # 4. 使用 face_recognition 的公共接口检测人脸
        # 注意：这里我们不再使用内部的 _raw_face_locations
        detected_faces = api.face_locations(img, model="hog")
        print(f"检测到 {len(detected_faces)} 张人脸")

        # 5. 创建一个 Pillow 图像对象以便绘制
        pil_image = Image.fromarray(img)
        draw = ImageDraw.Draw(pil_image)

        # 6. 遍历检测到的人脸并画出矩形框
        for face_location in detected_faces:
            top, right, bottom, left = face_location
            print(f"人脸位置: Top={top}, Right={right}, Bottom={bottom}, Left={left}")
            draw.rectangle(((left, top), (right, bottom)), outline="red", width=3)


        # 第二种
        # img = api.load_image_file(os.path.join(os.path.dirname(__file__), 'test_images', 'obama.jpg'))
        # # 确保是 8-bit uint8 格式
        # print("Image dtype:", img.dtype)  # 应输出: uint8
        # print("Image shape:", img.shape)  # 应输出: (height, width, 3) —— RGB
        # # 使用 raw_face_locations 获取人脸位置
        # detected_faces = api.face_locations(img, model="hog")
        # print(f"检测到 {len(detected_faces)} 张人脸")
        # # 创建一个 Pillow 图像对象以便绘制
        # pil_image = Image.fromarray(img)
        # draw = ImageDraw.Draw(pil_image)
        #
        # # 遍历检测到的人脸并画出矩形框
        # for rect in detected_faces:
        #     # 获取坐标
        #     top = rect.top()
        #     right = rect.right()
        #     bottom = rect.bottom()
        #     left = rect.left()
        #
        #     print(f"人脸位置: Top={top}, Right={right}, Bottom={bottom}, Left={left}")
        #
        #     # 在图像上绘制矩形框
        #     draw.rectangle(((left, top), (right, bottom)), outline="red", width=3)
        #
        # # 显示结果
        # pil_image.show()

        self.assertEqual(len(detected_faces), 1)
        self.assertAlmostEqual(detected_faces[0].rect.top(), 144, delta=25)
        self.assertAlmostEqual(detected_faces[0].rect.bottom(), 389, delta=25)

    def test_raw_face_locations_32bit_image(self):
        img = api.load_image_file(os.path.join(os.path.dirname(__file__), 'test_images', '32bit.png'))
        detected_faces = api._raw_face_locations(img)

        self.assertEqual(len(detected_faces), 1)
        self.assertEqual(detected_faces[0].top(), 290)
        self.assertEqual(detected_faces[0].bottom(), 558)

    def test_cnn_raw_face_locations_32bit_image(self):
        img = api.load_image_file(os.path.join(os.path.dirname(__file__), 'test_images', '32bit.png'))
        detected_faces = api._raw_face_locations(img, model="cnn")

        self.assertEqual(len(detected_faces), 1)
        self.assertAlmostEqual(detected_faces[0].rect.top(), 259, delta=25)
        self.assertAlmostEqual(detected_faces[0].rect.bottom(), 552, delta=25)

    def test_face_locations(self):
        img = api.load_image_file(os.path.join(os.path.dirname(__file__), 'test_images', 'obama.jpg'))
        detected_faces = api.face_locations(img)

        self.assertEqual(len(detected_faces), 1)
        self.assertEqual(detected_faces[0], (142, 617, 409, 349))

    def test_cnn_face_locations(self):
        img = api.load_image_file(os.path.join(os.path.dirname(__file__), 'test_images', 'obama.jpg'))
        detected_faces = api.face_locations(img, model="cnn")

        self.assertEqual(len(detected_faces), 1)
        self.assertAlmostEqual(detected_faces[0][0], 144, delta=25)
        self.assertAlmostEqual(detected_faces[0][1], 608, delta=25)
        self.assertAlmostEqual(detected_faces[0][2], 389, delta=25)
        self.assertAlmostEqual(detected_faces[0][3], 363, delta=25)

    def test_partial_face_locations(self):
        img = api.load_image_file(os.path.join(os.path.dirname(__file__), 'test_images', 'obama_partial_face.jpg'))
        detected_faces = api.face_locations(img)

        self.assertEqual(len(detected_faces), 1)
        self.assertEqual(detected_faces[0], (142, 191, 365, 0))

        img = api.load_image_file(os.path.join(os.path.dirname(__file__), 'test_images', 'obama_partial_face2.jpg'))
        detected_faces = api.face_locations(img)

        self.assertEqual(len(detected_faces), 1)
        self.assertEqual(detected_faces[0], (142, 551, 409, 349))

    def test_raw_face_locations_batched(self):
        img = api.load_image_file(os.path.join(os.path.dirname(__file__), 'test_images', 'obama.jpg'))
        images = [img, img, img]
        batched_detected_faces = api._raw_face_locations_batched(images, number_of_times_to_upsample=0)

        for detected_faces in batched_detected_faces:
            self.assertEqual(len(detected_faces), 1)
            self.assertEqual(detected_faces[0].rect.top(), 154)
            self.assertEqual(detected_faces[0].rect.bottom(), 390)

    def test_batched_face_locations(self):
        img = api.load_image_file(os.path.join(os.path.dirname(__file__), 'test_images', 'obama.jpg'))
        images = [img, img, img]

        batched_detected_faces = api.batch_face_locations(images, number_of_times_to_upsample=0)

        for detected_faces in batched_detected_faces:
            self.assertEqual(len(detected_faces), 1)
            self.assertEqual(detected_faces[0], (154, 611, 390, 375))

    def test_raw_face_landmarks(self):
        img = api.load_image_file(os.path.join(os.path.dirname(__file__), 'test_images', 'obama.jpg'))
        face_landmarks = api._raw_face_landmarks(img)
        example_landmark = face_landmarks[0].parts()[10]

        self.assertEqual(len(face_landmarks), 1)
        self.assertEqual(face_landmarks[0].num_parts, 68)
        self.assertEqual((example_landmark.x, example_landmark.y), (552, 399))

    def test_face_landmarks(self):
        img = api.load_image_file(os.path.join(os.path.dirname(__file__), 'test_images', 'obama.jpg'))
        face_landmarks = api.face_landmarks(img)

        self.assertEqual(
            set(face_landmarks[0].keys()),
            set(['chin', 'left_eyebrow', 'right_eyebrow', 'nose_bridge',
                 'nose_tip', 'left_eye', 'right_eye', 'top_lip',
                 'bottom_lip']))
        self.assertEqual(
            face_landmarks[0]['chin'],
            [(369, 220), (372, 254), (378, 289), (384, 322), (395, 353),
             (414, 382), (437, 407), (464, 424), (495, 428), (527, 420),
             (552, 399), (576, 372), (594, 344), (604, 314), (610, 282),
             (613, 250), (615, 219)])

    def test_face_landmarks_small_model(self):
        img = api.load_image_file(os.path.join(os.path.dirname(__file__), 'test_images', 'obama.jpg'))
        face_landmarks = api.face_landmarks(img, model="small")

        self.assertEqual(
            set(face_landmarks[0].keys()),
            set(['nose_tip', 'left_eye', 'right_eye']))
        self.assertEqual(face_landmarks[0]['nose_tip'], [(496, 295)])

    def test_face_encodings(self):
        img = api.load_image_file(os.path.join(os.path.dirname(__file__), 'test_images', 'obama.jpg'))
        encodings = api.face_encodings(img)

        self.assertEqual(len(encodings), 1)
        self.assertEqual(len(encodings[0]), 128)

    def test_face_encodings_large_model(self):
        img = api.load_image_file(os.path.join(os.path.dirname(__file__), 'test_images', 'obama.jpg'))
        encodings = api.face_encodings(img, model='large')

        self.assertEqual(len(encodings), 1)
        self.assertEqual(len(encodings[0]), 128)

    def test_face_distance(self):
        img_a1 = api.load_image_file(os.path.join(os.path.dirname(__file__), 'test_images', 'obama.jpg'))
        img_a2 = api.load_image_file(os.path.join(os.path.dirname(__file__), 'test_images', 'obama2.jpg'))
        img_a3 = api.load_image_file(os.path.join(os.path.dirname(__file__), 'test_images', 'obama3.jpg'))

        img_b1 = api.load_image_file(os.path.join(os.path.dirname(__file__), 'test_images', 'biden.jpg'))

        face_encoding_a1 = api.face_encodings(img_a1)[0]
        face_encoding_a2 = api.face_encodings(img_a2)[0]
        face_encoding_a3 = api.face_encodings(img_a3)[0]
        face_encoding_b1 = api.face_encodings(img_b1)[0]

        faces_to_compare = [
            face_encoding_a2,
            face_encoding_a3,
            face_encoding_b1]

        distance_results = api.face_distance(faces_to_compare, face_encoding_a1)

        # 0.6 is the default face distance match threshold. So we'll spot-check that the numbers returned
        # are above or below that based on if they should match (since the exact numbers could vary).
        self.assertEqual(type(distance_results), np.ndarray)
        self.assertLessEqual(distance_results[0], 0.6)
        self.assertLessEqual(distance_results[1], 0.6)
        self.assertGreater(distance_results[2], 0.6)

    def test_face_distance_empty_lists(self):
        img = api.load_image_file(os.path.join(os.path.dirname(__file__), 'test_images', 'biden.jpg'))
        face_encoding = api.face_encodings(img)[0]

        # empty python list
        faces_to_compare = []

        distance_results = api.face_distance(faces_to_compare, face_encoding)
        self.assertEqual(type(distance_results), np.ndarray)
        self.assertEqual(len(distance_results), 0)

        # empty numpy list
        faces_to_compare = np.array([])

        distance_results = api.face_distance(faces_to_compare, face_encoding)
        self.assertEqual(type(distance_results), np.ndarray)
        self.assertEqual(len(distance_results), 0)

    def test_compare_faces(self):
        img_a1 = api.load_image_file(os.path.join(os.path.dirname(__file__), 'test_images', 'obama.jpg'))
        img_a2 = api.load_image_file(os.path.join(os.path.dirname(__file__), 'test_images', 'obama2.jpg'))
        img_a3 = api.load_image_file(os.path.join(os.path.dirname(__file__), 'test_images', 'obama3.jpg'))

        img_b1 = api.load_image_file(os.path.join(os.path.dirname(__file__), 'test_images', 'biden.jpg'))

        face_encoding_a1 = api.face_encodings(img_a1)[0]
        face_encoding_a2 = api.face_encodings(img_a2)[0]
        face_encoding_a3 = api.face_encodings(img_a3)[0]
        face_encoding_b1 = api.face_encodings(img_b1)[0]

        faces_to_compare = [
            face_encoding_a2,
            face_encoding_a3,
            face_encoding_b1]

        match_results = api.compare_faces(faces_to_compare, face_encoding_a1)

        self.assertEqual(type(match_results), list)
        self.assertTrue(match_results[0])
        self.assertTrue(match_results[1])
        self.assertFalse(match_results[2])

    def test_compare_faces_empty_lists(self):
        img = api.load_image_file(os.path.join(os.path.dirname(__file__), 'test_images', 'biden.jpg'))
        face_encoding = api.face_encodings(img)[0]

        # empty python list
        faces_to_compare = []

        match_results = api.compare_faces(faces_to_compare, face_encoding)
        self.assertEqual(type(match_results), list)
        self.assertListEqual(match_results, [])

        # empty numpy list
        faces_to_compare = np.array([])

        match_results = api.compare_faces(faces_to_compare, face_encoding)
        self.assertEqual(type(match_results), list)
        self.assertListEqual(match_results, [])

    def test_command_line_interface_options(self):
        target_string = 'Show this message and exit.'
        runner = CliRunner()
        help_result = runner.invoke(face_recognition_cli.main, ['--help'])
        self.assertEqual(help_result.exit_code, 0)
        self.assertTrue(target_string in help_result.output)

    def test_command_line_interface(self):
        target_string = 'obama.jpg,obama'
        runner = CliRunner()
        image_folder = os.path.join(os.path.dirname(__file__), 'test_images')
        image_file = os.path.join(os.path.dirname(__file__), 'test_images', 'obama.jpg')

        result = runner.invoke(face_recognition_cli.main, args=[image_folder, image_file])

        self.assertEqual(result.exit_code, 0)
        self.assertTrue(target_string in result.output)

    def test_command_line_interface_big_image(self):
        target_string = 'obama3.jpg,obama'
        runner = CliRunner()
        image_folder = os.path.join(os.path.dirname(__file__), 'test_images')
        image_file = os.path.join(os.path.dirname(__file__), 'test_images', 'obama3.jpg')

        result = runner.invoke(face_recognition_cli.main, args=[image_folder, image_file])

        self.assertEqual(result.exit_code, 0)
        self.assertTrue(target_string in result.output)

    def test_command_line_interface_tolerance(self):
        target_string = 'obama.jpg,obama'
        runner = CliRunner()
        image_folder = os.path.join(os.path.dirname(__file__), 'test_images')
        image_file = os.path.join(os.path.dirname(__file__), 'test_images', 'obama.jpg')

        result = runner.invoke(face_recognition_cli.main, args=[image_folder, image_file, "--tolerance", "0.55"])

        self.assertEqual(result.exit_code, 0)
        self.assertTrue(target_string in result.output)

    def test_command_line_interface_show_distance(self):
        target_string = 'obama.jpg,obama,0.0'
        runner = CliRunner()
        image_folder = os.path.join(os.path.dirname(__file__), 'test_images')
        image_file = os.path.join(os.path.dirname(__file__), 'test_images', 'obama.jpg')

        result = runner.invoke(face_recognition_cli.main, args=[image_folder, image_file, "--show-distance", "1"])

        self.assertEqual(result.exit_code, 0)
        self.assertTrue(target_string in result.output)

    def test_fd_command_line_interface_options(self):
        target_string = 'Show this message and exit.'
        runner = CliRunner()
        help_result = runner.invoke(face_detection_cli.main, ['--help'])
        self.assertEqual(help_result.exit_code, 0)
        self.assertTrue(target_string in help_result.output)

    def test_fd_command_line_interface(self):
        runner = CliRunner()
        image_file = os.path.join(os.path.dirname(__file__), 'test_images', 'obama.jpg')

        result = runner.invoke(face_detection_cli.main, args=[image_file])
        self.assertEqual(result.exit_code, 0)
        parts = result.output.split(",")
        self.assertTrue("obama.jpg" in parts[0])
        self.assertEqual(len(parts), 5)

    def test_fd_command_line_interface_folder(self):
        runner = CliRunner()
        image_file = os.path.join(os.path.dirname(__file__), 'test_images')

        result = runner.invoke(face_detection_cli.main, args=[image_file])
        self.assertEqual(result.exit_code, 0)
        self.assertTrue("obama_partial_face2.jpg" in result.output)
        self.assertTrue("obama.jpg" in result.output)
        self.assertTrue("obama2.jpg" in result.output)
        self.assertTrue("obama3.jpg" in result.output)
        self.assertTrue("biden.jpg" in result.output)

    def test_fd_command_line_interface_hog_model(self):
        target_string = 'obama.jpg'
        runner = CliRunner()
        image_file = os.path.join(os.path.dirname(__file__), 'test_images', 'obama.jpg')

        result = runner.invoke(face_detection_cli.main, args=[image_file, "--model", "hog"])
        self.assertEqual(result.exit_code, 0)
        self.assertTrue(target_string in result.output)

    def test_fd_command_line_interface_cnn_model(self):
        target_string = 'obama.jpg'
        runner = CliRunner()
        image_file = os.path.join(os.path.dirname(__file__), 'test_images', 'obama.jpg')

        result = runner.invoke(face_detection_cli.main, args=[image_file, "--model", "cnn"])
        self.assertEqual(result.exit_code, 0)
        self.assertTrue(target_string in result.output)
