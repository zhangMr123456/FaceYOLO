import os.path
from pathlib import Path
from typing import List

import cv2
import numpy as np
from PIL import Image
from insightface.app.common import Face

from core.database import write_embedding, get_exist_keys
from core.face_analysis import buffalo_model
from core.utils import get_files_from_list, exception_print
from core.yolo import detect_faces_results
from settings import FILE_MAX_BYTE_CNT, WRITE_BATCH_SIZE, ALLOWED_IMG_TYPES, ALLOWED_VIDEO_TYPES


def get_embeddings_by_media(file_path, img_types, video_types):
    """
    输入文件路径转为向量
    :param file_path: 文件路径
    :return:
    """
    if not os.path.isfile(file_path):
        return

    suffix = Path(file_path).suffix.lower()
    try:
        if suffix in img_types:
            return get_img_embeddings(file_path)
        elif suffix in video_types:
            image = extract_video_face_return_image(file_path)
            # 视频可能不存在人脸
            if image is None:
                return
            return get_img_embeddings(image)
    except Exception as e:
        raise Exception(f"file_path: {file_path}, suffix: {suffix}") from e


def get_img_embeddings(image):
    try:
        results: List[Face] = buffalo_model.get(image)
    except:
        image = Image.open(image)
        # 转换为三通道, 有的为四通道
        image = image.convert("RGB")
        results: List[Face] = buffalo_model.get(np.array(image))

    emb_list = []
    for res in results:
        emb_list.append(res["embedding"])
    return emb_list


def extract_video_face_return_image(video_path, fps=5, max_score=0.8) -> np.ndarray | None:
    cap = cv2.VideoCapture(video_path)
    frame_count, t_max_score, image = 0, 0, None
    while True:
        ret, frame = cap.read()
        if not ret: break
        if frame_count % int(cap.get(cv2.CAP_PROP_FPS) / fps) != 0:
            frame_count += 1
            continue
        else:
            frame_count += 1

        resized_frame = cv2.resize(frame, (640, 640))
        confidence_list = get_face_confidence(resized_frame)
        for confidence in confidence_list:
            # 如果当前比率大于最大值，则返回当前帧
            if confidence >= max_score:
                return frame

            if confidence <= t_max_score:
                continue
            else:
                image = frame
                t_max_score = confidence

    if image is not None:
        return image


def get_face_confidence(frame_buffer) -> List[float]:
    confidence_list = []
    results = detect_faces_results(frame_buffer)
    for result in results:
        summary = result.summary()
        if not summary:
            continue

        confidence_list.append(summary[0]["confidence"])

    return confidence_list


def extract_video_face_return_image1(video_path, fps=5, max_score=0.8, shape=(640, 640)) -> np.ndarray | None:
    # 初始化视频捕获
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"无法打开视频文件, video_path: {video_path}")
        return None

    # 获取视频基本信息
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = max(1, int(video_fps / fps))

    # 预分配缓冲区
    frame_buffer = np.empty((shape[0], shape[1], 3), dtype=np.uint8)
    best_image, best_score = None, 0

    # 使用多尺度采样策略
    sampling_strategy = [
        (0, min(100, total_frames // 10), 1),  # 开头部分密集采样
        (total_frames - 100, total_frames, 1),  # 结尾部分密集采样
        (100, total_frames - 100, frame_interval),  # 中间部分常规采样
    ]

    for start, end, interval in sampling_strategy:
        if start >= end:
            continue

        # 跳转到起始帧
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)

        frame_cnt = start
        for frame_pos in range(start, end, 1):
            ret, frame = cap.read()
            if not ret:
                break
            else:
                frame_cnt += 1

            if frame_cnt % interval != 0:
                continue

            # 快速人脸检测预处理
            cv2.resize(frame, shape, frame_buffer)

            # 完整人脸检测
            confidence_list = get_face_confidence(frame_buffer)
            for confidence in confidence_list:
                # 达到阈值立即返回
                print(f"confidence: {confidence}, frame_cnt: {frame_cnt}")
                if confidence >= max_score:
                    cap.release()
                    return frame

                # 更新最佳人脸
                if confidence > best_score:
                    best_image = frame
                    best_score = confidence

    cap.release()
    return best_image if best_score > 0 else None


def check_file_size(file_path):
    if os.path.getsize(file_path) > FILE_MAX_BYTE_CNT:
        print(f"file_path: {file_path} too large, continue")
        return False
    return True


@exception_print
def gen_embedding(paths: List[str]):
    # 获取所有的文件列表
    exists_files = get_exist_keys()
    file_list = get_files_from_list(paths, exists_files)
    del exists_files
    file_size, emb_dict = len(file_list), {}
    for index, file_path in enumerate(file_list):
        # 检查文件大小
        if not check_file_size(file_path):
            yield index + 1, file_size
            continue

        # 写入数据库
        if len(emb_dict.keys()) > WRITE_BATCH_SIZE:
            write_embedding(emb_dict)
            emb_dict = {}

        try:
            embs = get_embeddings_by_media(file_path, ALLOWED_IMG_TYPES, ALLOWED_VIDEO_TYPES)
        except Exception as e:
            import traceback
            traceback.print_exc()
            traceback.print_stack()
            print(f"识别错误文件: {file_path}")
            yield index + 1, file_size
            continue

        if not embs:
            print(f"未识别到人脸: {file_path}")
            yield index + 1, file_size
            continue

        emb_dict[file_path] = embs
        for emb in embs:
            print(type(emb))
            print(f"file_path: {file_path}, emb: {emb}")
        yield index + 1, file_size

    if emb_dict:
        write_embedding(emb_dict)

    yield file_size, file_size


