import os


def get_dir_files(dir_path):
    for file_name in os.listdir(dir_path):
        file_path = os.path.join(dir_path, file_name)
        if os.path.isfile(file_path):
            yield file_path
        elif os.path.isdir(file_path):
            yield from get_dir_files(file_path)


def get_files_from_list(dir_list, exist_file_paths):
    file_list = []
    for obj_path in dir_list:
        if obj_path in exist_file_paths:
            continue
        if os.path.isfile(obj_path):
            file_list.append(obj_path)
        elif os.path.isdir(obj_path):
            for file_path in get_dir_files(obj_path):
                if file_path in exist_file_paths:
                    continue
                file_list.append(file_path)

    return file_list


def exception_print(func):
    def __exe__(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except:
            import traceback
            traceback.print_exc()
            traceback.print_stack()
            raise

    return __exe__


import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim


def remove_video_suffix(input_path, output_path, threshold=0.95, buffer_frames=10):
    """
    智能去除视频末尾重复内容
    :param input_path: 输入视频路径
    :param output_path: 输出视频路径
    :param threshold: 相似度阈值(0.8-0.95)
    :param buffer_frames: 安全保留帧数
    """
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise ValueError("无法打开视频文件")

    # 获取视频参数
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 初始化逆向分析
    prev_frame = None
    cut_position = total_frames
    found_cut = False
    step_size = max(1, int(fps))  # 每秒检测一次

    # 从末尾开始检测
    for pos in range(total_frames - 1, 0, -step_size):
        cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
        ret, current_frame = cap.read()
        if not ret:
            continue

        current_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)

        if prev_frame is not None:
            # 计算结构相似性
            similarity = ssim(prev_frame, current_gray,
                              win_size=3,
                              data_range=current_gray.max() - current_gray.min())

            if similarity < threshold:
                cut_position = min(pos + buffer_frames, total_frames)
                found_cut = True
                break

        prev_frame = current_gray.copy()

    if not found_cut:
        print("未检测到重复后缀")
        return

    # 重新定位到视频开头
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # 配置视频写入器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # 写入有效帧
    for _ in range(cut_position):
        ret, frame = cap.read()
        if ret:
            out.write(frame)
        else:
            break

    cap.release()
    out.release()
    print(f"视频处理完成，保留前 {cut_position} 帧（原始 {total_frames} 帧）")


# 使用示例
if __name__ == '__main__':
    remove_video_suffix(r"D:\手机\20240828\下载 (4).mp4", r"C:\Users\admin\Desktop\下载 (4).mp4", threshold=0.92)
