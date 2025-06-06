﻿# 人脸相似度检索系统

## 项目特色

本项目是一个基于PyQt5和深度学习技术的人脸相似度检索工具，旨在帮助用户快速检索与目标图片或视频中人脸相似的内容。以下是项目的主要特色：

1. **直观的用户界面**
   - 提供拖放文件的功能，支持批量上传图片（`.jpg`, `.jpeg`, `.png`）和视频（`.mp4`, `.avi`）。
   - 检索结果以表格形式展示，包含匹配分数、文件路径、媒体预览以及操作按钮。

2. **高效的人脸嵌入向量生成**
   - 使用`insightface`模型提取人脸特征，生成高质量的嵌入向量。
   - 支持对图片和视频中的人脸进行检测和特征提取。

3. **强大的相似度检索功能**
   - 基于`FAISS`库构建高效的向量索引，支持快速检索。
   - 用户可以设置`Top-K`和`MinScore`参数，灵活调整检索结果的数量和质量。

4. **多线程处理**
   - 文件处理和嵌入向量生成在后台线程中运行，避免阻塞主线程，提升用户体验。

5. **错误处理与提示**
   - 对不支持的文件类型、无法加载的文件等异常情况提供友好的错误提示。



## 如何使用

### 环境依赖

- Python 3.x
- PyQt5
- OpenCV (`cv2`)
- FAISS
- InsightFace
- Ultralytics YOLO

安装依赖：
```bash
pip install PyQt5 opencv-python faiss-cpu insightface ultralytics
```


### 运行程序

克隆项目后，运行以下命令启动程序：
```bash
python ui/file_drop_widget.py
```


### 使用步骤

#### 1. 导入数据
- 打开“导入数据”Tab页。
- 将图片或视频文件拖放到指定区域，或点击“浏览选择”按钮手动选择文件。
- 点击“确认上传”按钮，开始生成嵌入向量并保存到数据库中。
- 处理进度会实时显示在进度条中。

#### 2. 检索数据
- 打开“检索数据”Tab页。
- 将待检索的图片或视频文件拖放到指定区域，或点击“浏览选择”按钮手动选择文件。
- 设置检索参数：
  - `Top-K`：返回最相似的前K个结果。
  - `MinScore`：设置最低匹配分数阈值。
- 点击“开始检索”按钮，系统会根据嵌入向量数据库返回匹配结果。
- 检索结果以表格形式展示，支持查看图片缩略图或播放视频。



## 技术实现

### 核心模块

1. **文件拖放与处理**
   - 使用`QLabel`实现文件拖放区域，支持拖放事件（`dragEnterEvent`和`dropEvent`）。
   - 文件路径通过信号槽机制传递给处理函数。

2. **多线程处理**
   - 使用`QThread`和`Worker`类实现文件处理的异步操作，避免阻塞主线程。
   - 进度条实时更新处理进度。

3. **人脸检测与特征提取**
   - 使用`insightface`模型提取人脸特征，生成嵌入向量。
   - 视频文件通过`OpenCV`抽取关键帧，并调用`YOLO`模型进行人脸检测。

4. **向量索引与检索**
   - 使用`FAISS`库构建高效的向量索引，支持快速检索。
   - 检索结果通过`core.search`模块返回，包含匹配分数和文件路径。

5. **媒体处理**
   - 图片加载：使用`QPixmap`加载并显示图片。
   - 视频缩略图抽取：使用`OpenCV`读取视频的第一帧并转换为`QPixmap`。



## 示例截图

### 准备数据
![准备数据](images/准备数据.png)

准备图片或者视频数据放置指定文件夹

### 选择文件夹
![选择文件夹](images/选择文件夹.png)

拖拽或者点击浏览选择文件夹或者文件

### 上传完成
![上传完成](images/上传完成.png)

点击开始上传, 后台自动处理, 进度条会实时更新

### 检索
![检索](images/检索.png)

选择图片或者视频进行相似度检索, 可以设置Top-K和MinScore参数，灵活调整检索结果的数量和质量

## 注意事项

1. **支持的文件类型**
   - 图片：`.jpg`, `.jpeg`, `.png`
   - 视频：`.mp4`, `.avi`
   - 不支持的文件类型会显示警告信息。

2. **性能优化**
   - 对于大规模数据集，建议在后台服务器上部署嵌入向量生成和检索服务，以提高效率。

3. **错误排查**
   - 如果遇到文件无法加载或检索失败，请检查文件路径是否正确以及文件格式是否支持。



## 贡献指南

欢迎提交Issue或Pull Request！如果您对项目有任何改进建议或发现了Bug，请随时联系我们。



## 许可证

本项目采用MIT许可证。详细信息请参阅[LICENSE](LICENSE)文件。
