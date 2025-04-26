import os
import sys
import cv2

from core.embedding import gen_embedding
from core.search import search_function

from PyQt5.QtWidgets import *
from PyQt5.QtCore import Qt, pyqtSignal, QObject, QThread, QUrl
from PyQt5.QtGui import QDragEnterEvent, QDropEvent, QPixmap, QDesktopServices, QImage

from settings import ALLOWED_IMG_TYPES, ALLOWED_VIDEO_TYPES


class FileDropWidget(QLabel):
    filesDropped = pyqtSignal(list)

    def __init__(self, text, parent=None):
        super().__init__(text, parent)
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet(""" QLabel { border: 2px dashed #aaa; border-radius: 10px; padding: 20px; } """)
        self.setAcceptDrops(True)

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent):
        paths = [url.toLocalFile() for url in event.mimeData().urls()]
        self.filesDropped.emit(paths)
        self.setText("\n".join(paths))


class Worker(QObject):
    progress = pyqtSignal(int, int)  # current, total
    finished = pyqtSignal()

    def __init__(self, gen):
        super().__init__()
        self.gen = gen

    def run(self):
        for current, total in self.gen:
            self.progress.emit(current, total)
        self.finished.emit()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("人脸相似度检索")
        self.setGeometry(100, 100, 800, 600)

        # 创建Tab容器
        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)

        # 初始化两个Tab
        self.tab1 = QWidget()
        self.tab2 = QWidget()
        self.tabs.addTab(self.tab2, "检索数据")
        self.tabs.addTab(self.tab1, "导入数据")

        # 初始化UI
        self.init_tab1()
        self.init_tab2()

    def init_tab1(self):
        layout = QVBoxLayout()

        # 文件拖放区域
        self.drop_area1 = FileDropWidget("拖放文件夹或文件到这里\n或点击按钮选择")
        btn_browse = QPushButton("浏览选择")
        btn_browse.clicked.connect(self.select_files_tab1)

        # 确认按钮和进度条
        self.btn_confirm1 = QPushButton("确认上传")
        self.progress1 = QProgressBar()
        self.progress1.setAlignment(Qt.AlignCenter)

        # 布局
        layout.addWidget(self.drop_area1)
        layout.addWidget(btn_browse)
        layout.addWidget(self.btn_confirm1)
        layout.addWidget(self.progress1)
        self.tab1.setLayout(layout)

        # 信号连接
        self.drop_area1.filesDropped.connect(self.handle_files_tab1)
        self.btn_confirm1.clicked.connect(self.start_processing_tab1)

    def init_tab2(self):
        layout = QVBoxLayout()

        # 文件拖放区域
        self.drop_area2 = FileDropWidget("拖放待检索文件到这里\n或点击按钮选择")
        btn_browse = QPushButton("浏览选择")
        btn_browse.clicked.connect(self.select_file_tab2)

        # 参数输入
        param_layout = QHBoxLayout()
        self.top_k = QSpinBox()
        self.top_k.setRange(1, 1000)
        self.top_k.setValue(100)
        self.min_score = QSpinBox()
        self.min_score.setRange(0, 1000)
        self.min_score.setValue(200)

        param_layout.addWidget(QLabel("Top-K:"))
        param_layout.addWidget(self.top_k)
        param_layout.addWidget(QLabel("MinScore:"))
        param_layout.addWidget(self.min_score)

        # 结果展示
        self.results_table = QTableWidget()
        self.results_table.setColumnCount(4)
        self.results_table.setHorizontalHeaderLabels(["分数", "文件路径", "文件内容", "操作"])

        # 确认按钮
        self.btn_confirm2 = QPushButton("开始检索")

        # 布局
        layout.addWidget(self.drop_area2)
        layout.addWidget(btn_browse)
        layout.addLayout(param_layout)
        layout.addWidget(self.btn_confirm2)
        layout.addWidget(self.results_table)
        self.tab2.setLayout(layout)

        # 信号连接
        self.drop_area2.filesDropped.connect(self.handle_file_tab2)
        self.btn_confirm2.clicked.connect(self.start_search_tab2)

    # Tab1相关方法
    def select_files_tab1(self):
        files, _ = QFileDialog.getOpenFileNames(self, "选择文件", "", "媒体文件 (*.jpg *.jpeg *.png *.mp4 *.avi)")
        if files:
            self.drop_area1.setText("\n".join(files))

    def handle_files_tab1(self, paths):
        self.tab1_files = paths

    def start_processing_tab1(self):
        if not hasattr(self, 'tab1_files'):
            QMessageBox.warning(self, "警告", "请先选择文件或文件夹")
            return

        # 创建线程处理生成器
        self.thread = QThread()
        self.worker = Worker(gen_embedding(self.tab1_files))  # 替换为实际生成器
        self.worker.moveToThread(self.thread)

        # 连接信号
        self.worker.progress.connect(self.update_progress_tab1)
        self.worker.finished.connect(self.thread.quit)
        self.thread.started.connect(self.worker.run)
        self.thread.start()

        # 禁用按钮
        self.btn_confirm1.setEnabled(False)
        self.thread.finished.connect(lambda: self.btn_confirm1.setEnabled(True))

    def update_progress_tab1(self, current, total):
        self.progress1.setMaximum(total)
        self.progress1.setValue(current)

    # Tab2相关方法
    def select_file_tab2(self):
        file, _ = QFileDialog.getOpenFileName(self, "选择文件", "", "媒体文件 (*.jpg *.jpeg *.png *.mp4 *.avi)")
        if file:
            self.drop_area2.setText(file)

    def handle_file_tab2(self, paths):
        if paths:
            self.tab2_file = paths[0]
            self.drop_area2.setText(self.tab2_file)

    def start_search_tab2(self):
        if not hasattr(self, 'tab2_file'):
            QMessageBox.warning(self, "警告", "请先选择待检索文件")
            return

        # 执行搜索回调
        results = search_function(  # 替换为实际搜索函数
            self.tab2_file,
            self.top_k.value(),
            self.min_score.value()
        )

        # 清空结果表格
        self.results_table.setRowCount(len(results))

        for row, (score, path) in enumerate(results):
            # 更新分数和文件路径
            self.results_table.setRowHeight(row, 200)
            self.results_table.setColumnWidth(1, 300)
            self.results_table.setItem(row, 0, QTableWidgetItem(str(score)))
            self.results_table.setItem(row, 1, QTableWidgetItem(path))

            label, media_type = self.get_media_row_label(path)
            self.results_table.setCellWidget(row, 2, label)

            media_button = QPushButton(media_type)
            media_button.clicked.connect(lambda _, p=path: self.play_media(p))
            self.results_table.setCellWidget(row, 3, media_button)

    def get_media_row_label(self, path):
        # 检查文件是否存在
        if not os.path.exists(path):
            error_label = QLabel("文件不存在")
            error_label.setStyleSheet("color: red;")
            return error_label, "未知"

        # 根据文件类型展示内容
        if path.lower().endswith(ALLOWED_IMG_TYPES):
            # 显示图片
            pixmap = QPixmap(path)
            if pixmap.isNull():
                error_label = QLabel("无法加载图片")
                error_label.setStyleSheet("color: red;")
                return error_label, "查看图片"
            else:
                label = QLabel()
                label.setPixmap(pixmap.scaled(200, 200, Qt.KeepAspectRatio, Qt.SmoothTransformation))
                label.setAlignment(Qt.AlignCenter)
                label.setScaledContents(True)
                return label, "查看图片"

        elif path.lower().endswith(ALLOWED_VIDEO_TYPES):
            # 使用 OpenCV 抽取视频缩略图
            thumbnail = self.extract_video_thumbnail(path)
            if thumbnail is None:
                error_label = QLabel("无法加载视频缩略图")
                error_label.setStyleSheet("color: red;")
                return error_label, '播放视频'
            else:
                label = QLabel()
                label.setPixmap(thumbnail.scaled(200, 200, Qt.KeepAspectRatio, Qt.SmoothTransformation))
                label.setAlignment(Qt.AlignCenter)
                label.setScaledContents(True)
                return label, '播放视频'

        else:
            # 不支持的文件类型
            unsupported_label = QLabel("不支持的文件类型")
            unsupported_label.setStyleSheet("color: orange;")
            return unsupported_label, "未知"

    def extract_video_thumbnail(self, path):
        """
        使用 OpenCV 抽取视频的第一帧作为缩略图
        """
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            return None

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.set(cv2.CAP_PROP_POS_FRAMES, min(100, total_frames))

        ret, frame = cap.read()
        cap.release()

        if not ret:
            return None

        # 将 BGR 格式的帧转换为 RGB 格式
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 将帧转换为 QPixmap
        h, w, ch = frame_rgb.shape
        bytes_per_line = ch * w
        q_image = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)

        return pixmap

    def play_media(self, path):
        """
        使用系统默认的视频播放器播放视频
        """
        url = QUrl.fromLocalFile(path)
        if not QDesktopServices.openUrl(url):
            QMessageBox.warning(self, "警告", "无法打开视频/图片文件")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
