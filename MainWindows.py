import subprocess
import time
import sys
import os
from parse import parse_sequence, load_ps
from evaluate import evaluate_pose
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QTextEdit,
    QPushButton, QLabel, QComboBox, QFileDialog
)
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from PyQt5.QtMultimediaWidgets import QVideoWidget
from PyQt5.QtGui import QFont
from PyQt5.QtCore import QUrl
from terminal import build_topdown_command, build_bottomup_command
from moviepy.editor import VideoFileClip

class VideoTextWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.video_input = ""
        self.exercise = ""
        self.flag = False
        self.method = "bottom-up"
        self.detection_model = "yolov3"
        self.pose_model = "openpose"

        # Create a video player and video widget
        self.mediaPlayer = QMediaPlayer(None, QMediaPlayer.VideoSurface)
        self.videoWidget = QVideoWidget()

        # Create buttons
        self.selectVideoButton = QPushButton("选择文件")
        self.selectVideoButton.clicked.connect(self.select_video_file)

        self.confirmButton = QPushButton("确定")
        self.confirmButton.clicked.connect(self.confirm_selection)

        # Create play button
        self.playButton = QPushButton("播放")
        self.playButton.clicked.connect(self.play_video)

        # Create a text editor
        self.textEdit = QTextEdit()
        self.textEdit.setFont(QFont('Arial', 12))

        # Create a combo box for selecting items
        self.comboBox1 = QComboBox()
        self.comboBox1.addItems(["bottom-up", "top-down"])
        self.comboBox1.currentTextChanged.connect(self.update_model)

        self.comboBox2 = QComboBox()
        self.comboBox2.addItems(["yolov3", "hrnet", "yolact", "cascaderpn", "yolox", "rtmdet"])
        self.comboBox2.currentTextChanged.connect(self.update_detection_model)

        self.comboBox3 = QComboBox()
        self.comboBox3.addItems(["openpose","simcc", "yoloxpose", "edpose", "rtmpose", "rtmo"])
        self.comboBox3.currentTextChanged.connect(self.update_pose_model)

        self.comboBox4 = QComboBox()
        items = {
            "二头肌弯举": "bicep_curl",
            "前平举": "front_raise",
            "平板支撑": "elbow_plank",
            "杠铃深蹲": "barbell_squats",
            "侧平举（哑铃飞鸟）": "Dumbbell_bird",
            "坐姿推肩（和肩部推举区分开）": "shoulder_push",
            "杠铃卧推": "BarbellBenchPress",
            "俯身划船": "Lean_row"
        }
        self.comboBox4.addItems(items.keys())
        self.items = items

        # Set up layouts
        self.setWindowTitle("视频和文本展示")
        self.resize(1200, 800)

        mainLayout = QVBoxLayout()

        # Top layout for selection
        topLayout = QHBoxLayout()
        topLayout.addWidget(self.comboBox1)
        topLayout.addWidget(self.comboBox2)
        topLayout.addWidget(self.comboBox3)
        topLayout.addWidget(self.comboBox4)
        topLayout.addWidget(self.selectVideoButton)
        topLayout.addWidget(self.confirmButton)

        # Bottom layout for video and text
        bottomLayout = QHBoxLayout()

        videoLayout = QVBoxLayout()
        videoLayout.addWidget(self.videoWidget)
        videoLayout.addWidget(self.playButton)

        textLayout = QVBoxLayout()
        textLabel = QLabel("建议区域")
        textLabel.setAlignment(Qt.AlignCenter)
        textLayout.addWidget(textLabel)
        textLayout.addWidget(self.textEdit)

        bottomLayout.addLayout(videoLayout, 2)
        bottomLayout.addLayout(textLayout, 1)

        mainLayout.addLayout(topLayout)
        mainLayout.addLayout(bottomLayout)

        self.setLayout(mainLayout)

    def update_model(self, text):
        self.method = text

    def update_detection_model(self, text):
        self.detection_model = text

    def update_pose_model(self, text):
        self.pose_model = text

    def send_message_to_mainpy(self):
        self.flag = True

    def select_video_file(self):
        # Open file dialog to select video
        video_path, _ = QFileDialog.getOpenFileName(self, "选择视频文件", "", "Video Files (*.avi *.mp4 *.mkv)")
        self.video_input = video_path

    def confirm_selection(self):
        # Get selected item from combo box
        selected_key = self.comboBox4.currentText()
        selected_value = self.items[selected_key]
        self.exercise = selected_value

        if self.video_input != "" and self.exercise != "":
            print('processing video file...')
            input_video_path = self.video_input
            video = os.path.basename(input_video_path)
            output_path = os.path.join('..', os.path.splitext(video)[0])

            if self.method == "bottom-up" and self.pose_model == "openpose":
                openpose_path = os.path.join('bin', 'OpenPoseDemo.exe')
                output = os.path.join('..', input_video_path.split('.')[0] + '_output.avi')

                os.chdir('openpose')
                subprocess.call([openpose_path,
                                '--model_pose', 'COCO',
                                '--net_resolution', '-1x176',
                                '--video', input_video_path,
                                '--write_video', output,
                                '--write_json', output_path])
                self.load_video(output)
                self.set_text("正在获取建议，请稍等！")
                print(output_path)
            elif  self.method == "top-down":
                output_dir = input_video_path.split('.')[0]
                command = build_topdown_command(self.detection_model, self.pose_model, input_video_path, output_dir)
                subprocess.call(command)
                output = output_dir+"/"+output_dir.split('/')[-1] + '_output.mp4'
                print("video rst path:", output)
                clip = VideoFileClip(output)
                avi_file_path = output.split('.')[0] + '.avi'
                print("avi_file_path:", avi_file_path)
                clip.write_videofile(avi_file_path, codec='libx264')
                self.load_video(avi_file_path)
                self.set_text("正在获取建议，请稍等！")
                output_path = output_dir

            elif self.method == "bottom-up" and self.pose_model !="openpose":#只需要pose_model，不需要别的东西
                output_dir = input_video_path.split('.')[0]
                command = build_bottomup_command(self.pose_model, input_video_path, output_dir)
                subprocess.call(command)
                output = output_dir + "/" + output_dir.split('/')[-1] + '_output.mp4'
                print("video rst path:", output)
                clip = VideoFileClip(output)
                avi_file_path = output.split('.')[0] + '.avi'
                print("avi_file_path:", avi_file_path)
                clip.write_videofile(avi_file_path, codec='libx264')
                self.load_video(avi_file_path)
                self.set_text("正在获取建议，请稍等！")
                output_path = output_dir


            parse_sequence(output_path, '..')
            pose_seq = load_ps(os.path.splitext(video)[0] + '.npy')
            messege = evaluate_pose(pose_seq, self.exercise)
            print(messege)
            self.set_text(messege)
        else:
            print('No video file specified.')

    def set_text(self, prompt):
        self.textEdit.setText(prompt)

    def load_video(self, video_path):
        self.mediaPlayer.setMedia(QMediaContent(QUrl.fromLocalFile(video_path)))
        self.mediaPlayer.setVideoOutput(self.videoWidget)

    def play_video(self):
        self.mediaPlayer.play()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    self = VideoTextWindow()
    self.show()
    sys.exit(app.exec_())