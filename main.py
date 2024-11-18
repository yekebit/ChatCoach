"""Pose trainer main script."""

import argparse
import os
import  time
import sys
import numpy as np
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import QApplication
import threading
from parse import parse_sequence, load_ps
from evaluate import evaluate_pose
from MainWindows import VideoTextWindow
import sys
from PyQt5 import QtWidgets

def main():


    app = QApplication(sys.argv)
    window = VideoTextWindow()
    window.show()

        #等待一秒
    time.sleep(1000)
        # if window.video_input != "" and window.exercise != "":
        #     print('processing video file...')
        #     input_video_path = window.video_input
        #     video = os.path.basename(input_video_path)
        #     #
        #     #     # Run OpenPose on the video, and write a folder of JSON pose keypoints to a folder in
        #     #     # the repository root folder with the same name as the input video.
        #     output_path = os.path.join('..', os.path.splitext(video)[0])
        #     openpose_path = os.path.join('bin', 'OpenPoseDemo.exe')
        #
        #
        #     output = os.path.join('..',input_video_path.split('.')[0]+'_output.avi')
        #
        #
        #     os.chdir('openpose')
        #     subprocess.call([openpose_path,
        #                         # Use the COCO model since it outputs the keypoint format pose trainer is expecting.
        #                     '--model_pose', 'COCO',
        #                         # Use lower resolution for CPU only machines.
        #                         # If you're running the GPU version of OpenPose, or want to wait longer
        #                         # for higher quality, you can remove this line.
        #                     '--net_resolution', '-1x176',
        #                     '--video', input_video_path,
        #                     '--write_video', output,
        #                     '--write_json', output_path])
        #     window.load_video(output)
        #     print(output_path)
        #         # Parse the pose JSON in the output path, and write it as a .npy file to the repository root folder.
        #     parse_sequence(output_path, '..')
        #         # Load the .npy pose sequence and evaluate the pose as the specified exercise.
        #     pose_seq = load_ps(os.path.join('..', os.path.splitext(video)[0] + '.npy'))
        #
        #     messege = evaluate_pose(pose_seq, window.exercise)
        #     print(messege)
        #     window.load_video(output)
        #     window.set_text(messege)
        #     sys.exit(app.exec_())
        #
        # else:
        #     print('No video file specified.')
        #
        #




    # # Extract pose JSON files for every video in the input folder.
    # elif args.mode == 'batch_json':
    #     # read filenames from the videos directory
    #     videos = os.listdir(args.input_folder)
    #
    #     os.chdir('openpose')
    #
    #     for video in videos:
    #         print('processing video file:' + video)
    #         video_path = os.path.join('..', args.input_folder, video)
    #         output_path = os.path.join('..', args.output_folder, os.path.splitext(video)[0])
    #         openpose_path = os.path.join('bin', 'OpenPoseDemo.exe')
    #         subprocess.call([openpose_path,
    #                         # Use the COCO model since it outputs the keypoint format pose trainer is expecting.
    #                         '--model_pose', 'COCO',
    #                         # Use lower resolution for CPU only machines.
    #                         # If you're running the GPU version of OpenPose, or want to wait longer
    #                         # for higher quality, you can remove this line.
    #                         '--net_resolution', '-1x176',
    #                         '--video', video_path,
    #                         '--write_json', output_path])
    #
    # # Evaluate the .npy file as a pose sequence for the specified exercise.
    # elif args.mode == 'evaluate_npy':
    #     if args.file:
    #         pose_seq = load_ps(args.file)
    #         (correct, feedback) = evaluate_pose(pose_seq, args.exercise)
    #         if correct:
    #             print('Exercise performed correctly:')
    #         else:
    #             print('Exercise could be improved:')
    #         print(feedback)
    #     else:
    #         print('No npy file specified.')
    #         return
    #
    # else:
    #     print('Unrecognized mode option.')
    #     return




if __name__ == "__main__":
    main()
