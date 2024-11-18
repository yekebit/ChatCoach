import argparse
import glob
import json
import numpy as np
import os

from pose import Pose, Part, PoseSequence
from pprint import pprint


def main():

    parser = argparse.ArgumentParser(description='Pose Trainer Parser')

    # parser.add_argument('--video', type=str, default='bandicam2024-10-1611-55-44-849.mp4', help='input folder for videos')
    parser.add_argument('--input_folder', type=str, default='poses', help='input folder for json files')
    parser.add_argument('--output_folder', type=str, default='poses_compressed', help='output folder for npy files')
    
    args = parser.parse_args()

    video_paths = glob.glob(os.path.join(args.input_folder, '*'))
    # print("video_paths",video_paths)
    video_paths = sorted(video_paths)

    # Get all the json sequences for each video
    all_ps = []
    for video_path in video_paths:
        all_ps.append(parse_sequence(video_path, args.output_folder))
    return video_paths, all_ps


def parse_sequence(json_folder, output_folder):
    """Parse a sequence of OpenPose JSON frames and saves a corresponding numpy file.

    Args:
        json_folder: path to the folder containing OpenPose JSON for one video.
        output_folder: path to save the numpy array files of keypoints.

    """
    json_files = glob.glob(os.path.join(json_folder, '*.json'))
    json_files = sorted(json_files)

    num_frames = len(json_files)
    all_keypoints = np.zeros((num_frames, 18, 3))
    for i in range(num_frames):
        with open(json_files[i]) as f:
            json_obj = json.load(f)
            if 'people' in json_obj and len(json_obj['people']) > 0:
                keypoints = np.array(json_obj['people'][0]['pose_keypoints_2d'])
                indices = np.arange(2, 54, 3)  # 从2开始，因为索引从0开始计数

                # 使用布尔索引来检查这些位置的元素是否小于0.9，并置为0
                keypoints[indices[keypoints[indices] < 0.5]] = 0
                all_keypoints[i] = keypoints.reshape((18, 3))

    
    output_dir = os.path.join(output_folder, os.path.basename(json_folder))
    output_dir = "./pose-trainer/"+output_dir
    np.save(output_dir, all_keypoints)


def load_ps(filename):
    """Load a PoseSequence object from a given numpy file.

    Args:
        filename: file name of the numpy file containing keypoints.
    
    Returns:
        PoseSequence object with normalized joint keypoints.
    """
    all_keypoints = np.load(filename)
    #这里对于neck进行处理，对于没有neck的帧，使用leftshoulder加上rightshoulder来进行平均
    for i in range(all_keypoints.shape[0]):
        # 获取第3个关节和第6个关节的坐标
        joint_3 = all_keypoints[i, 2, :]  # 第3个关节
        joint_6 = all_keypoints[i, 5, :]  # 第6个关节，因为索引从0开始，所以是5

        # 计算平均值
        new_joint_2 = (joint_3 + joint_6) / 2

        # 更新第2个关节的坐标
        all_keypoints[i, 1, :] = new_joint_2

    pose_seq = PoseSequence(all_keypoints)
    for pose in pose_seq.poses:
        if_frame_valid = False
        for name in ['nose', 'neck',  'rshoulder', 'relbow', 'rwrist', 'lshoulder', 'lelbow', 'lwrist', 'rhip', 'rknee', 'rankle', 'lhip', 'lknee', 'lankle', 'reye', 'leye', 'rear', 'lear']:
            if getattr(pose, name).c != 0:
                if_frame_valid = True
        if if_frame_valid == False:#去掉这帧
            pose_seq.poses.remove(pose)
    return  pose_seq
    # return PoseSequence(all_keypoints)


if __name__ == '__main__':
    main()
