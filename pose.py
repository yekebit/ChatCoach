import numpy as np
import math

class PoseSequence:
    def __init__(self, sequence):
        self.poses = []
        for parts in sequence:#对于一个帧
            self.poses.append(Pose(parts))
        
        # normalize poses based on the average torso pixel length 归一化 这部分代码计算所有姿态的平均躯干长度。
        torso_lengths = np.array([Part.dist(pose.neck, pose.lhip) for pose in self.poses if pose.neck.exists and pose.lhip.exists] +
                                 [Part.dist(pose.neck, pose.rhip) for pose in self.poses if pose.neck.exists and pose.rhip.exists])
        mean_torso = np.mean(torso_lengths)

        if not math.isnan(mean_torso):
            for pose in self.poses:
                for attr, part in pose:
                    setattr(pose, attr, part / mean_torso)
        else:
            print("Warning: mean_torso is NaN, skipping normalization.")



class Pose:
    PART_NAMES = ['nose', 'neck',  'rshoulder', 'relbow', 'rwrist', 'lshoulder', 'lelbow', 'lwrist', 'rhip', 'rknee', 'rankle', 'lhip', 'lknee', 'lankle', 'reye', 'leye', 'rear', 'lear']
    # nose - 鼻子
    # neck - 脖子
    # rshoulder - 右肩
    # relbow - 右肘
    # rwrist - 右腕
    # lshoulder - 左肩
    # lelbow - 左肘
    # lwrist - 左腕
    # rhip - 右髋
    # rknee - 右膝
    # rankle - 右踝
    # lhip - 左髋
    # lknee - 左膝
    # lankle - 左踝
    # reye - 右眼
    # leye - 左眼
    # rear - 右耳
    # lear - 左耳

    def __init__(self, parts):
        """Construct a pose for one frame, given an array of parts

        Arguments:
            parts - 18 * 3 ndarray of x, y, confidence values
        """
        for name, vals in zip(self.PART_NAMES, parts):
            setattr(self, name, Part(vals))
    
    def __iter__(self):
        for attr, value in self.__dict__.items():
            yield attr, value
    
    def __str__(self):
        out = ""
        for name in self.PART_NAMES:
            _ = "{}: {},{}".format(name, getattr(self, name).x, getattr(self, name).x)
            out = out + _ +"\n"
        return out
    
    def print(self, parts):
        out = ""
        for name in parts:
            if not name in self.PART_NAMES:
                raise NameError(name)
            _ = "{}: {},{}".format(name, getattr(self, name).x, getattr(self, name).x)
            out = out + _ +"\n"
        return out

class Part:
    def __init__(self, vals):
        self.x = vals[0]
        self.y = vals[1]
        self.c = vals[2]  #置信度
        self.exists = self.c != 0.0

    def __floordiv__(self, scalar):
        __truediv__(self, scalar)

    def __truediv__(self, scalar):
        return Part([self.x / scalar, self.y / scalar, self.c])

    @staticmethod
    def dist(part1, part2):
        return np.sqrt(np.square(part1.x - part2.x) + np.square(part1.y - part2.y))