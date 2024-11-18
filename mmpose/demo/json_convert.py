import json


def process_keypoints(kp_map, kp_array, kp_score, output_file):
	# 指定的关键点顺序
	keypoint_order = ['nose', 'neck', 'right_shoulder', 'right_elbow', 'right_wrist',
	                  'left_shoulder', 'left_elbow', 'left_wrist', 'right_hip', 'right_knee',
	                  'right_ankle', 'left_hip', 'left_knee', 'left_ankle', 'reye',
	                  'leye', 'right_ear', 'left_ear']

	# COCO 17 key-points list
	# 'keypoint_id2name': {0: 'nose', 1: 'left_eye', 2: 'right_eye', 3: 'left_ear', 4: 'right_ear',
	# 5: 'left_shoulder', 6: 'right_shoulder', 7: 'left_elbow', 8: 'right_elbow', 9: 'left_wrist',
	# 10: 'right_wrist', 11: 'left_hip', 12: 'right_hip', 13: 'left_knee', 14: 'right_knee',
	# 15: 'left_ankle', 16: 'right_ankle'},

	#
	# keypoint_mapping = {
	#     "nose": 0, "head_bottom": 1, "head_top": 2, "left_ear": 3,
	#     "right_ear": 4, "left_shoulder": 5, "right_shoulder": 6,
	#     "left_elbow": 7, "right_elbow": 8, "left_wrist": 9,
	#     "right_wrist": 10, "left_hip": 11, "right_hip": 12,
	#     "left_knee": 13, "right_knee": 14, "left_ankle": 15,
	#     "right_ankle": 16
	# }
	# 初始化重排后的 keypoints 和 scores
	reordered_keypoints_scores = []

	# 进行重排
	for keypoint_name in keypoint_order:
		if keypoint_name in kp_map:
			keypoint_id = kp_map[keypoint_name]
			if keypoint_id < len(kp_array[0]):
				x, y = kp_array[0][keypoint_id]
				score = kp_score[0][keypoint_id]
				reordered_keypoints_scores.append([x.item(), y.item(), score.item()])
			else:
				reordered_keypoints_scores.append([0, 0, 0])
		else:
			# 如果在 mapping 中没有找到，填充 [0, 0, 0]
			reordered_keypoints_scores.append([0, 0, 0])

	# 创建最终的 JSON 结构
	output_data = {
		"version": 1.3,
		"people": [{
			"person_id": [-1],
			"pose_keypoints_2d": [coord for point in reordered_keypoints_scores for coord in point]
		}]
	}

	# 保存为输出 JSON 文件
	with open(output_file, 'w') as f:
		json.dump(output_data, f, indent=4)
