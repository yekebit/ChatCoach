mmdet_dict = {
	"yolov3": {
		"config": "mmdetection/configs/yolo/yolov3_d53_8xb8-ms-608-273e_coco.py",
		"model": "checkpoint/mmdet/yolov3/yolov3_d53_320_273e_coco-421362b6.pth",
		"link": "https://download.openmmlab.com/mmdetection/v2.0/yolo/yolov3_d53_320_273e_coco/yolov3_d53_320_273e_coco-421362b6.pth"
	},
	"hrnet": {
		"config": "mmdetection/configs/hrnet/faster-rcnn_hrnetv2p-w18-1x_coco.py",
		"model": "checkpoint/mmdet/hrnet/faster_rcnn_hrnetv2p_w18_1x_coco_20200130-56651a6d.pth",
		"link": "https://download.openmmlab.com/mmdetection/v2.0/hrnet/faster_rcnn_hrnetv2p_w18_1x_coco/faster_rcnn_hrnetv2p_w18_1x_coco_20200130-56651a6d.pth"
	},
	"yolact": {
		"config": "mmdetection/configs/yolact/yolact_r50_1xb8-55e_coco.py",
		"model": "checkpoint/mmdet/yolact/yolact_r50_1x8_coco_20200908-f38d58df.pth",
		"link": "https://download.openmmlab.com/mmdetection/v2.0/yolact/yolact_r50_1x8_coco/yolact_r50_1x8_coco_20200908-f38d58df.pth"
	},
	"cascaderpn": {
		"config": "mmdetection/configs/cascade_rpn/cascade-rpn_faster-rcnn_r50-caffe_fpn_1x_coco.py",
		"model": "checkpoint/mmdet/cascade_rpn/crpn_faster_rcnn_r50_caffe_fpn_1x_coco-c8283cca.pth",
		"link": "https://download.openmmlab.com/mmdetection/v2.0/cascade_rpn/crpn_faster_rcnn_r50_caffe_fpn_1x_coco/crpn_faster_rcnn_r50_caffe_fpn_1x_coco-c8283cca.pth"
	},
	"yolox": {
		"config": "mmdetection/configs/yolox/yolox_tiny_8xb8-300e_coco.py",
		"model": "checkpoint/mmdet/yolox/yolox_tiny_8x8_300e_coco_20211124_171234-b4047906.pth",
		"link": "https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/yolox_pose/yoloxpose_tiny_4xb64-300e_coco-416-76eb44ca_20230829.pth"
	},
	"rtmdet": {
		"config": "mmdetection/configs/rtmdet/rtmdet_tiny_8xb32-300e_coco.py",
		"model": "checkpoint/mmdet/rtmdet/rtmdet_tiny_8xb32-300e_coco_20220902_112414-78e30dcc.pth",
		"link": "https://download.openmmlab.com/mmdetection/v3.0/rtmdet/rtmdet_tiny_8xb32-300e_coco/rtmdet_tiny_8xb32-300e_coco_20220902_112414-78e30dcc.pth"
	}
}

mmpose_dict = {
	"simcc": {
		"config": "mmpose/configs/body_2d_keypoint/simcc/coco/simcc_res50_8xb64-210e_coco-256x192.py",
		"model": "checkpoint/mmpose/simcc/simcc_res50_8xb64-210e_coco-256x192-8e0f5b59_20220919.pth",
		"link": "https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/simcc/coco/simcc_res50_8xb64-210e_coco-256x192-8e0f5b59_20220919.pth"
	},
	"yoloxpose": {
		"config": "mmpose/configs/body_2d_keypoint/yoloxpose/coco/yoloxpose_tiny_4xb64-300e_coco-416.py",
		"model": "checkpoint/mmpose/yoloxpose/yoloxpose_tiny_4xb64-300e_coco-416-76eb44ca_20230829.pth",
		"link": "https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/yolox_pose/yoloxpose_tiny_4xb64-300e_coco-416-76eb44ca_20230829.pth"
	},
	"edpose": {
		"config": "mmpose/configs/body_2d_keypoint/edpose/coco/edpose_res50_8xb2-50e_coco-800x1333.py",
		"model": "checkpoint/mmpose/edpose/edpose_res50_coco_3rdparty.pth",
		"link": "https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/edpose/coco/edpose_res50_coco_3rdparty.pth"
	},
	"rtmpose": {
		"config": "mmpose/projects/rtmpose/rtmpose/body_2d_keypoint/rtmpose-s_8xb256-420e_coco-256x192.py",
		"model": "checkpoint/mmpose/rtmpose/rtmpose-s_simcc-aic-coco_pt-aic-coco_420e-256x192-fcb2599b_20230126.pth",
		"link": "https://download.openmmlab.com/mmpose/v1/projects/rtmpose/rtmpose-s_simcc-aic-coco_pt-aic-coco_420e-256x192-fcb2599b_20230126.pth"
	},
	"rtmo": {
		"config": "mmpose/configs/body_2d_keypoint/rtmo/crowdpose/rtmo-s_8xb32-700e_crowdpose-640x640.py",
		"model": "checkpoint/mmpose/rtmo/rtmo-s_8xb32-700e_crowdpose-640x640-79f81c0d_20231211.pth",
		"link": "https://download.openmmlab.com/mmpose/v1/projects/rtmo/rtmo-s_8xb32-700e_crowdpose-640x640-79f81c0d_20231211.pth"
	}
}


# 访问示例
# print(mmdet_dict["yolov3"]["config"])
# print(mmpose_dict["edpose"]["link"])


def build_topdown_command(det_model_name, pose_model_name, input_source, output_dir):
	det_config = mmdet_dict[det_model_name]["config"]
	det_model = mmdet_dict[det_model_name]["model"]
	pose_config = mmpose_dict[pose_model_name]["config"]
	pose_model = mmpose_dict[pose_model_name]["model"]

	command = [
		"python",
		"mmpose/demo/topdown_demo_with_mmdet.py",
		det_config,
		det_model,
		pose_config,
		pose_model,
		"--input",
		input_source,
		"--output-root",
		output_dir
	]

	return command


def build_bottomup_command(pose_model_name, input_source, output_dir):
	pose_config = mmpose_dict[pose_model_name]["config"]
	pose_model = mmpose_dict[pose_model_name]["model"]

	command = [
		"python",
		"mmpose/demo/bottomup_demo.py",
		pose_config,
		pose_model,
		"--input",
		input_source,
		"--output-root",
		output_dir
	]

	return command

import subprocess

# if __name__ == "__main__":
# 	# 示例输入
# 	det_model_name = "yolox"  # mmdet 模型名称
# 	pose_model_name = "rtmo"  # mmpose 模型名称
# 	input_source = "webcam"  # 输入源
# 	outptu_dir = ""
#
# 	# command = build_topdown_command(det_model_name, pose_model_name, input_source, outptu_dir)
# 	command = build_bottomup_command(pose_model_name, input_source, outptu_dir)
#
# 	# 打印命令
# 	print("Executing command:", " ".join(command))
# 	subprocess.run(command)




