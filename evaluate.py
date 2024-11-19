import os
import numpy as np
from openai import OpenAI

from openai import OpenAI

def client_prompt(prompt):#连接api

    client = OpenAI(
        api_key="sk-YYBXmcVSvoa0kNObDxqqHJByvIUMCdkY6YH2ZjS4UpNkwPyj",  # 在这里将 MOONSHOT_API_KEY 替换为你从 Kimi 开放平台申请的 API Key
        base_url="https://api.moonshot.cn/v1/",
    )

    completion = client.chat.completions.create(
        model="moonshot-v1-8k",
        messages=[
            {"role": "system",
             "content": "你是 Kimi，由 Moonshot AI 提供的人工智能助手，你更擅长中文和英文的对话。你会为用户提供安全，有帮助，准确的回答。同时，你会拒绝一切涉及恐怖主义，种族歧视，黄色暴力等问题的回答。Moonshot AI 为专有名词，不可翻译成其他语言。"},
            {"role": "user", "content":prompt}
        ],
        temperature=0.3,
    )
    return completion.choices[0].message.content
def remove_elements_with_large_gap(lst): #用于避免角度之间较大的的gap
    i = 0
    while i < len(lst) - 1:
        if abs(lst[i + 1] - lst[i]) > 25:
            lst = lst[:i + 1]
            break  # 一旦找到大于20的gap，就停止循环
        i += 1
    return lst
def evaluate_pose(pose_seq, exercise):
    """Evaluate a pose sequence for a particular exercise.

    Args:
        pose_seq: PoseSequence object.
        exercise: String name of the exercise to evaluate.

    Returns:
        correct: Bool whether exercise was performed correctly.
        feedback: Feedback string.

    """
    if exercise == 'bicep_curl':
        return _bicep_curl(pose_seq)
    elif exercise == 'shoulder_press':
        return _shoulder_press(pose_seq)
    elif exercise == 'front_raise':
        return _front_raise(pose_seq)
    elif exercise == 'shoulder_shrug':
        return _shoulder_shrug(pose_seq)
    elif exercise == 'elbow_plank': #平板支撑
        return _elbow_plank(pose_seq)
    elif exercise == 'Dumbbell_bird': #哑铃飞鸟
        return _Dumbbell_bird(pose_seq)
    elif  exercise == 'shoulder_push':
        return _shoulder_push(pose_seq)
    elif exercise == 'barbell_squats':
        return  _barbell_squats(pose_seq)
    elif exercise == 'Barbell_Bench_Press': #杠铃卧推
        return _Barbell_Bench_Press(pose_seq)
    elif exercise == 'Lean_row':#俯身划船
        return _Lean_row(pose_seq)
    else:
        return (False, "Exercise string not recognized.")
def is_right(pose_seq):#区分一个姿势是否是右侧
    poses = pose_seq.poses
    right_present = [1 for pose in poses
                     if pose.rshoulder.exists and pose.relbow.exists and pose.rwrist.exists]
    left_present = [1 for pose in poses
                    if pose.lshoulder.exists and pose.lelbow.exists and pose.lwrist.exists]
    right_count = sum(right_present)
    left_count = sum(left_present)
    if right_count > left_count:
        print('Exercise arm detected as: right.')
        return True
    else:
        print('Exercise arm detected as: left.')
        return False
def nomalize_vecs(vecs):#归一化vectors
    return vecs / np.expand_dims(np.linalg.norm(vecs, axis=1), axis=1)
def cal_degrees(vec1,vec2):#计算两个向量列表的夹角,只会返回一个180以内的角度
    return np.degrees(
        np.arccos(np.clip(np.sum(np.multiply(vec1, vec2), axis=1), -1.0, 1.0)))
'''
计算角度的一堆函数，得到的是一个joints
输出的是一组度数
'''
def  cal_degrees_joints(joints,which_degree):#计算角度的函数
#计算度数的函数列表
    help = {"ren":["neck","rshoulder","relbow"],
            "len":["neck","lshoulder","lelbow"],
            "reh":["relbow","rshoulder","rhip"],
            "leh":["lelbow","lshoulder","lhip"],
            "rws":["rwrist","relbow","rshoulder"],
            "lws":["lwrist","lelbow","lshoulder"],
            "rsk":["rknee","rhip","rshoulder"],
            "lsk":["lknee","lhip","lshoulder"],
            "rnk":["neck","rhip","rknee"],
            "lnk":["neck","lhip","lknee"],
            "rha":["rankle","rknee","rhip"],
            "lha":["lankle","lknee","lhip"],
            "rshg":["rshoudler","rhip","ground"],
            "lshg":["lshoudler","lhip","ground"],
            "rnhg":["neck","rhip","ground"],
            "lnhg":["neck","lhip","ground"],
            "rseg":["rshoulder","relbow","ground"],
            "lseg":["lshoulder","lelbow","ground"],
            "rewg":["relbow","rwrist","ground"],
            "lewg":["lelbow","lwrist","ground"],
            "reeg":["reye","rear","ground"],
            "leeg":["leye","lear","ground"],
            "ssg":["rshoulder","lshoulder","ground"],
            "hhg":["rhip","lhip","ground"],
            "rnsg":["neck","rshoulder","ground"],
            "lnsg":["neck","lshoulder","ground"],
            "lkhg":[ "lknee","lhip","ground"],
            "rkhg":["rknee","rhip","ground"],
            "rekg":["rear","rknee","ground"],
            "lekg":["lear","lknee","ground"],
            "rnwg":["neck","rwrist","ground"],
            "lnwg":[ "neck","lwrist","ground"],
            "lheg":["lhip","lelbow","ground"],
            "rheg":["rhip","relbow","ground"]


            }

    #获取which_degree对应的三个关节点并且计算两个向量,为了便于使用，这里进行判断一下便于 rsw和rws是一样的
    swap_which_degree = f"{which_degree[0]}{which_degree[2]}{which_degree[1]}{which_degree[3:]}"
    if which_degree in help:
        joints_need_list =  help[which_degree]
    elif swap_which_degree in help:
        joints_need_list = help[swap_which_degree]
    else:
        #抛出异常暂停，暂停程序
        raise KeyError(f"Key '{which_degree}' or '{swap_which_degree}' not found in help dictionary.")
    #判断一下是否有和地面的夹角计算
    if "ground" in joints_need_list:
        joints_need_list.remove("ground")
        vecs1 = np.array([(joint[joints_need_list[0]].x - joint[joints_need_list[1]].x, joint[joints_need_list[0]].y - joint[joints_need_list[1]].y) for joint in joints])
        vecs2 = np.full((vecs1.shape[0],2),[1,0])
    else:
        vecs1 = np.array([(joint[joints_need_list[0]].x - joint[joints_need_list[1]].x, joint[joints_need_list[0]].y - joint[joints_need_list[1]].y) for joint in joints])
        vecs2 = np.array([(joint[joints_need_list[1]].x - joint[joints_need_list[2]].x, joint[joints_need_list[1]].y - joint[joints_need_list[2]].y) for joint in joints])
# 全部归一化
    vecs1 = nomalize_vecs(vecs1)
    vecs2 = nomalize_vecs(vecs2)
    return cal_degrees(vecs1,vecs2)
def _Lean_row(pose_seq):#俯身划船,最好从左侧录制视频
    poses = pose_seq.poses
    if is_right(pose_seq):#
        if_right = True
        joints_ = [{ "neck":pose.neck, "rhip": pose.rhip,
                   "rknee": pose.rknee, "rankle": pose.rankle,"rshoulder":pose.rshoulder,"relbow":pose.relbow,"rwrist":pose.rwrist} for pose in poses]
    else:
        if_right = False
        joints_ = [{"neck":pose.neck,"lhip": pose.lhip,
                   "lknee": pose.lknee, "lankle": pose.lankle,"lshoulder":pose.lshoulder,"lelbow":pose.lelbow,"lwrist":pose.lwrist}
                   for pose in poses]
    joints_ = [joint for joint in joints_ if all(part.exists for part in list(joint.values()))]
    if if_right:#是右侧
        ha = cal_degrees_joints(joints_,"rha")#大腿与小腿的夹角
        nhg = cal_degrees_joints(joints_,"rnhg")#躯干与地面的夹角
        seg =  cal_degrees_joints(joints_,"rseg")#大臂和地面的夹角

    else:
        ha = cal_degrees_joints(joints_, "lha")
        nhg = cal_degrees_joints(joints_, "lnhg")
        seg = cal_degrees_joints(joints_, "lseg")
    ha = remove_elements_with_large_gap(ha)
    if ha[0] < 90:
        ha = 180 - ha

    nhg = remove_elements_with_large_gap(nhg)
    if nhg[0] < 90:
        nhg = 180 - nhg

    seg = remove_elements_with_large_gap(seg)
    if seg[0] > 90:
        seg = 180 - seg

    prompt = "这个是一个俯身划船的动作，下面我将提供一些从侧面看这个人的一些关于该动作的关键骨骼的角度信息，请你分析这个人的这个动作是否标准，如果不标准请你给我建议。下面是一些信息："
    elbow_higher_back = False
    for i in range(0,len(seg)):
        if seg[i] > nhg[i] and seg[i]-nhg[i] > 10:#大臂与地面夹角大于躯干与地面夹角
            elbow_higher_back = True
            break

    if elbow_higher_back:
        prompt = prompt + f"在做俯身划船运动时在最高点时胳膊肘高于躯干(这被认为是标准的，表示充分锻炼到背部肌肉）,"
    else:
        prompt = prompt + f"在做俯身划船运动时在最高点时胳膊肘低于躯干(动作没有做到位，手臂没有抬到较高位置）,"

    prompt = prompt + f"大腿和小腿的夹角范围为{int(np.min(ha))}度到{int(np.max(ha))}度（一般认为120度到160度是标准的,否则太高或太低不利于发力，可能造成受伤）,躯干与地面的夹角范围为{int(np.min(nhg))}度到{int(np.max(nhg))}度."

    print(prompt)
    return client_prompt(prompt)







def _Barbell_Bench_Press(pose_seq):#杠铃卧推正面
    poses = pose_seq.poses
    # 正面动作
    # joints = [(pose.rshoulder, pose.relbow, pose.rwrist, pose.rhip, pose.rknee, pose.rankle, pose.neck) for pose in poses]
    joints_ = [{"rshoulder": pose.rshoulder, "relbow": pose.relbow, "rwrist": pose.rwrist,
                "lshoulder": pose.lshoulder, "lelbow": pose.lelbow, "lwrist": pose.lwrist,
                "neck": pose.neck}
               for pose in poses]

    joints_ = [joint for joint in joints_ if all(part.exists for part in list(joint.values()))]

    rewg = cal_degrees_joints(joints_,"rewg")
    lewg = cal_degrees_joints(joints_,"lewg")
    rewg = remove_elements_with_large_gap(rewg)
    lewg = remove_elements_with_large_gap(lewg)
    rnwg =  cal_degrees_joints(joints_,"rnwg")
    lnwg =  cal_degrees_joints(joints_,"lnwg")
    rnwg = remove_elements_with_large_gap(rnwg)
    lnwg = remove_elements_with_large_gap(lnwg)
    if rnwg[0]>90:
        rnwg = 180 - rnwg
    if lnwg[0]>90:
        lnwg = 180 - lnwg
    if rewg[0]>90:
        rewg = 180 - rewg
    if lewg[0]>90:
        lewg = 180 - lewg
    prompt = f"这个是一个杠铃卧推的动作，下面我将提供一些从脚部往头部看这个人的一些关于该动作的关键骨骼的角度信息（也即平行于这个人的身体看他），请你分析这个人的这个动作是否标准，如果不标准请你给我建议。下面是一些信息："
    prompt = prompt + f"在做这个杠铃卧推的过程中，右侧小臂与地面的夹角范围为{int(np.min(rewg))}度至{int(np.max(rewg))}度，左侧小臂与地面的夹角范围为{int(np.min(lewg))}度至{int(np.max(lewg))}度，"
    prompt = prompt + f"这里还有一个指标用来衡量卧推过程中是否将杠铃放在了胸上（也即脖子与左手腕的连线与地面的夹角和脖子与右手腕的连线与地面的夹角，这两个夹角约接近0度，越标准），这个人做动作时脖子与左手腕的连线与地面的夹角最小值为{int(np.min(rnwg))}度，脖子与右手腕的连线与地面的夹角最小值为{int(np.min(lnwg))}度."
    print(prompt)
    return client_prompt(prompt)

def _barbell_squats(pose_seq):#杠铃深蹲
    poses = pose_seq.poses

    if is_right(pose_seq):#
        if_right = True
        # joints = [(pose.rshoulder, pose.relbow, pose.rwrist, pose.rhip, pose.rknee, pose.rankle, pose.neck) for pose in poses]
        joints_ = [{ "rhip": pose.rhip,
                   "rknee": pose.rknee, "rankle": pose.rankle,"rear":pose.rear} for pose in poses]
    else:
        if_right = False
        joints_ = [{"lhip": pose.lhip,
                   "lknee": pose.lknee, "lankle": pose.lankle,"lear":pose.lear}
                   for pose in poses]
    # filter out data points where a part does not exist
    # joints = [joint for joint in joints if all(part.exists for part in joint)]
    joints_ = [joint for joint in joints_ if all(part.exists for part in list(joint.values()))]
    if if_right == True:
        hkg = cal_degrees_joints(joints_,"rhkg")
        ekg = cal_degrees_joints(joints_,"rekg")
    else:
        hkg = cal_degrees_joints(joints_,"lhkg")
        ekg = cal_degrees_joints(joints_,"lekg")
    hkg = remove_elements_with_large_gap(hkg)
    ekg = remove_elements_with_large_gap(ekg)
    if hkg[0] > 90:
        hkg = 180 - hkg

    prompt = f"这个是一个杠铃深蹲的动作，下面我将提供一些从侧面看这个人的一些关于该动作的关键骨骼的角度信息，请你分析这个人的这个动作是否标准，如果不标准请你给我一些可采取的、一定是正确的建议。下面是一些信息："
    prompt = prompt + f"大腿与地面的夹角从起始的{int(np.max(hkg))}度至{int(np.min(hkg))}度。（这个夹角，需要关注到终止的角度，因为终止时角度代表下蹲的幅度是否到位，越接近0度越到位，也即大腿与地面平行，这个角度如果超出10度就认为做的不到位；"
    prompt = prompt + f"这里还有一个关于眼睛到膝盖的直线与地面的夹角（这个夹角主要反应人的身体是否有前倾或者后倾状态,一般认为度数范围不低于80度或不高于100度都算标准），这个人做动作时这个角度范围为：{int(np.min(ekg))}度到{int(np.max(ekg))}度。"
    print(prompt)
    return client_prompt(prompt)




def _shoulder_push(pose_seq):#坐姿推肩
    poses = pose_seq.poses
    # 正面动作
    # joints = [(pose.rshoulder, pose.relbow, pose.rwrist, pose.rhip, pose.rknee, pose.rankle, pose.neck) for pose in poses]
    joints_ = [{"rshoulder": pose.rshoulder, "relbow": pose.relbow, "rwrist": pose.rwrist,
                "lshoulder": pose.lshoulder, "lelbow": pose.lelbow, "lwrist": pose.lwrist}
               for pose in poses]

    joints_ = [joint for joint in joints_ if all(part.exists for part in list(joint.values()))]

    rws =  cal_degrees_joints(joints_, "rws") #小臂与大臂的夹角
    rws= 180 - rws
    lws =  cal_degrees_joints(joints_, "lws")
    lws = 180 - lws
    rewg = cal_degrees_joints(joints_,"rewg")#小臂与地面的夹角
    lewg = cal_degrees_joints(joints_,"lewg")
    rseg = cal_degrees_joints(joints_,"rseg")#大臂与地面的夹角
    rseg = 180 - rseg
    lseg = cal_degrees_joints(joints_,"lseg")

    rseg = remove_elements_with_large_gap(rseg)
    rws =  remove_elements_with_large_gap(rws)
    lws =  remove_elements_with_large_gap(lws)
    rewg =  remove_elements_with_large_gap(rewg)
    lewg =  remove_elements_with_large_gap(lewg)
    lseg =  remove_elements_with_large_gap(lseg)





    prompt = f"这个是一个坐姿推肩的动作，下面我将提供一些从正面看这个人的一些关于该动作的关键骨骼的角度信息，请你分析这个人的这个动作是否标准，如果不标准请你给我建议。下面是一些信息："
    prompt = prompt + f"右侧大臂和小臂的之间的夹角从起始的{int(np.min(rws))}度至{int(np.max(rws))}度，左侧大臂和小臂的之间的夹角从起始的{int(np.min(lws))}度至{int(np.max(lws))}度，"
    if rseg[0] != np.min(rseg):
        prompt = prompt + f"右侧大臂与地面之间的夹角先从{int(rseg[0])}度减少至0度再增大到{int(np.max(rseg))}度"
    else:
        prompt = prompt + f"，左侧大臂与地面之间的夹角范围为{int(np.min(rseg))}度至{int(np.max(rseg))}度"

    if lseg[0] != np.min(lseg):
        prompt = prompt + f"，左侧大臂与地面之间的从{int(lseg[0])}度减少至0度再增大到{int(np.max(lseg))}度"
    else:
        prompt = prompt + f"，左侧大臂与地面之间的夹角范围为{int(np.min(lseg))}度至{int(np.max(lseg))}度"

    prompt = prompt + f"，右侧小臂与地面之间的夹角范围为{int(np.min(rewg))}度至{int(np.max(rewg))}度，左侧小臂与地面之间的夹角范围为{int(np.min(lewg))}度至{int(np.max(lewg))}度"
    print(prompt)
    return client_prompt(prompt)
def _elbow_plank(pose_seq):#表示平板支撑的方式
    poses = pose_seq.poses
    if is_right(pose_seq):#
        # joints = [(pose.rshoulder, pose.relbow, pose.rwrist, pose.rhip, pose.rknee, pose.rankle, pose.neck) for pose in poses]
        joints_ = [{"rshoulder": pose.rshoulder, "relbow": pose.relbow, "rwrist": pose.rwrist, "rhip": pose.rhip,
                   "rknee": pose.rknee, "rankle": pose.rankle, "neck": pose.neck} for pose in poses]
    else:
        joints_ = [{"lshoulder": pose.lshoulder, "lelbow": pose.lelbow, "lwrist": pose.lwrist, "lhip": pose.lhip,
                    "lknee": pose.lknee, "lankle": pose.lankle, "neck": pose.neck}
                   for pose in poses]
    # filter out data points where a part does not exist
    # joints = [joint for joint in joints if all(part.exists for part in joint)]
    joints_ = [joint for joint in joints_ if all(part.exists for part in list(joint.values()))]

    if is_right(pose_seq):
        rah = cal_degrees_joints(joints_, "rah")
        rnk = cal_degrees_joints(joints_, "rnk")
        rsw = cal_degrees_joints(joints_, "rsw")
    else:
        rah = cal_degrees_joints(joints_, "lah")
        rnk = cal_degrees_joints(joints_, "lnk")
        rsw = cal_degrees_joints(joints_, "lsw")

    max_rah =  np.max(rah)
    max_rnk = np.max(rnk)
    rsw_range =  np.max(rsw) - np.min(rsw)

    prompt = f"这个平板支撑动作，下面我将提供一些这个人的一些关于该动作的关键骨骼的角度信息，请你分析这个人的这个动作是否标准，如果不标准请你给我建议。下面是一些信息："
    prompt =  prompt + f"小腿和大腿之间的夹角为{180-int(max_rah)}度（小腿与大腿的夹角越接近180度表示越标准，腿越接近伸直），从侧面看躯干与大腿的夹角为{180-int(max_rnk)}度，小臂和大臂的夹角范围为{180-int(np.min(rsw))}度至{180-int(np.max(rsw))}度"
    print(prompt)
    return client_prompt(prompt)



def _Dumbbell_bird(pose_seq):#哑铃飞鸟，侧平举
    poses = pose_seq.poses
    # 正面动作
        # joints = [(pose.rshoulder, pose.relbow, pose.rwrist, pose.rhip, pose.rknee, pose.rankle, pose.neck) for pose in poses]
    joints_ = [{"rshoulder": pose.rshoulder, "relbow": pose.relbow, "rwrist": pose.rwrist,
                "lshoulder": pose.lshoulder, "lelbow": pose.lelbow, "lwrist": pose.lwrist,
                "rhip": pose.rhip,"lhip": pose.lhip,"rknee": pose.rknee,"lknee": pose.lknee}
                   for pose in poses]

    joints_ = [joint for joint in joints_ if all(part.exists for part in list(joint.values()))]

    reh = cal_degrees_joints(joints_, "reh")
    reh = 180 - reh
    leh = cal_degrees_joints(joints_, "leh")
    leh = 180 - leh
    rsw = cal_degrees_joints(joints_, "rsw")
    rsw = 180 - rsw
    lsw = cal_degrees_joints(joints_, "lsw")
    lsw = 180 - lsw
    rsk =  cal_degrees_joints(joints_, "rsk")
    # rnsg = cal_degrees_joints(joints_, "rnsg")
    # lnsg = cal_degrees_joints(joints_, "lnsg")
    # lnsg = 180 - lnsg

    prompt = "这个是一个哑铃飞鸟(或者说是侧平举）的动作，下面我将提供一些从正面看这个人的一些关于该动作的关键骨骼的角度信息，请你分析这个人的这个动作是否标准，如果不标准请你给我建议。下面是一些信息："
    prompt = prompt + f"右侧大臂和小臂的之间的夹角范围为{int(np.min(rsw))}度至{int(np.max(rsw))}度，左侧大臂和小臂的之间的夹角范围为{int(np.min(lsw))}度至{int(np.max(lsw))}度，"
    prompt = prompt + f"右侧大臂与躯干之间的夹角范围为{int(np.min(reh))}度至{int(np.max(reh))}度,左侧大臂与躯干之间的夹角范围为{int(np.min(leh))}度至{int(np.max(leh))}度"
    print(prompt)
    return client_prompt(prompt)

def _bicep_curl(pose_seq):
    # find the arm that is seen most consistently
    poses = pose_seq.poses

    if is_right(pose_seq):#
        joints_ = [{"rshoulder": pose.rshoulder, "relbow": pose.relbow, "rwrist": pose.rwrist, "rhip": pose.rhip} for pose in poses]
    else:
        joints_ = [{"lshoulder": pose.lshoulder, "lelbow": pose.lelbow, "lwrist": pose.lwrist, "lhip": pose.lhip } for pose in poses]

    joints_ = [joint for joint in joints_ if all(part.exists for part in list(joint.values()))]

    if is_right(pose_seq):
        eh = cal_degrees_joints(joints_, "reh")#大臂和躯干的夹角
        ws = cal_degrees_joints(joints_,"rws") #大臂和小臂的夹角
    else:
        eh = cal_degrees_joints(joints_, "leh")
        ws = cal_degrees_joints(joints_, "lws")
    eh = remove_elements_with_large_gap(eh)
    ws = remove_elements_with_large_gap(ws)

    if eh[0] > 90:
        eh = 180 - eh

    prompt = "这个是一个肱二头肌弯举（是一种主要针对肱二头肌的锻炼动作。通常通过手持哑铃或杠铃，将重量从身体两侧向上弯曲至胸前，再缓慢放回原位）的动作，下面我将提供一些从侧面看这个人的一些关于该动作的关键骨骼的角度信息，请你分析这个人的这个动作是否标准，如果不标准请你给我建议。下面是一些信息："
    prompt = prompt + f"大臂和躯干的夹角范围为{int(np.min(eh))}度至{int(np.max(eh))}度(如果这个越接近于0越好），大臂和小臂的夹角范围从最开始的{int(np.min(ws))}度提升至至{int(np.max(ws))}度."
    print(prompt)
    return client_prompt(prompt)


    
def _front_raise(pose_seq): # 直臂前臂举
    poses = pose_seq.poses
    if is_right(pose_seq):#
        joints_ = [{"rshoulder": pose.rshoulder, "relbow": pose.relbow, "rwrist": pose.rwrist, "rhip": pose.rhip} for pose in poses]
    else:
        joints_ = [{"lshoulder": pose.lshoulder, "lelbow": pose.lelbow, "lwrist": pose.lwrist, "lhip": pose.lhip } for pose in poses]

    joints_ = [joint for joint in joints_ if all(part.exists for part in list(joint.values()))]

    if is_right(pose_seq):
        eh = cal_degrees_joints(joints_, "reh")#大臂和躯干的夹角
        ws = cal_degrees_joints(joints_,"rws") #大臂和小臂的夹角
    else:
        eh = cal_degrees_joints(joints_, "leh")
        ws = cal_degrees_joints(joints_, "lws")
    eh = remove_elements_with_large_gap(eh)
    ws = remove_elements_with_large_gap(ws)

    if eh[0] > 90:
        eh = 180 - eh
    if ws[0] < 90:
        ws = 180 - ws

    prompt = "这个是一个直臂前平举的动作，下面我将提供一些从侧面看这个人的一些关于该动作的关键骨骼的角度信息，请你分析这个人的这个动作是否标准，如果不标准请你给我建议。下面是一些信息："
    prompt = prompt + f"大臂和躯干的夹角从最开始的{int(np.min(eh))}度变化至{int(np.max(eh))}度再到{int(eh[len(eh)-1])}度，大臂和小臂的夹角最小值为{int(np.min(ws))}度（这个值越接近180越标准）."
    print(prompt)
    return client_prompt(prompt)

    
    
