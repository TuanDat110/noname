import argparse
import pickle
import numpy as np
from tqdm import tqdm

label = open("../data/WLASL2000/test_label.pkl", "rb")
file = pickle.load(label)
_, label_list = file

# Labels may be stored as strings; cast them to int once.
label_list = [int(l) for l in label_list]
num_class = int(max(label_list)) + 1
label = np.array(file, dtype=object)

# Score files of the four streams, as dicts keyed by sample name.
r1 = dict(pickle.load(open("../work_dir/TEST_test_joint_wlasl2000/eval_results/test_joint.pkl", "rb")))
r2 = dict(pickle.load(open("../work_dir/TEST_test_bone_wlasl2000/eval_results/test_joint.pkl", "rb")))
r3 = dict(pickle.load(open("../work_dir/TEST_test_joint_motion_wlasl2000/eval_results/test_joint.pkl", "rb")))
r4 = dict(pickle.load(open("../work_dir/TEST_test_bone_motion_wlasl2000/eval_results/test_joint.pkl", "rb")))

# alpha = [1.2,1.3,1.6,0.7]  # NMFs-CSL
# alpha = [1.6, 1.1, 0.9, 0.4]  # 53.68 for WLASL2000
alpha = [1.1, 0.6, 0.5, 0.5]  # 53.68 for WLASL2000
# alpha = [1.5,0.6,0.5,0.1]  # WLASL1000
# alpha = [1.2,0.3,0.2,0.9]  # WLASL300
# alpha = [1.1, 0.7, 0.9, 0.3]# WLASL100
# alpha = [1.3,1.3,0.8,1.0]  # SLR500
# alpha = [1.8,0.5,0.1,0.7]  # AUTSL
# alpha = [1.7,1.5,1.0,1.1]  # MSASL1000
# alpha = [1.6,1.2,0.6,0.9]  # MSASL500
# alpha = [1.7,0.5,0.4,0.6]  # MSASL200
# alpha = [1.9,1.4,0.3,0.4]  # MSASL100
# alpha = [1.2,1.3,1.6,0.7]  # NMFs-CSL

right_num = total_num = right_num_5 = 0
names = []
preds = []
scores = []
mean = 0

with open("predictions.csv", "w") as f:
    for i in tqdm(range(len(label[0]))):
        name, l = label[:, i]
        l = int(l)
        names.append(name)

        # Look the scores up by sample name, not by position.
        r11 = r1[name]
        r22 = r2[name]
        r33 = r3[name]
        r44 = r4[name]
        
        mean += r11.mean()
        score = (
            r11 * alpha[0] + r22 * alpha[1] + r33 * alpha[2] + r44 * alpha[3]
        ) / np.array(alpha).sum()
        
        rank_5 = score.argsort()[-5:]
        right_num_5 += int(l in rank_5)
        r = np.argmax(score)
        scores.append(score)
        right_num += int(r == l)
        total_num += 1
        f.write("{}, {}\n".format(name, r))

    scores = np.stack(scores)
    rank = scores.argsort()
    
    hit_top_1 = [l in rank[i, -1:] for i, l in enumerate(label_list)]
    
    # Per-class accuracy, averaged over the classes that occur in this split.
    valid_acc = []
    for c in range(num_class):
        c_count = label_list.count(c)
        if c_count > 0:
            hit_label = [l == c for l in label_list]
            acc_c = np.sum(
                np.array(hit_top_1).astype(float) * np.array(hit_label).astype(float)
            ) / c_count
            valid_acc.append(acc_c)
    acc_per_class = np.mean(valid_acc)

    hit_top_5 = [l in rank[i, -5:] for i, l in enumerate(label_list)]
    
    valid_acc_5 = []
    for c in range(num_class):
        c_count = label_list.count(c)
        if c_count > 0:
            hit_label = [l == c for l in label_list]
            acc_c_5 = np.sum(
                np.array(hit_top_5).astype(float) * np.array(hit_label).astype(float)
            ) / c_count
            valid_acc_5.append(acc_c_5)
    acc_per_class_5 = np.mean(valid_acc_5)

    acc = right_num / total_num
    acc5 = right_num_5 / total_num
    print(total_num)
    print("top1: ", acc)
    print("top1 per class: ", acc_per_class)
    print("top5: ", acc5)
    print("top5 per class: ", acc_per_class_5)

print(mean / len(label[0]))

with open("./gcn_ensembled.pkl", "wb") as f:
    score_dict = dict(zip(names, scores))
    pickle.dump(score_dict, f)