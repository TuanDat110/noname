import pickle, numpy as np
from itertools import product
from tqdm import tqdm

sample_names, label_list = pickle.load(open("../data/WLASL300/val_only_label.pkl", "rb"))

def load(p): return dict(pickle.load(open(p, "rb")))
r1 = load("../work_dir/TEST_val_joint_wlasl300/eval_results/val_joint.pkl")
r2 = load("../work_dir/TEST_val_bone_wlasl300/eval_results/val_bone.pkl")
r3 = load("../work_dir/TEST_val_joint_motion_wlasl300/eval_results/val_joint_motion.pkl")
r4 = load("../work_dir/TEST_val_bone_motion_wlasl300/eval_results/val_bone_motion.pkl")

S1 = np.stack([r1[n] for n in sample_names])
S2 = np.stack([r2[n] for n in sample_names])
S3 = np.stack([r3[n] for n in sample_names])
S4 = np.stack([r4[n] for n in sample_names])
labels = np.array([int(l) for l in label_list])

grid = [round(x*0.1, 1) for x in range(0, 21)]   # 0.0 .. 2.0
combos = [(a1,a2,a3,a4) for a1,a2,a3,a4 in product(grid, repeat=4)
          if (a1+a2+a3+a4) > 0]

results = []
for a1, a2, a3, a4 in tqdm(combos, desc="grid-search alpha"):
    score = S1*a1 + S2*a2 + S3*a3 + S4*a4
    acc = (score.argmax(1) == labels).mean()
    results.append((acc, (a1, a2, a3, a4)))

results.sort(key=lambda x: x[0], reverse=True)

print("\n===== TOP 20 alpha =====")
for acc, a in results[:20]:
    print(f"top1={acc:.4f}  alpha={a}")

# restricted to weightings in which the joint stream (a1) has the largest weight
print("\n===== TOP alpha with joint as the largest weight =====")
cnt = 0
for acc, a in results:
    if a[0] == max(a):
        print(f"top1={acc:.4f}  alpha={a}")
        cnt += 1
        if cnt >= 10:
            break

best_acc, best_a = results[0]
print(f"\nBest overall: top1={best_acc:.4f}  alpha={best_a}")