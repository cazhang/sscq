"""
Draw Precision-Recall curves and Precision at Top-K curves
"""

import matplotlib.pyplot as plt
import os
import numpy as np
plt.rcParams.update({'font.size': 14})

data_path = './PATH_TO_CURVE_DATA'
dataset = 'cifar10'

# Draw Precision-Recall Curves
PR_curve = {}
PR_curve['SSCQ'] = np.loadtxt(os.path.join(data_path, dataset, 'SSCQ/PR_curve.txt'))
PR_curve['SSCQ-p'] = np.loadtxt(os.path.join(data_path, dataset, 'SSCQ-p/PR_curve.txt'))
PR_curve['SPQ'] = np.loadtxt(os.path.join(data_path, dataset, 'SPQ/PR_curve.txt'))
PR_curve['MeCoQ'] = np.loadtxt(os.path.join(data_path, dataset, 'MeCoQ/PR_curve.txt'))
PR_curve['Bi-half'] = np.loadtxt(os.path.join(data_path, dataset, 'Bi-half/PR_curve.txt'))
PR_curve['GreedyHash'] = np.loadtxt(os.path.join(data_path, dataset, 'GreedyHash/PR_curve.txt'))

plt.figure(figsize=(3.5,3.5))
for method in PR_curve:
    if method != 'MeCoQ':
        plt.plot(PR_curve[method][1], PR_curve[method][0], linestyle="-", label=method)
    else:
        plt.plot(PR_curve[method][0], PR_curve[method][1], linestyle="-", label=method)

plt.tight_layout()
plt.subplots_adjust(bottom=0.15)
plt.subplots_adjust(left=0.22)
plt.grid(True)
plt.xlim(0, 1)
#plt.ylim(0, 1)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend(fontsize=10, loc=1)
plt.savefig("{}_PR.png".format(dataset))


# Draw Precision at Top-1000 Curves
P_at_top = {}
P_at_top['SSCQ'] = np.loadtxt(os.path.join(data_path, dataset, 'SSCQ/P_at_topk_curve.txt'))
P_at_top['SSCQ-p'] = np.loadtxt(os.path.join(data_path, dataset, 'SSCQ-p/P_at_topk_curve.txt'))
P_at_top['SPQ'] = np.loadtxt(os.path.join(data_path, dataset, 'SPQ/P_at_topk_curve.txt'))
P_at_top['MeCoQ'] = np.loadtxt(os.path.join(data_path, dataset, 'MeCoQ/P_at_topk_curve.txt'))
P_at_top['Bi-half'] = np.loadtxt(os.path.join(data_path, dataset, 'Bi-half/P_at_topk_curve.txt'))
P_at_top['GreedyHash'] = np.loadtxt(os.path.join(data_path, dataset, 'GreedyHash/P_at_topk_curve.txt'))


plt.figure(figsize=(3.5,3.5))
for method in P_at_top:
    plt.plot(P_at_top[method][0], P_at_top[method][1], linestyle="-", label=method)

plt.tight_layout()
plt.subplots_adjust(bottom=0.15)
plt.subplots_adjust(left=0.22)
plt.grid(True)
plt.xlim(0, 1000)
#plt.ylim(0, 1)
plt.xlabel('Top returned samples')
plt.ylabel('Precision')
plt.legend(fontsize=10, loc=3)
plt.savefig("{}_PTOPK.png".format(dataset))


