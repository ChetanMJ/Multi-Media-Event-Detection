import sys
import os
from sklearn.metrics import average_precision_score

if __name__=="__main__":
    #   load the ground-truth file list
    gt_fn=open(sys.argv[1]).readlines()
    pred_fn=open(sys.argv[2]).readlines()

    print("Evaluating the average precision (AP)")

    y_gt=[]
    y_score=[]
    assert(len(y_gt)==len(y_score))

    for lines in gt_fn:
        y_gt.append(float(lines.strip()))

    for lines in pred_fn:
        y_score.append(float(lines.strip()))

    assert(len(y_gt) == len(y_score))
    print "Average precision: ",average_precision_score(y_gt,y_score)
