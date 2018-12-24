import json
import cPickle
import argparse
import numpy as np

def _tiou(pred, gt):
    inter_left = np.maximum(pred[:,0], gt[0])
    inter_right = np.minimum(pred[:,1], gt[1])
    inter = np.maximum(0.0, inter_right - inter_left)
    union_left = np.minimum(pred[:,0], gt[0])
    union_right = np.maximum(pred[:,1], gt[1])
    union = np.maximum(0.0, union_right - union_left)
    return 1.0 * inter / union

def _nms(dets, scores, thresh=0.4):
    """Pure Python NMS baseline."""
    if len(dets) == 0: return []
    x1 = dets[:, 0]
    x2 = dets[:, 1]
    scores = scores
    lengths = x2 - x1
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        inter = np.maximum(0.0, xx2 - xx1)
        ovr = inter / (lengths[i] + lengths[order[1:]] - inter)
        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]
    return keep

def computer_eval(preds, gts, thresh=0.1, recall=[1]):
   results = {}
   for key, val in preds.iteritems():
     # if key == '58792004@N06_8977946361_0b26f4930d.mp4' or \
     #    key == '11699242@N07_8283285682_d1267bb1cc.mp4':
     #   continue
     queries = val[0]['query']
     #gt_key = key.split('-cam')[0]
     gt_key = key
     gt = gts[gt_key]
     proposals = np.empty((0,2))
     scores = np.empty((queries.shape[0], 0))
     

     for item in val:
         proposals = np.vstack((proposals, item['timestamp'][:,:2]))
         scores = np.hstack((scores, item['sim_scores']))

     recall_queries = []
     for i,q in enumerate(queries):
       keep = _nms(proposals, scores[i], thresh=thresh-0.05)[:max(recall)]
       gt_win = gt['timestamps'][i]
       pred_win = proposals[keep]
       overlap = _tiou(pred_win, gt_win)
       recall_queries.append([(overlap > thresh)[:d].any() for d in recall])
     results[gt_key] = recall_queries
   return results

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Process some integers.')
  parser.add_argument('pred_file', type=str,
                     help='a prediction pickle file')
  parser.add_argument('-gt', dest='gt_file', type=str,
                     default='........./experiments/preprocess/caption_gt_test.json',
                     help='the ground truth json file')
  parser.add_argument('-recall', dest='recall', metavar='N', type=int, nargs='+',
                     default=[1,5,10])
  parser.add_argument('-tiou', dest='tiou', metavar='N', type=float, nargs='+',
                     default=[0.1,0.3,0.5,0.7])
  
  args = parser.parse_args()
  print(args)

  gt_data = json.load(open(args.gt_file))
  pred_data = cPickle.load(open(args.pred_file, 'rb' ) )

  for t in args.tiou:
    results = computer_eval(pred_data, gt_data, thresh=t, recall=args.recall)
    eval_res = []
    for k, v in results.iteritems():
      eval_res = eval_res + v
    evals = np.array(eval_res).mean(axis=0)
    print "\ntiou@%.1f : " % t, args.recall
    for i, r in enumerate(args.recall):
        #print r, evals[i]
        print evals[i]
