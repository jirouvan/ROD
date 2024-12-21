import numpy as np
import os, json, math
from tqdm import tqdm
from mmdet.apis import ( inference_detector,init_detector)
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def summarize(self, catId=None):
    """
    Compute and display summary metrics for evaluation results.
    Note this functin can *only* be applied on the default parameter setting
    """

    def _summarize(ap=1, iouThr=None, areaRng='all', maxDets=100):
        p = self.params
        iStr = ' {:<18} {} @[ IoU={:<9} | area={:>6s} | maxDets={:>3d} ] = {:0.3f}'
        titleStr = 'Average Precision' if ap == 1 else 'Average Recall'
        typeStr = '(AP)' if ap == 1 else '(AR)'
        iouStr = '{:0.2f}:{:0.2f}'.format(p.iouThrs[0], p.iouThrs[-1]) \
            if iouThr is None else '{:0.2f}'.format(iouThr)

        aind = [i for i, aRng in enumerate(p.areaRngLbl) if aRng == areaRng]
        mind = [i for i, mDet in enumerate(p.maxDets) if mDet == maxDets]

        if ap == 1:
            # dimension of precision: [TxRxKxAxM]
            s = self.eval['precision']
            # IoU
            if iouThr is not None:
                t = np.where(iouThr == p.iouThrs)[0]
                s = s[t]

            # 判断是否传入catId，如果传入就计算指定类别的指标
            if isinstance(catId, int):
                s = s[:, :, catId, aind, mind]
            else:
                s = s[:, :, :, aind, mind]

        else:
            # dimension of recall: [TxKxAxM]
            s = self.eval['recall']
            if iouThr is not None:
                t = np.where(iouThr == p.iouThrs)[0]
                s = s[t]

            # 判断是否传入catId，如果传入就计算指定类别的指标
            if isinstance(catId, int):
                s = s[:, catId, aind, mind]
            else:
                s = s[:, :, aind, mind]

        if len(s[s > -1]) == 0:
            mean_s = -1
        else:
            mean_s = np.mean(s[s > -1])

        print_string = iStr.format(titleStr, typeStr, iouStr, areaRng, maxDets, mean_s)
        return mean_s, print_string

    stats, print_list = [0] * 12, [""] * 12
    stats[0], print_list[0] = _summarize(1)
    stats[1], print_list[1] = _summarize(1, iouThr=.5, maxDets=self.params.maxDets[2])
    stats[2], print_list[2] = _summarize(1, iouThr=.75, maxDets=self.params.maxDets[2])
    stats[3], print_list[3] = _summarize(1, areaRng='small', maxDets=self.params.maxDets[2])
    stats[4], print_list[4] = _summarize(1, areaRng='medium', maxDets=self.params.maxDets[2])
    stats[5], print_list[5] = _summarize(1, areaRng='large', maxDets=self.params.maxDets[2])
    stats[6], print_list[6] = _summarize(0, maxDets=self.params.maxDets[0])
    stats[7], print_list[7] = _summarize(0, maxDets=self.params.maxDets[1])
    stats[8], print_list[8] = _summarize(0, maxDets=self.params.maxDets[2])
    stats[9], print_list[9] = _summarize(0, areaRng='small', maxDets=self.params.maxDets[2])
    stats[10], print_list[10] = _summarize(0, areaRng='medium', maxDets=self.params.maxDets[2])
    stats[11], print_list[11] = _summarize(0, areaRng='large', maxDets=self.params.maxDets[2])

    print_info = "\n".join(print_list)

    if not self.eval:
        raise Exception('Please run accumulate() first')

    return stats, print_info

def iou(box1, box2):
    '''
    两个框（二维）的 iou 计算

    注意：边框以左上为原点

    box:[top, left, bottom, right]
    '''
    in_h = min(box1[2], box2[2]) - max(box1[0], box2[0])
    in_w = min(box1[3], box2[3]) - max(box1[1], box2[1])
    inter = 0 if in_h < 0 or in_w < 0 else in_h * in_w
    union = (box1[2] - box1[0]) * (box1[3] - box1[1]) + \
            (box2[2] - box2[0]) * (box2[3] - box2[1]) - inter
    iou = inter / union
    return iou

def predict_json(image_root,val_json,model,thr=0.3):
    images_root = image_root
    predicts_cls=[]
    predicts_state = []
    predicts_merge = []
    val_json = json.load(open(val_json,'r'))
    for image in tqdm(val_json['images']):
        image_path = image['file_name']
        # id_cls, scores_cls, result_state, id_merge, scores_merge, bboxes = inference_detector(model, os.path.join(images_root,image_path))
        result = inference_detector(model, os.path.join(images_root, image_path))
######################################  cls  ###########################################
        id_cls = result[0].pred_instances_cls.labels.tolist()
        scores_cls = result[0].pred_instances_cls.scores.tolist()
        bboxes_cls = result[0].pred_instances_cls.bboxes.tolist()
        for i in range(len(result[0].pred_instances_cls.labels)):
            content = scores_cls[i]
            if content > thr:
                bbox = bboxes_cls[i]
                predict = {
                    'image_id': image['id'],
                    'category_id': id_cls[i],
                    'bbox': [bbox[0], bbox[1], (bbox[2] - bbox[0]), (bbox[3] - bbox[1])],
                    'score': np.float64(content)
                }
                predicts_cls.append(predict)
            else:
                break
        json_str_cls = json.dumps(predicts_cls, indent=4)
        with open('./predict_result_cls.json', 'w') as json_file:
            json_file.write(json_str_cls)

######################################  state  ###########################################
        id_state = result[0].pred_instances_state.labels.tolist()
        scores_state = result[0].pred_instances_state.scores.tolist()
        bboxes_state = result[0].pred_instances_state.bboxes.tolist()
        for i in range(len(result[0].pred_instances_state.labels)):
            # content_cls = scores_cls[i]
            content = scores_state[i]
            if content > thr:
                bbox = bboxes_state[i]
                predict = {
                    'image_id': image['id'],
                    'category_id': id_state[i],
                    'bbox': [bbox[0], bbox[1], (bbox[2] - bbox[0]),(bbox[3] - bbox[1])],
                    'score': np.float64(content)
                }
                predicts_state.append(predict)
            else:
                break
        json_str_state = json.dumps(predicts_state, indent=4)
        with open('./predict_result_state.json', 'w') as json_file:
            json_file.write(json_str_state)

######################################  merge  ###########################################
        id_merge = []
        scores_merge = []
        if len(id_cls)>len(id_state):
            bboxes_merge = bboxes_state
        else:
            bboxes_merge = bboxes_cls

        for i in range(min(len(id_cls),len(id_state))):
            id_merge.append(id_cls[i] + id_state[i] * 10)
            scores_merge.append(math.sqrt(scores_state[i] * scores_cls[i]))
            # scores_merge.append((scores_state[i] + scores_cls[i])/2)
        for i in range(len(id_merge)):
            content = scores_merge[i]
            # content_cls = scores_cls[i]
            # content_state = scores_state[i]
            # if content_cls > 0.3:
            # if content_cls > 0.3 and content_state > 0.3:
            if content > 0.3:
                bbox = bboxes_merge[i]
                predict = {
                    'image_id': image['id'],
                    'category_id': id_merge[i],
                    'bbox': [bbox[0], bbox[1], (bbox[2] - bbox[0]), (bbox[3] - bbox[1])],
                    'score': np.float64(content)
                }
                predicts_merge.append(predict)
            else:
                break
        json_str_merge = json.dumps(predicts_merge, indent=4)
        with open('./predict_result_merge.json', 'w') as json_file:
            json_file.write(json_str_merge)

def cls_coco_ap_per_class(val_json,pre_json,i):
# def cls_coco_ap_per_class(val_json, pre_json):
    dts = json.load(open(pre_json,'r'))
    # dts = json.load(open(val_json, 'r'))
    imgIds = [imid['image_id'] for imid in dts]
    # imgIds = [imid['image_id'] for imid in dts['annotations']]

    imgIds = sorted(list(set(imgIds)))
    del dts
    coco_true = COCO(annotation_file=val_json)
    coco_pre = coco_true.loadRes(pre_json)
    coco_evaluator = COCOeval(cocoGt=coco_true,cocoDt=coco_pre,iouType='bbox')
    coco_evaluator.params.imgIds = imgIds
    # coco_evaluator.params.catIds = [0,1,2,3,4,5,6,7,8,9]
    coco_evaluator.params.catIds = [i]
    coco_evaluator.evaluate()
    coco_evaluator.accumulate()
    coco_evaluator.summarize()

def state_coco_ap_per_class(val_json,pre_json):
# def state_coco_ap_per_class(val_json, pre_json):
    dts = json.load(open(pre_json,'r'))
    # dts = json.load(open(val_json, 'r'))
    imgIds = [imid['image_id'] for imid in dts]
    # imgIds = [imid['image_id'] for imid in dts['annotations']]

    imgIds = sorted(list(set(imgIds)))
    del dts
    coco_true = COCO(annotation_file=val_json)
    coco_pre = coco_true.loadRes(pre_json)
    coco_evaluator = COCOeval(cocoGt=coco_true,cocoDt=coco_pre,iouType='bbox')
    coco_evaluator.params.imgIds = imgIds
    coco_evaluator.params.catIds = [0,1]
    # coco_evaluator.params.catIds = [i]
    coco_evaluator.evaluate()
    coco_evaluator.accumulate()
    coco_evaluator.summarize()
def merge_coco_ap_per_class(val_json,pre_json):
    dts = json.load(open(pre_json,'r'))
    # dts = json.load(open(val_json, 'r'))
    imgIds = [imid['image_id'] for imid in dts]
    # imgIds = [imid['image_id'] for imid in dts['annotations']]
    imgIds = sorted(list(set(imgIds)))
    del dts
    coco_true = COCO(annotation_file=val_json)
    coco_pre = coco_true.loadRes(pre_json)
    coco_evaluator = COCOeval(cocoGt=coco_true,cocoDt=coco_pre,iouType='bbox')
    coco_evaluator.params.imgIds = imgIds
    # coco_evaluator.params.catIds = [0]
    coco_evaluator.params.catIds = [0,1,2,3,4,5,6,7,8,9,10]
    # coco_evaluator.params.catIds = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]
    coco_evaluator.evaluate()
    coco_evaluator.accumulate()
    coco_evaluator.summarize()


def coco_map(val_json,pre_json):
    coco_true = COCO(annotation_file=val_json)
    coco_pre = coco_true.loadRes(pre_json)
    coco_evaluator = COCOeval(cocoGt=coco_true,cocoDt=coco_pre,iouType='bbox')
    coco_evaluator.evaluate()
    coco_evaluator.accumulate()
    coco_evaluator.summarize()

# def ls_scores(pre_json):
#     # dts = json.load(open(pre_json, 'r'))
#     # dts = json.load(open(val_json, 'r'))
#     # imgIds = [imid['image_id'] for imid in dts]
#     # imgIds = [imid['image_id'] for imid in dts['annotations']]
#     # print(len(dts['annotations']))
#
#     sum =  lower_30 = 0
#     dts = json.load(open(pre_json, 'r'))
#     for item in dts:
#         sum += 1
#         if item['score'] < 0.3:
#             lower_30 += 1
#     print(f'sum={sum}',f"lower_30={lower_30}",f"rate={lower_30/sum}")


if __name__ =="__main__":
    mode = 'coco_map'# predict_json coco_map
    num_state = 10
    thr = 0.3
    cfg = r'jiyan/rtmdet_s_syncbn_fast_8xb32-300e_coco.py'
    checkpoint = r'./jiyan/rtm_epoch_40.pth'
    image_root = './data/coco/images'

    model = init_detector(cfg, checkpoint, device='cuda:0')

    # 整个测试集关于cls的json文件
    val_cls = "./data/coco/annotations/val_cls.json"
    # 整个测试集关于state的json文件
    val_state = './data/coco/annotations/val_state.json'
    # 整个测试集关于小类（20个）的json文件
    val_merge = './data/coco/annotations/instances_val2017.json'

    if mode == 'predict_json':
        print('-------------------prepare json------------------------')
        predict_json(image_root,val_cls,model, thr)# val_cls
        print('-------------------end------------------------')

    elif mode == 'coco_map':
        # 新的AP
        # print('-------------------ap per for classes------------------------------')
        # cls_coco_ap_per_class(val_cls, './predict_result_cls.json')
        # print('-------------------ap per for state------------------------------')
        # state_coco_ap_per_class(val_state, './predict_result_state.json')
        # print('-------------------ap per for merge------------------------------')
        # merge_coco_ap_per_class(val_merge, './predict_result_merge.json')


        for i in range(10):
            print("\n\n")
            print(i)
            print("\n\n")
            cls_coco_ap_per_class(val_cls, './predict_result_cls.json',i)

        # ls_scores('./predict_result_cls.json')
        # ls_scores('./predict_result_state.json')
        # ls_scores('./predict_result_merge.json')
        # # ls_scores(val_merge)
        # print("\n")
        # ls_scores('./v8_predict_result_cls.json')
        # ls_scores('./v8_predict_result_state.json')
        # ls_scores('./v8_predict_result_merge.json')

