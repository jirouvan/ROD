import json
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

def cls_coco_ap_per_class(val_json,pre_json):
    dts = json.load(open(pre_json,'r'))
    imgIds = [imid['image_id'] for imid in dts]

    imgIds = sorted(list(set(imgIds)))
    del dts
    coco_true = COCO(annotation_file=val_json)
    coco_pre = coco_true.loadRes(pre_json)
    coco_evaluator = COCOeval(cocoGt=coco_true,cocoDt=coco_pre,iouType='bbox')
    coco_evaluator.params.imgIds = imgIds
    coco_evaluator.params.catIds = [0,1,2,3,4,5,6,7,8,9]
    # coco_evaluator.params.catIds = [2]
    coco_evaluator.evaluate()
    coco_evaluator.accumulate()
    coco_evaluator.summarize()
def state_coco_ap_per_class(val_json,pre_json):
    dts = json.load(open(pre_json,'r'))
    imgIds = [imid['image_id'] for imid in dts]

    imgIds = sorted(list(set(imgIds)))
    del dts
    coco_true = COCO(annotation_file=val_json)
    coco_pre = coco_true.loadRes(pre_json)
    coco_evaluator = COCOeval(cocoGt=coco_true,cocoDt=coco_pre,iouType='bbox')
    coco_evaluator.params.imgIds = imgIds
    coco_evaluator.params.catIds = [0,1]
    coco_evaluator.evaluate()
    coco_evaluator.accumulate()
    coco_evaluator.summarize()
def merge_coco_ap_per_class(val_json,pre_json):
    dts = json.load(open(pre_json,'r'))
    imgIds = [imid['image_id'] for imid in dts]

    imgIds = sorted(list(set(imgIds)))
    del dts
    coco_true = COCO(annotation_file=val_json)
    coco_pre = coco_true.loadRes(pre_json)
    coco_evaluator = COCOeval(cocoGt=coco_true,cocoDt=coco_pre,iouType='bbox')
    coco_evaluator.params.imgIds = imgIds
    coco_evaluator.params.catIds = sorted(coco_true.getCatIds())
    # coco_evaluator.params.catIds = [0, 1, 2, 3, 4]
    # coco_evaluator.params.catIds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    # coco_evaluator.params.catIds = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]
    coco_evaluator.evaluate()
    coco_evaluator.accumulate()
    coco_evaluator.summarize()


# 整个测试集关于物体种类的json文件
val_cls = "./data/coco/annotations/val_cls.json"
# 整个测试集关于遮挡方向的json文件
val_state = './data/coco/annotations/val_state.json'
# 整个测试集关于小类（20个）的json文件
val_merge = './data/coco/annotations/instances_val2017.json'

# 新的AP
print('-------------------ap per for classes------------------------------')
cls_coco_ap_per_class(val_cls, './predict_result_cls.json')
print('-------------------ap per for state------------------------------')
state_coco_ap_per_class(val_state, './predict_result_state.json')
print('-------------------ap per for merge------------------------------')
merge_coco_ap_per_class(val_merge, './predict_result_merge.json')

