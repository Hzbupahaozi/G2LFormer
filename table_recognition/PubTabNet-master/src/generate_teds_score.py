import os
import json
import time
import pickle
from metric import TEDS
from multiprocessing import Pool

def htmlPostProcess(text):
    text = '<html><body><table>' + text + '</table></body></html>'
    return text

def singleEvaluation(teds, file_name, context, gt_context):
    # save problem log
    # save_folder = ''

    # html format process
    htmlContext = htmlPostProcess(context)
    htmlGtContext = htmlPostProcess(gt_context)
    # Evaluate
    score = teds.evaluate(htmlContext, htmlGtContext)

    print("FILENAME : {}".format(file_name))
    print("SCORE    : {}".format(score))
    return score

if __name__=="__main__":
    t_start = time.time()
    pool = Pool(64)
    start_time = time.time()
    # # WTW
    predFile = '/home/chs/tablemaster-mmocr/work_dir_chs_wtw2024/work_dir_chs_wtw1218/results_90/structure_master_results.pkl'
    gtJsonFile = '/home/chs/tablemaster-mmocr/wtw_html.json'
    
    # HWL
    # predFile = '../../../work_dir_chs_hwl1104/results/structure_master_results.pkl'
    # gtJsonFile = '../../../hwl_val.json'

     # Initialize TEDS object
    teds = TEDS(n_jobs=1)
    with open(predFile, 'rb') as f:
        predDict = pickle.load(f)

    with open(gtJsonFile, 'r') as f:
        gtValDict = json.load(f)
        new_gtValDict = {}
        for key, value in gtValDict.items():
            key = key.replace('.png','.jpg')
            new_gtValDict[key] = value

    print('lenth of preddict',len(predDict))
    print('lenth of gtvaldict',len(gtValDict))
    # print(new_gtValDict)
    scores = []
    caches = dict()

    for idx, (file_name, context) in enumerate(predDict.items()):
        # loading
        # file_name = os.path.basename(file_path)
        # print(file_name)
        if file_name in new_gtValDict:
            gt_context = new_gtValDict[file_name]
            # print(file_name)
            context_str = ''.join(context.get('text')).replace(",","").replace('"','\\"')
            # print(gt_context)
            gt_context_str = ''.join(gt_context).replace(",","").replace('"','\\"')
            # print(gt_context_str)
            # print('context',context_str)
            # print('gt_context',gt_context_str)
            score = pool.apply_async(func=singleEvaluation, args=(teds, file_name, context_str, gt_context_str,))
            scores.append(score)
            # print('length:',len(scores))
            tmp = {'score':score, 'gt':gt_context_str, 'pred':context_str}
            caches.setdefault(file_name, tmp)
        else:
            # print('pass file')
            continue

    pool.close()
    pool.join() # 进程池中进程执行完毕后再关闭，如果注释，那么程序直接关闭。
    pool.terminate()

    # get score from scores
    cal_scores = []
    for score in scores:
        cal_scores.append(score.get())
    # print('length of valP:', len(cal_scores))
    print('AVG TEDS score: {}'.format(sum(cal_scores)/len(cal_scores)))
    print('TEDS cost time: {}s'.format(time.time()-start_time))

    print("Save cache for analysis.")
    save_folder = '../../../work_dir_chs_wtw2024/work_dir_chs_wtw1218/teds_score_all'
    for file_name in caches.keys():
        info = caches[file_name]
        if info['score']._value < 2.0:
            f = open(os.path.join(save_folder, file_name.replace('.jpg', '.txt')), 'w')
            f.write(file_name+'\n'+'\n')
            f.write('Score:'+'\n')
            f.write(str(info['score']._value)+'\n'+'\n')
            f.write('Pred:'+'\n')
            f.write(str(info['pred'])+'\n'+'\n')
            f.write('Gt:' + '\n')
            f.write(str(info['gt'])+'\n'+'\n')