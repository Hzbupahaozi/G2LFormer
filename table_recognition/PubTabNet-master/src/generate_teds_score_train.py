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
    # predFile = '/home/chs/tablemaster-mmocr/work_dir_chs_honor0320/structure_master_results.pkl'
    # predFile = '/home/chs/tablemaster-mmocr/work_dir_chs_pubtabnet0228/results_19/structure_master_results.pkl'
    predFile = '/home/chs/tablemaster-mmocr/work_dir_chs_tabrecset0612/results_80/structure_master_results.pkl'
    # gtJsonFile = '/home/chs/tablemaster-mmocr/wtw_html.json'
    # gtFile = '/data/chs/wtw/chs_wtw_train_txt'
    # gtFile = '/data/chs/honor/chs_honor_test_txt'
    gtFile = '/data/chs/tabrecset/StructureLabelAddEmptyBbox_test'
    
    # HWL
    # predFile = '../../../work_dir_chs_hwl1104/results/structure_master_results.pkl'
    # gtJsonFile = '../../../hwl_val.json'

     # Initialize TEDS object
    teds = TEDS(n_jobs=1)
    with open(predFile, 'rb') as f:
        predDict = pickle.load(f)

    scores = []
    caches = dict()

    for idx, (file_name, context) in enumerate(predDict.items()):
        # loading
        # file_name = os.path.basename(file_path)
        # print(file_name)
        # file_name = file_name.replace(".jpg",".txt")
        file_name = file_name.replace(".jpg",".txt")    # pubtabnet
        txt_path = os.path.join(gtFile, file_name)
        try:
            with open(txt_path,'r') as f:
                lines = f.readlines()
                train_context = lines[1].strip()
                # gt_context_str=''.join(train_context).replace("<td>,", "").replace(",", "").replace('"', '\\"') # pubtabnet
                gt_context_str=''.join(train_context).replace(",", "").replace('"', '\\"') 
                # print(file_name)
                # context_str = ''.join(context.get('text')).replace("<UKN>,","").replace(",","").replace('"','\\"')  # pubtabnet
                context_str = ''.join(context.get('text')).replace(",","").replace('"','\\"')  
                # print('context',context_str)
                # print('gt_context',gt_context_str)
                score = pool.apply_async(func=singleEvaluation, args=(teds, file_name, context_str, gt_context_str,))
                scores.append(score)
                # print('length:',len(scores))
                tmp = {'score':score, 'gt':gt_context_str, 'pred':context_str}
                caches.setdefault(file_name, tmp)
        except:
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
    save_folder = '../../../work_dir_chs_tabrecset0612/teds_score_80'
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