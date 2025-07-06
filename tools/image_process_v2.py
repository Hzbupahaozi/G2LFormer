import linecache
import numpy as np
import random
from numpy import dtype
import os
import json
from tqdm import tqdm
from PIL import Image
import shutil
import xml.etree.ElementTree as ET
from os import getcwd
import xml.dom.minidom
# file1 = os.listdir('/data/zml/mmocr_WTW_recognition/train')
# txt_file = os.listdir('/data/zml/mmocr_WTW_recognition/StructureLabelAddEmptyBbox_train')
# for file in txt_file:
#     # flex = file[-6:]
#     # print(flex)
#     img_name = file.replace('.txt','.jpg')
#     # file1 = file.replace('.jpg','.txt')
#     if(img_name not in file1):
#         os.remove("/data/zml/mmocr_WTW_recognition/StructureLabelAddEmptyBbox_train/"+file)

txt_path = '/data/zml/mmocr_WTW_recognition/StructureLabelAddEmptyBbox_train'
image_path = '/data/zml/mmocr_WTW_recognition/hard1'
json_out_path = '/data/zml/mmocr_WTW_recognition/json'
output_path ='/data/zml/mmocr_WTW_recognition/outimg'
img_list = os.listdir(image_path)
# cutform(txt路径,image路径,json输出路径,image输出路径)

def span_num(caps,cur_row,cur_col):
    maxrow,maxcol = caps.shape[0],caps.shape[1]
    row_span,col_span = 0,0
    for row in range(cur_row,maxrow):
        if (caps[row][cur_col]!=caps[cur_row][cur_col]):
            row_span = row - cur_row
            break
        elif row == maxrow-1 :
            row_span = row - cur_row+1
    for col in range (cur_col,maxcol):
        if (caps[cur_row][col]!=caps[cur_row][cur_col]):
            col_span = col - cur_col
            break
        elif col == maxcol-1 :
            col_span = col - cur_col+1
    return row_span,col_span

def occupy(caps, startrow, startcol, rowspan, colspan, id):
    '''
    填充占用矩阵
    input:
    caps
    startrow
    startcol
    rowspan
    colspan
    output:
    cap
    '''
    for i in range(startrow, startrow + rowspan):
        for j in range(startcol, startcol + colspan):
            caps[i][j] = 1
    return caps

def annotation(xml_id):
    # 打开xml文档
    dom = xml.dom.minidom.parse('/data/zml/mmocr_WTW_recognition/xml/%s'%(xml_id))
    # 得到文档元素对象
    root = dom.documentElement
    itemlist = root.getElementsByTagName('object')
    # print(root.nodeName)
    # print(root.nodeValue)
    # print(root.nodeType)
    
    data = {}
    for item in itemlist:
        # print(item.nodeName)
        end_col = item.getElementsByTagName('endcol')
        ec = end_col[0].firstChild.data
        end_row = item.getElementsByTagName('endrow')
        er = end_row[0].firstChild.data
        start_col = item.getElementsByTagName('startcol')
        sc = start_col[0].firstChild.data
        start_row = item.getElementsByTagName('startrow')
        sr = start_row[0].firstChild.data
        tableid = item.getElementsByTagName('tableid')
        id = tableid[0].firstChild.data
        x1 = item.getElementsByTagName('x1')
        x1 = x1[0].firstChild.data
        x2 = item.getElementsByTagName('x2')
        x2 = x2[0].firstChild.data
        x3 = item.getElementsByTagName('x3')
        x3 = x3[0].firstChild.data
        x4 = item.getElementsByTagName('x4')
        x4 = x4[0].firstChild.data
        y1 = item.getElementsByTagName('y1')
        y1 = y1[0].firstChild.data
        y2 = item.getElementsByTagName('y2')
        y2 = y2[0].firstChild.data
        y3 = item.getElementsByTagName('y3')
        y3 = y3[0].firstChild.data
        y4 = item.getElementsByTagName('y4')
        y4 = y4[0].firstChild.data
       
        if id not in data:
            data[id] = []
        sr,sc,er,ec = int(sr),int(sc),int(er),int(ec)
        tmp=(sr,sc,er,ec,x1,y1,x2,y2,x3,y3,x4,y4) 
        data[id].append(tmp)                                                                                                                                                                                                                                                       
    # data = sorted(data, key=lambda x: (x[0],x[1]))
    return data

def structure(xml_id):
        # if(id!= "customs-declaration-02175..xml"):    continue    #  检查某个文件
        print(xml_id)
        xml_id  = xml_id.split('_')
        print(xml_id)
        index = xml_id[1][:-5]
        xml_id = xml_id[0]+"..xml"
        data = annotation(xml_id)   
        print(data.keys())
        print(index)
        for i in data:      #一张图片有多个表格，i表示第几个
            if(i != index): continue
            s = "_"+i+'.jpg'
            filename = ''
            tmp = xml_id.replace('..xml',s)
            tmp = tmp.split(' ')
            d = {}
            for s in tmp:
                filename = filename+s
            print(filename)
            # print(xml_id, i)
            logic = []
            maxrows,maxcols = 0 , 0
            cells = []
            # for item in data[i]:
            #     print(item)
            # print("done")
            data[i] = sorted(data[i],key = lambda x:(x[0],x[1]))
            # for item in data[i]:
            #     print(item)
            xmin,xmax,ymin,ymax = 1e5,-1, 1e5,-1
                
            for item in data[i]:
                tmp = [int(float(it)) for it in item]
                sr,sc ,er,ec = tmp[0:4]
                t={}
                t["start_row"],t["start_col"],t["end_row"],t["end_col"]= sr,sc ,er,ec
                logic.append(t)
                # logic_save(logic,s)
                bbox = tmp[4:]
                bbox = [int(float(i)) for i in bbox]
                tmp = min(bbox[::2])
                xmin = min(xmin,tmp)
                tmp = max(bbox[::2])
                xmax = max(xmax,tmp)
                tmp = min(bbox[1::2])
                ymin = min(ymin,tmp)
                tmp = max(bbox[1::2])
                ymax = max(ymax,tmp)
                maxrows,maxcols = max(maxrows,er),max(maxcols,ec)
                cells.append([sr,sc,er,ec]+bbox)
            maxrows+=1
            maxcols+=1
            cell_map = np.zeros((maxrows,maxcols)) 
            cell = [ [[]for i in range(maxcols)]for i in range(maxrows)]  
            num = 0
            for item in cells:
                tmp = dict()
                n = len(item)
                #cut
                for k in range(4,n):
                    if(k%2):
                        item[k] = max(item[k]-ymin,0)
                    else:
                        item[k] = max(item[k]-xmin,0)

            
            for item in cells:
                num+=1
                sr,sc ,er,ec = item[0:4]
                # print(sr,sc ,er,ec)
                for row in range(sr,er+1):
                    if(ec<sc): ec=sc
                    for col in range(sc,ec+1):
                        if(cell_map[row][col]!=0): 
                            wrong_list.append(filename)
                        else:
                            cell_map[row][col]= num
                            cell[row][col] = item[4:]
            # print(cell_map,cell)
            # break
            
            return cell_map,cell
            
        # print(filename)
        
    #         with open(os.path.join(save_path,filename.replace("jpg","json")), 'w') as f:
    #             json.dump(d,f)
    #         html_list[filename] = struct
    # return 

def checkcol(caps, currow, curcol):
    '''
    检查该列是否已被其他跨行跨列单元格占用
    input：
    caps 占用矩阵
    currow 当前行
    curcol 当前列
    output：
    curcol 当前行合法列
    '''
    maxcols = len(caps[0])
    # print(currow,curcol)
    while curcol<maxcols and caps[currow][curcol]!=[]:
        curcol += 1
    return curcol

def cutform(txtpath,imagepath,jsonoutpath,imageoutpath):
    filename = os.listdir(txtpath)
    tab_num = 0
    for fn in tqdm(filename):
        f = linecache.getline(os.path.join(txtpath, fn), 2)
        p = random.random()
        # print(p)
        # print(fn)
        if(fn.replace('txt','jpg') not in img_list): continue
        if(p<0.3):
            # shutil.copy("/home/Dataset/WTW/data/zml/mmocr_WTW_recognition/"+fn,"/home/Dataset/WTW/data/zml/train")    
            # print("no",p)
            continue
        # if(fn!="customs-declaration-15963_0.txt"): continue
        
        # print(f)
        print(fn)
        # cell_map,cells = structure(fn.replace(".txt","..xml"))
        my_list = f.split(",")
        row_num = 0
        col_num = 0
        # print(cell_map)
        # print(cells)
        # print(my_list)
        #row_num  , col_num
        for a in range(len(my_list)):
            if my_list[a] == '<tr>':
                 row_num = row_num+1

        for a in range(len(my_list)):
            if my_list[a] == '<td></td>':
                col_num = col_num+1
            elif my_list[a] == '>':
                item = my_list[a-1]
                if item.find('colspan') != -1:
                    item_filter = filter(str.isdigit, item)
                    item_list = list(item_filter)
                    item_str = "".join(item_list)
                    item_int = int(item_str)
                    col_num = col_num+item_int
                elif item.find('rowspan') != -1:
                    col_num = col_num+1
            elif my_list[a] == '</tr>':
                break

        row_martix = np.ones((row_num, col_num), dtype=np.int)
        col_martix = np.ones((row_num, col_num), dtype=np.int)
        annotation_path = os.path.join(txtpath, fn)
        with open(annotation_path) as f:
             lines = f.readlines()
        html = lines[1]
        data = []
        for line in lines[2:]:
            data.append(line)
        # print("data:",data)

        # match cell to table
        d = [ [[]for i in range(col_num)]for i in range(row_num)]  
        row,col = -1,-1
        num = -1
        html = html.split(",")
        # print(html)
        i,n  =0,len(html)
        print(row_num,col_num)
        while(i<n):
            # print(html[i])
            if(html[i]=="<tr>"):
                row+=1
                col =0
            elif(html[i] == "<td></td>"):
                col = checkcol(d, row, col)
                num+=1
                # print("1:",num,row,col)
                d[row][col].append(data[num])
            elif(html[i]== "<td"):
                rowspan,colspan =1,1
                if(html[i+1].find("rowspan")>0):
                    rowspan = int(html[i+1].split("=")[1][1:-1])
                    i+=1
                if(html[i+1].find("colspan")>0):
                    colspan = int(html[i+1].split("=")[1][1:-1])
                num+=1
                col = checkcol(d, row, col)
                # print(num,rowspan,colspan,row,col)
                for j in range(rowspan):
                    for k in range(colspan):
                        # print(row+j,k+col)
                        d[row+j][k+col] = [data[num]]
            i+=1
        # for i in range(col_num): print(d[0][i])
        # for i in range(row_num):
        #     for j in range(col_num):
        #         if(d[i][j]==[]):
        #             print("wrong:",i,j)
        i = 0
        j = 0
        m = 0
        n = 0
        for k in range(len(my_list)):
            if my_list[k] == '<tr>':
                i = i+1
            elif my_list[k] == '<td></td>':
                while row_martix[i - 1][j] == 0:
                    j = j + 1
                j = j+1
            elif my_list[k] == '>':
                item_row_martix_col = my_list[k - 1]
                item_row_martix_row = my_list[k - 1]
                item_row_martix_row_front=my_list[k - 2]
                if item_row_martix_col.find('colspan') != -1 and item_row_martix_row_front.find('rowspan') != -1:
                    item_row_martix_row_front_filter = filter(str.isdigit, item_row_martix_row_front)
                    item_row_martix_row_front_list = list(item_row_martix_row_front_filter)
                    item_row_martix_row_front_str = "".join(item_row_martix_row_front_list)
                    item_row_martix_row_front_int = int(item_row_martix_row_front_str)
                    item_row_martix_col_filter = filter(str.isdigit, item_row_martix_row)
                    item_row_martix_col_list = list(item_row_martix_col_filter)
                    item_row_martix_col_str = "".join(item_row_martix_col_list)
                    item_row_martix_col_int = int(item_row_martix_col_str)
                    row_martix[i-1][j] = item_row_martix_row_front_int
                    for x in range(j+1, j+item_row_martix_col_int):
                        row_martix[i-1][x] =  item_row_martix_row_front_int
                    for x in range(i, i+item_row_martix_row_front_int-1):
                        for y in range(j,j+item_row_martix_col_int):
                            row_martix[i][j] = 0
                    j = j+item_row_martix_col_int
                elif item_row_martix_row.find('rowspan') != -1:
                    item_row_martix_row_filter = filter(str.isdigit, item_row_martix_row)
                    item_row_martix_row_list = list(item_row_martix_row_filter)
                    item_row_martix_row_str = "".join(item_row_martix_row_list)
                    item_row_martix_row_int = int(item_row_martix_row_str)
                    j = j + 1
                    row_martix[i-1][j-1] = item_row_martix_row_int
                    for x in range(i, i+item_row_martix_row_int-1):
                        row_martix[x][j-1] = 0
            elif my_list[k] == '</tr>':
                j = 0
            elif my_list[k] == '<td':
                while row_martix[i - 1][j] == 0:
                    j = j + 1
            elif my_list[k].find(('colspan')) !=-1:
                continue
            elif my_list[k].find(('rowspan')) !=-1:
                continue
            elif my_list[k] =='</td>' :
                continue


        for k in range(len(my_list)):
            if my_list[k] == '<tr>':
                m = m+1
            elif my_list[k] == '<td></td>':
                while row_martix[m - 1][n] == 0:
                    n = n + 1
                n = n+1
            elif my_list[k] == '>':
                item_col_martix_col = my_list[k - 1]
                item_col_martix_row = my_list[k - 1]
                item_col_martix_row_front = my_list[k - 2]
                if item_col_martix_col.find('colspan') != -1 and item_col_martix_row_front.find('rowspan') == -1:
                    item_col_martix_col_filter = filter(str.isdigit, item_col_martix_row)
                    item_col_martix_col_list = list(item_col_martix_col_filter)
                    item_col_martix_col_str = "".join(item_col_martix_col_list)
                    item_col_martix_col_int = int(item_col_martix_col_str)
                    col_martix[m-1][n] = item_col_martix_col_int
                    for x in range(n+1, n + item_col_martix_col_int ):
                        col_martix[m-1][x] = 0
                    n = n + item_col_martix_col_int
                elif item_col_martix_col.find('colspan') != -1 and item_col_martix_row_front.find('rowspan') != -1:
                    item_col_martix_row_front_filter = filter(str.isdigit, item_col_martix_row_front)
                    item_col_martix_row_front_list = list(item_col_martix_row_front_filter)
                    item_col_martix_row_front_str = "".join(item_col_martix_row_front_list)
                    item_col_martix_row_front_int = int(item_col_martix_row_front_str)
                    item_col_martix_col_filter = filter(str.isdigit, item_col_martix_row)
                    item_col_martix_col_list = list(item_col_martix_col_filter)
                    item_col_martix_col_str = "".join(item_col_martix_col_list)
                    item_col_martix_col_int = int(item_col_martix_col_str)
                    col_martix[m - 1][n] = item_col_martix_col_int
                    for x in range(m, m+item_col_martix_row_front_int-1 ):
                        col_martix[x][n] = item_col_martix_col_int
                    for x in range(m,m+item_col_martix_row_front_int-1):
                        for y in range(n+1,n+item_col_martix_col_int):
                            col_martix[x][y] = 0
                    n=n+item_col_martix_col_int
               
            elif my_list[k] == '</tr>':
                n = 0
            elif my_list[k] == '<td':
                while row_martix[m - 1][n] == 0:
                    n = n + 1
            elif my_list[k].find(('colspan')) != -1:
                continue
            elif my_list[k].find(('rowspan')) != -1:
                continue
            elif my_list[k] == '</td>':
                continue

        row_martix_row_index = []
        row_martix_col_index = []
        col_martix_row_index = []
        col_martix_col_index = []
        ones_row_matrix = np.ones((1, row_num), dtype=np.int)
        ones_col_matrix = np.ones((1, col_num), dtype=np.int)

        #可裁切的index,  row与col取交集
        for e in np.arange(len(row_martix)):                    
            if np.all(row_martix[e] == ones_col_matrix[0]):
                row_martix_row_index.append(e)
        for e in np.arange(len(row_martix.T)):
            if np.all(row_martix.T[e] == ones_row_matrix[0]):
                row_martix_col_index.append(e)
        for e in np.arange(len(col_martix)):
            if np.all(col_martix[e] == ones_col_matrix[0]):
                col_martix_row_index.append(e)

        for e in np.arange(len(col_martix.T)):
            if np.all(col_martix.T[e] == ones_row_matrix[0]):
                col_martix_col_index.append(e)
        cut_index = []
        cut_row_num = -1
        if(row_num != 1):
            cut_index = [x for x in row_martix_row_index if x in col_martix_row_index]
            cut_row_num = len(cut_index)-1
        if(col_num!=1):
            for g in row_martix_col_index:
                if g in col_martix_col_index:
                    cut_index.append(g)

        cut_num = len(cut_index)
        if cut_num == 0:#没有可裁剪的行或列，直接输出json文件
            print("There is no col or row can be cut!")
            # print(row_martix)
            # print(col_martix)

            newfn = ' cut' + fn
            # jsonfn = cut.strip('.txt') + '.json'

            # with open(os.path.join(txtpath, fn), 'r', encoding='utf-8') as f:  # 打开txt文件
            #     for line in f:
            #         d = {}
            #         d['content'] = line.rstrip('\n')
            #         with open(os.path.join(jsonoutpath, jsonfn), 'a', encoding='utf-8') as file:  # 创建一个json文件，mode设置为'a'
            #             json.dump(d, file,
            #                       ensure_ascii=False)
            #             file.write('\n')
            continue



        #random delete
        p = random.random()
        if(cut_row_num>=0 and cut_num-1>cut_row_num):
            print("number:",cut_row_num,cut_num)
            if(p>=0.5):
                cut =  random.randint(cut_row_num+1, cut_num-1)
            else:
                cut = random.randint(0,cut_row_num)
        else:
            cut = random.randint(0, cut_num-1)
        if cut <= cut_row_num:
            changed_row_martix = np.delete(row_martix, cut_index[cut], axis=0)
            changed_col_martix = np.delete(col_martix, cut_index[cut], axis=0)
        elif cut > cut_row_num:
            changed_row_martix = np.delete(row_martix, cut_index[cut], axis=1)
            changed_col_martix = np.delete(col_martix, cut_index[cut], axis=1)

        zeros_col_matrix = np.zeros((1, changed_row_martix.shape[1]), dtype=np.int)
        zeros_row_matrix = np.zeros((1, changed_col_martix.shape[0]), dtype=np.int)
        restart = True

        while restart:
            restart = False
            for e in np.arange(len(changed_row_martix)):
                if np.all(changed_row_martix[e] == zeros_col_matrix[0]):
                    changed_row_martix = np.delete(changed_row_martix, e, axis=0)
                    changed_col_martix = np.delete(changed_col_martix, e, axis=0)
                    for g in range(0, e):
                        for h in range(0, changed_col_martix.shape[1]):
                            if changed_row_martix[g][h] > 1:
                                changed_row_martix[g][h] = changed_row_martix[g][h]-1
                    restart = True
                    break

        while restart:
            restart = False
            for e in np.arange(len(changed_col_martix.T)):
                if np.all(changed_col_martix.T[e] == zeros_row_matrix[0]):
                    changed_row_martix = np.delete(changed_row_martix, e, axis=1)
                    changed_col_martix = np.delete(changed_col_martix, e, axis=1)
                    for g in range(0, changed_row_martix.shape[0]):
                        for h in range(0, e):
                            if changed_row_martix[g][h] > 1:
                                changed_row_martix[g][h] = changed_row_martix[g][h]-1
                    restart=True
                    break


        #htmltxt_save
        newfn = 'cut' + fn
        file = open(os.path.join(jsonoutpath, newfn), "w", encoding="utf8")
        # file.close()
        for e in range(0, changed_row_martix.shape[0]) :
            for g in range(0, changed_col_martix.shape[1]) :
                # file = open(os.path.join(jsonoutpath, newfn), "a", encoding="utf8")
                if g == 0:
                    file.write("<tr>,")
                if changed_row_martix[e][g] == 1 and changed_col_martix[e][g] == 1:
                    file.write("<td></td>,")
                elif changed_row_martix[e][g] > 1 and changed_col_martix[e][g] == 1:
                    file.write('<td,"rowspan={}",>,</td>,'.format(changed_row_martix[e][g]))
                elif changed_col_martix[e][g] > 1 and changed_row_martix[e][g] == 1:
                    file.write('<td,"colspan={}",>,</td>,'.format(changed_col_martix[e][g]))         
                elif changed_col_martix[e][g] > 1 and changed_row_martix[e][g] > 1:
                     file.write('<td,"rowspan={}","colspan={}",>,</td>,'.format(changed_row_martix[e][g],changed_col_martix[e][g]))
                # file.close()
            # file = open(os.path.join(jsonoutpath, newfn), "a", encoding="utf8")
            # if e < len(changed_row_martix) - 1:
            #     file.write("</tr>,")
            #     file.close()
            # else:
            file.write("</tr>")
        file.write("\n")   
        file.close()

        # print("row_martix_row_index:",row_martix_row_index)
        # print("row_martix_col_index:",row_martix_col_index)
        # print("col_martix_row_index:",col_martix_row_index)
        # print("col_martix_col_index:",col_martix_col_index)
        print("cut_index:",cut_index)
        # print("row_martix:",row_martix)
        # print("col_martix:",col_martix)
        # print("changed_row_martix:",changed_row_martix)
        # print("changed_col_martix:",changed_col_martix)
        


        cut_indexnum = cut_index[cut]
        firstgrid = linecache.getline(os.path.join(txtpath, fn), 3)
        firstgrid_bbox = firstgrid.split(",")
        print("cut:",cut,cut_row_num,cut_indexnum)
        print("row_num:",row_num)
        print("col_num:",col_num)

        with open(os.path.join(txtpath, fn), 'r') as fp:
            lines = fp.readlines()
            last_line = lines[-1]
        lastgrid_bbox = last_line.split(",")
        imagefn=fn.replace("txt",'jpg')
        newimagefn =newfn.replace('.txt','.jpg')
        
        if cut <= cut_row_num:  #裁行
            # if cut_index[cut] > 0 and cut_index[cut] < row_num-1:   #裁中间
                # grid_row_index = 0
                # for e in range(0, cut_index[cut]):      #因为存在跨行跨列所以只能逐行搜索
                #     for g in range(0, col_num):
                #         if row_martix[e][g] >= 1 and col_martix[e][g] >= 1:
                #             grid_row_index = grid_row_index+1
                up_lastgrid  = d[cut_indexnum][0][0] 
                # up_lastgrid = linecache.getline(os.path.join(txtpath, fn), 3+grid_row_index)
                up_lastgrid_bbox = up_lastgrid.split(",")
              
                down_firstgrid = d[cut_indexnum][0][0]
                # print(d[cut_indexnum][0][0])
                # down_firstgrid = linecache.getline(os.path.join(txtpath, fn), 3+grid_row_index+col_num)
                down_firstgrid_bbox = down_firstgrid.split(",")
                print("image_path:",os.path.join(imagepath, imagefn),imagefn,fn)
                image = Image.open(os.path.join(imagepath, imagefn))
                up_remaining_region = image.crop((0, 0, image.width, int(up_lastgrid_bbox[1])))
                down_remaining_region = image.crop((0, int(down_firstgrid_bbox[3]), image.width, image.height))
                new_width = up_remaining_region.width
                new_height = up_remaining_region.height + down_remaining_region.height
                new_image = Image.new('RGB', (new_width, new_height))
                if(up_remaining_region.height>0):
                    new_image.paste(up_remaining_region, (0, 0))
                if(down_remaining_region.height>0):
                    new_image.paste(down_remaining_region, (0, up_remaining_region.height))
                new_image.save(os.path.join(imageoutpath, newimagefn))
                cut_firstgrid = d[cut_indexnum][0][0]
                cut_firstgrid_bbox = cut_firstgrid.split(",")
                # print("cut_firstgrid_bbox:",cut_firstgrid_bbox)
                cut_lastgrid = d[cut_indexnum][col-1][0]
                cut_lastgrid_bbox = cut_lastgrid.split(",")
                # print("cut_lastgrids_bbox:",cut_lastgrid_bbox)
                h = int(cut_lastgrid_bbox[3])-int(cut_firstgrid_bbox[1])
               
                bbox_dict = {}
                num = 0
                file = open(os.path.join(jsonoutpath, newfn), "a", encoding="utf8")
                for  row in range(row_num) :
                    for col in range(col_num) :
                        # print(d[row][col])
                        grid_bbox = d[row][col][0]
                        if(grid_bbox  not in bbox_dict):
                            bbox_dict[grid_bbox ] =1
                        else: continue
                        if(row<cut_num):
                            num+=1
                            grid_bbox = d[row][col][0].split(',')
                            grid_bbox[3] = str(int(grid_bbox[3]))
                            # print(grid_bbox)
                            file.write(
                                "{},{},{},{}\n".format(grid_bbox[0], grid_bbox[1], grid_bbox[2], grid_bbox[3]))
                        elif row>col_num:
                            num+=1
                            grid_bbox = d[row][col][0].split(',')
                            # print(grid_bbox)
                            grid_bbox[1] = str(int(grid_bbox[1]) - h)
                            grid_bbox[3] = str(int(grid_bbox[3]) - h)
                                
                            file.write(
                                "{},{},{},{}\n".format(grid_bbox[0], grid_bbox[1], grid_bbox[2], grid_bbox[3]))
        if cut > cut_row_num:   #裁列
            # if cut_index[cut] > 0 and cut_index[cut] < col_num-1:
               
                right_firstgrid = d[0][cut_indexnum][0]#linecache.getline(os.path.join(txtpath, fn), 3 + grid_col_index)
                right_firstgrid_bbox = right_firstgrid.split(",")
                left_lastgrid = d[0][cut_indexnum][0]#linecache.getline(os.path.join(txtpath, fn), 3 + grid_col_index) 
                left_lastgrid_bbox = left_lastgrid.split(",")
                w = int(left_lastgrid_bbox[2]) - int(left_lastgrid_bbox[0])
                image = Image.open(os.path.join(imagepath, imagefn))
                left_remaining_region = image.crop((0, 0, int(left_lastgrid_bbox[0]), image.height))
                right_remaining_region = image.crop((int(right_firstgrid_bbox[2]), 0, image.width, image.height))
                new_width = left_remaining_region.width+right_remaining_region.width
                new_height = left_remaining_region.height
                new_image = Image.new('RGB', (new_width, new_height))
                if(left_remaining_region.width>0):
                    new_image.paste(left_remaining_region, (0, 0))
                if(right_remaining_region.width>0):
                    new_image.paste(right_remaining_region, (left_remaining_region.width, 0))
                new_image.save(os.path.join(imageoutpath, newimagefn))
                
                bbox_dict = {}
                num = 0
                file = open(os.path.join(jsonoutpath, newfn), "a", encoding="utf8")
                for  row in range(row_num) :
                    for col in range(col_num) :
                        grid_bbox = d[row][col][0]
                        if(grid_bbox  not in bbox_dict):
                            bbox_dict[grid_bbox ] =1
                        else: continue
                        if(col<cut_num):
                            num+=1
                            grid_bbox = d[row][col][0].split(',')
                            # print(grid_bbox)
                            grid_bbox[3] = str(int(grid_bbox[3]))
                            file.write(
                                "{},{},{},{}\n".format(grid_bbox[0], grid_bbox[1], grid_bbox[2], grid_bbox[3]))
                        elif col>col_num:
                            num+=1
                            grid_bbox = d[row][col][0].split(',')
                            # print(grid_bbox)
                            grid_bbox[0] = str(int(grid_bbox[0]) - w)
                            grid_bbox[2] = str(int(grid_bbox[2]) - w)
                            grid_bbox[3] = str(int(grid_bbox[3]))
                                
                            file.write(
                                "{},{},{},{}\n".format(grid_bbox[0], grid_bbox[1], grid_bbox[2], grid_bbox[3]))
                # print("num:",num)
           
        #json save
        jsonfn = newfn.strip('.txt') + '.json'

        # with open(os.path.join(jsonoutpath, newfn), 'r', encoding='utf-8') as f:  # 打开txt文件
        #     for line in f:
        #         d = {}
        #         d['content'] = line.rstrip('\n')
        #         with open(os.path.join(jsonoutpath, jsonfn), 'a', encoding='utf-8') as file:
        #             json.dump(d, file, ensure_ascii=False)
        #             file.write('\n')

        # if os.path.exists(os.path.join(jsonoutpath, newfn)):
        #     os.remove(os.path.join(jsonoutpath, newfn))
        tab_num+=1
        # if(tab_num==2):
        #     break 

# for roots,dirs,files in os.walk('/data/zml/mmocr_WTW_recognition/xml/'):
#         xml_root = roots
#         xml_ids =  files
# for item in xml_ids:
#     structure(item)
#     break
cutform(txt_path,image_path,json_out_path,output_path)   