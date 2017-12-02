import pickle
import os

def calculate_IoU(i0,i1):
    union=(min(i0[0],i1[0]) , max(i0[1],i1[1]))
    inter=(max(i0[0],i1[0]) , min(i0[1],i1[1]))
    iou=1.0*(inter[1]-inter[0])/(union[1]-union[0])
    return iou


it_path="./val_clip-sentvec.pkl"
sliding_clip_path="../Interval64_overlap0.5_c3d_fc6/"
cs=pickle.load(open(it_path))
clip_sentence_pairs=[]
for l in cs:
    clip_name=l[0]
    sent_vecs=l[1]
    for sent_vec in sent_vecs:
        clip_sentence_pairs.append((clip_name,sent_vec))
print len(clip_sentence_pairs)

sliding_clips_tmp=os.listdir(sliding_clip_path)
print len(sliding_clips_tmp)
iou_1=0
iou_2=0
iou_3=0
iou_4=0
iou_5=0
for clip_sentence in clip_sentence_pairs:
    sent_clip_name=clip_sentence[0]
    sent_movie_name=sent_clip_name.split("_")[0]
    max_iou=0
    for sliding_clip_name in sliding_clips_tmp:
        if sliding_clip_name.split(".")[2]!="npy": continue
        sliding_movie_name=sliding_clip_name.split("_")[0]
        if sliding_movie_name==sent_movie_name:
            sent_start=int(sent_clip_name.split("_")[1])
            sent_end=int(sent_clip_name.split("_")[2].split(".")[0])
            sliding_start=int(sliding_clip_name.split("_")[1])
            sliding_end=int(sliding_clip_name.split("_")[2].split(".")[0])
            iou=calculate_IoU((sent_start,sent_end),(sliding_start,sliding_end))
            if iou>max_iou: max_iou=iou
    if max_iou>0.1: iou_1+=1
    if max_iou>0.2: iou_2+=1 
    if max_iou>0.3: iou_3+=1
    if max_iou>0.4: iou_4+=1
    if max_iou>0.5: iou_5+=1
print "IoU=0.1, R@1: "+str(1.0*iou_1/len(clip_sentence_pairs))
print "IoU=0.2, R@1: "+str(1.0*iou_2/len(clip_sentence_pairs))
print "IoU=0.3, R@1: "+str(1.0*iou_3/len(clip_sentence_pairs))
print "IoU=0.4, R@1: "+str(1.0*iou_4/len(clip_sentence_pairs))
print "IoU=0.5, R@1: "+str(1.0*iou_5/len(clip_sentence_pairs))
