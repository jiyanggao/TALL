
import numpy as np
from math import sqrt
import os
import random
import pickle

'''
calculate temporal intersection over union
'''
def calculate_IoU(i0, i1):
    union = (min(i0[0], i1[0]), max(i0[1], i1[1]))
    inter = (max(i0[0], i1[0]), min(i0[1], i1[1]))
    iou = 1.0*(inter[1]-inter[0])/(union[1]-union[0])
    return iou

'''
calculate the non Intersection part over Length ratia, make sure the input IoU is larger than 0
'''
def calculate_nIoL(base, sliding_clip):
    inter = (max(base[0], sliding_clip[0]), min(base[1], sliding_clip[1]))
    inter_l = inter[1]-inter[0]
    length = sliding_clip[1]-sliding_clip[0]
    nIoL = 1.0*(length-inter_l)/length
    return nIoL

class TrainingDataSet(object):
    def __init__(self, sliding_dir, it_path, batch_size):
        
        self.counter = 0
        self.batch_size = batch_size
        self.context_num = 1
        self.context_size = 128
        print "Reading training data list from "+it_path
        cs = pickle.load(open(it_path))
        movie_length_info = pickle.load(open("./video_allframes_info.pkl"))
        self.clip_sentence_pairs = []
        for l in cs:
            clip_name = l[0]
            sent_vecs = l[1]
            for sent_vec in sent_vecs:
                self.clip_sentence_pairs.append((clip_name, sent_vec))

        movie_names_set = set()
        self.movie_clip_names = {}
        # read groundtruth sentence-clip pairs
        for k in range(len(self.clip_sentence_pairs)):
            clip_name = self.clip_sentence_pairs[k][0]
            movie_name = clip_name.split("_")[0]
            if not movie_name in movie_names_set:
                movie_names_set.add(movie_name)
                self.movie_clip_names[movie_name] = []
            self.movie_clip_names[movie_name].append(k)
        self.movie_names = list(movie_names_set)
        self.visual_feature_dim = 4096*3
        self.sent_vec_dim = 4800
        self.num_samples = len(self.clip_sentence_pairs)
        self.sliding_clip_path = sliding_dir
        print str(len(self.clip_sentence_pairs))+" clip-sentence pairs are readed"
        
        # read sliding windows, and match them with the groundtruths to make training samples
        sliding_clips_tmp = os.listdir(self.sliding_clip_path)
        self.clip_sentence_pairs_iou = []
        for clip_name in sliding_clips_tmp:
            if clip_name.split(".")[2]=="npy":
                movie_name = clip_name.split("_")[0]
                for clip_sentence in self.clip_sentence_pairs:
                    original_clip_name = clip_sentence[0] 
                    original_movie_name = original_clip_name.split("_")[0]
                    if original_movie_name==movie_name:
                        start = int(clip_name.split("_")[1])
                        end = int(clip_name.split("_")[2].split(".")[0])
                        o_start = int(original_clip_name.split("_")[1]) 
                        o_end = int(original_clip_name.split("_")[2].split(".")[0])
                        iou = calculate_IoU((start, end), (o_start, o_end))
                        if iou>0.5:
                            nIoL=calculate_nIoL((o_start, o_end), (start, end))
                            if nIoL<0.15:
                                movie_length = movie_length_info[movie_name.split(".")[0]]
                                start_offset =o_start-start
                                end_offset = o_end-end
                                self.clip_sentence_pairs_iou.append((clip_sentence[0], clip_sentence[1], clip_name, start_offset, end_offset))
        self.num_samples_iou = len(self.clip_sentence_pairs_iou)
        print str(len(self.clip_sentence_pairs_iou))+" iou clip-sentence pairs are readed"
       
    
    '''
    compute left (pre) and right (post) context features
    '''
    def get_context_window(self, clip_name, win_length):
        movie_name = clip_name.split("_")[0]
        start = int(clip_name.split("_")[1])
        end = int(clip_name.split("_")[2].split(".")[0])
        clip_length = self.context_size
        left_context_feats = np.zeros([win_length, 4096], dtype=np.float32)
        right_context_feats = np.zeros([win_length, 4096], dtype=np.float32)
        last_left_feat = np.load(self.sliding_clip_path+clip_name)
        last_right_feat = np.load(self.sliding_clip_path+clip_name)
        for k in range(win_length):
            left_context_start = start-clip_length*(k+1)
            left_context_end = start-clip_length*k
            right_context_start = end+clip_length*k
            right_context_end = end+clip_length*(k+1)
            left_context_name = movie_name+"_"+str(left_context_start)+"_"+str(left_context_end)+".npy"
            right_context_name = movie_name+"_"+str(right_context_start)+"_"+str(right_context_end)+".npy"
            if os.path.exists(self.sliding_clip_path+left_context_name):
                left_context_feat = np.load(self.sliding_clip_path+left_context_name)
                last_left_feat = left_context_feat
            else:
                left_context_feat = last_left_feat
            if os.path.exists(self.sliding_clip_path+right_context_name):
                right_context_feat = np.load(self.sliding_clip_path+right_context_name)
                last_right_feat = right_context_feat
            else:
                right_context_feat = last_right_feat
            left_context_feats[k] = left_context_feat
            right_context_feats[k] = right_context_feat
        return np.mean(left_context_feats, axis=0), np.mean(right_context_feats, axis=0)
    
    '''
    read next batch of training data, this function is used for training CTRL-aln
    '''
    def next_batch(self):
        
        random_batch_index = random.sample(range(self.num_samples), self.batch_size)
        image_batch = np.zeros([self.batch_size, self.visual_feature_dim])
        sentence_batch = np.zeros([self.batch_size, self.sent_vec_dim])
        offset_batch = np.zeros([self.batch_size, 2], dtype=np.float32) # this one is actually useless
        index = 0
        clip_set=set()
        while index < self.batch_size:
            k = random_batch_index[index]
            clip_name = self.clip_sentence_pairs[k][0]
            if not clip_name in clip_set: 
                clip_set.add(clip_name)
                feat_path = self.image_dir+self.clip_sentence_pairs[k][0]+".npy"
                featmap = np.load(feat_path)
                image_batch[index,:] = featmap
                sentence_batch[index,:] = self.clip_sentence_pairs[k][1][:self.sent_vec_dim]

                index+=1
            else:
                r = random.choice(range(self.num_samples))
                random_batch_index[index] = r
                continue 
                      
        return image_batch, sentence_batch, offset_batch

    '''
    read next batch of training data, this function is used for training CTRL-reg
    '''
    def next_batch_iou(self):

        random_batch_index = random.sample(range(self.num_samples_iou), self.batch_size)
        image_batch = np.zeros([self.batch_size, self.visual_feature_dim])
        sentence_batch = np.zeros([self.batch_size, self.sent_vec_dim])
        offset_batch = np.zeros([self.batch_size, 2], dtype=np.float32)
        index = 0
        clip_set = set()
        while index < self.batch_size:
            k = random_batch_index[index]
            clip_name = self.clip_sentence_pairs_iou[k][0]
            if not clip_name in clip_set:
                clip_set.add(clip_name)
                feat_path = self.sliding_clip_path+self.clip_sentence_pairs_iou[k][2]
                featmap = np.load(feat_path)
                # read context features
                left_context_feat, right_context_feat = self.get_context_window(self.clip_sentence_pairs_iou[k][2], self.context_num)
                image_batch[index,:] = np.hstack((left_context_feat, featmap, right_context_feat))
                sentence_batch[index,:] = self.clip_sentence_pairs_iou[k][1][:self.sent_vec_dim]
                p_offset = self.clip_sentence_pairs_iou[k][3]
                l_offset = self.clip_sentence_pairs_iou[k][4]
                offset_batch[index,0] = p_offset
                offset_batch[index,1] = l_offset
                index+=1
            else:
                r = random.choice(range(self.num_samples_iou))
                random_batch_index[index] = r
                continue
       
        return image_batch, sentence_batch, offset_batch


class TestingDataSet(object):
    def __init__(self, img_dir, csv_path, batch_size):
        #il_path: image_label_file path
        #self.index_in_epoch = 0
        #self.epochs_completed = 0
        self.batch_size = batch_size
        self.image_dir = img_dir
        print "Reading testing data list from "+csv_path
        self.semantic_size = 4800
        csv = pickle.load(open(csv_path))
        self.clip_sentence_pairs = []
        for l in csv:
            clip_name = l[0]
            sent_vecs = l[1]
            for sent_vec in sent_vecs:
                self.clip_sentence_pairs.append((clip_name, sent_vec))
        print str(len(self.clip_sentence_pairs))+" pairs are readed"
        movie_names_set = set()
        self.movie_clip_names = {}
        for k in range(len(self.clip_sentence_pairs)):
            clip_name = self.clip_sentence_pairs[k][0]
            movie_name = clip_name.split("_")[0]
            if not movie_name in movie_names_set:
                movie_names_set.add(movie_name)
                self.movie_clip_names[movie_name] = []
            self.movie_clip_names[movie_name].append(k)
        self.movie_names = list(movie_names_set)
        
        self.clip_num_per_movie_max = 0
        for movie_name in self.movie_clip_names:
            if len(self.movie_clip_names[movie_name])>self.clip_num_per_movie_max: self.clip_num_per_movie_max = len(self.movie_clip_names[movie_name])
        print "Max number of clips in a movie is "+str(self.clip_num_per_movie_max)
        
        self.sliding_clip_path = img_dir
        sliding_clips_tmp = os.listdir(self.sliding_clip_path)
        self.sliding_clip_names = []
        for clip_name in sliding_clips_tmp:
            if clip_name.split(".")[2]=="npy":
                movie_name = clip_name.split("_")[0]
                if movie_name in self.movie_clip_names:
                    self.sliding_clip_names.append(clip_name.split(".")[0]+"."+clip_name.split(".")[1])
        self.num_samples = len(self.clip_sentence_pairs)
        print "sliding clips number: "+str(len(self.sliding_clip_names))
        assert self.batch_size <= self.num_samples
        

    def get_clip_sample(self, sample_num, movie_name, clip_name):
        length=len(os.listdir(self.image_dir+movie_name+"/"+clip_name))
        sample_step=1.0*length/sample_num
        sample_pos=np.floor(sample_step*np.array(range(sample_num)))
        sample_pos_str=[]
        img_names=os.listdir(self.image_dir+movie_name+"/"+clip_name)
        # sort is very important! to get a correct sequence order
        img_names.sort()
       # print img_names
        for pos in sample_pos:
            sample_pos_str.append(self.image_dir+movie_name+"/"+clip_name+"/"+img_names[int(pos)])
        return sample_pos_str
    
    def get_context_window(self, clip_name, win_length):
        movie_name = clip_name.split("_")[0]
        start = int(clip_name.split("_")[1])
        end = int(clip_name.split("_")[2].split(".")[0])
        clip_length = 128#end-start
        left_context_feats = np.zeros([win_length,4096], dtype=np.float32)
        right_context_feats = np.zeros([win_length,4096], dtype=np.float32)
        last_left_feat = np.load(self.sliding_clip_path+clip_name)
        last_right_feat = np.load(self.sliding_clip_path+clip_name)
        for k in range(win_length):
            left_context_start = start-clip_length*(k+1)
            left_context_end = start-clip_length*k
            right_context_start = end+clip_length*k
            right_context_end = end+clip_length*(k+1)
            left_context_name = movie_name+"_"+str(left_context_start)+"_"+str(left_context_end)+".npy"
            right_context_name = movie_name+"_"+str(right_context_start)+"_"+str(right_context_end)+".npy"
            if os.path.exists(self.sliding_clip_path+left_context_name):
                left_context_feat = np.load(self.sliding_clip_path+left_context_name)
                last_left_feat = left_context_feat
            else:
                left_context_feat = last_left_feat
            if os.path.exists(self.sliding_clip_path+right_context_name):
                right_context_feat = np.load(self.sliding_clip_path+right_context_name)
                last_right_feat = right_context_feat
            else:
                right_context_feat = last_right_feat
            left_context_feats[k] = left_context_feat
            right_context_feats[k] = right_context_feat

        return np.mean(left_context_feats, axis=0), np.mean(right_context_feats, axis=0)


    def load_movie(self, movie_name):
        movie_clip_sentences=[]
        for k in range(len(self.clip_names)):
            if movie_name in self.clip_names[k]:
                movie_clip_sentences.append((self.clip_names[k], self.sent_vecs[k][:2400], self.sentences[k]))

        movie_clip_imgs=[]
        for k in range(len(self.movie_frames[movie_name])):
           # print str(k)+"/"+str(len(self.movie_frames[movie_name]))            
            if os.path.isfile(self.movie_frames[movie_name][k][1]) and os.path.getsize(self.movie_frames[movie_name][k][1])!=0:
                img=load_image(self.movie_frames[movie_name][k][1])
                movie_clip_imgs.append((self.movie_frames[movie_name][k][0],img))
                    
        return movie_clip_imgs, movie_clip_sentences

    def load_movie_byclip(self,movie_name,sample_num):
        movie_clip_sentences=[]
        movie_clip_featmap=[]
        clip_set=set()
        for k in range(len(self.clip_sentence_pairs)):
            if movie_name in self.clip_sentence_pairs[k][0]:
                movie_clip_sentences.append((self.clip_sentence_pairs[k][0],self.clip_sentence_pairs[k][1][:self.semantic_size]))

                if not self.clip_sentence_pairs[k][0] in clip_set:
                    clip_set.add(self.clip_sentence_pairs[k][0])
                    # print str(k)+"/"+str(len(self.movie_clip_names[movie_name]))
                    visual_feature_path=self.image_dir+self.clip_sentence_pairs[k][0]+".npy"
                    feature_data=np.load(visual_feature_path)
                    movie_clip_featmap.append((self.clip_sentence_pairs[k][0],feature_data))
        return movie_clip_featmap, movie_clip_sentences
    
    def load_movie_slidingclip(self, movie_name, sample_num):
        movie_clip_sentences = []
        movie_clip_featmap = []
        clip_set = set()
        for k in range(len(self.clip_sentence_pairs)):
            if movie_name in self.clip_sentence_pairs[k][0]:
                movie_clip_sentences.append((self.clip_sentence_pairs[k][0], self.clip_sentence_pairs[k][1][:self.semantic_size]))
        for k in range(len(self.sliding_clip_names)):
            if movie_name in self.sliding_clip_names[k]:
                # print str(k)+"/"+str(len(self.movie_clip_names[movie_name]))
                visual_feature_path = self.sliding_clip_path+self.sliding_clip_names[k]+".npy"
                #context_feat=self.get_context(self.sliding_clip_names[k]+".npy")
                left_context_feat,right_context_feat = self.get_context_window(self.sliding_clip_names[k]+".npy",1)
                feature_data = np.load(visual_feature_path)
                #comb_feat=np.hstack((context_feat,feature_data))
                comb_feat = np.hstack((left_context_feat,feature_data,right_context_feat))
                movie_clip_featmap.append((self.sliding_clip_names[k], comb_feat))
        return movie_clip_featmap, movie_clip_sentences


