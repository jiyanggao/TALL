import tensorflow as tf
import numpy as np
import ctrl_model
from six.moves import xrange
import time
from sklearn.metrics import average_precision_score
import pickle
import vs_multilayer
import operator

def dense_to_one_hot(labels_dense, num_classes):
    """Convert class labels from scalars to one-hot vectors."""
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset+labels_dense.ravel()] = 1
    return labels_one_hot

def compute_ap(class_score_matrix, labels):
    num_classes=class_score_matrix.shape[1]
    one_hot_labels=dense_to_one_hot(labels, num_classes)
    predictions=np.array(class_score_matrix>0, dtype="int32")
    average_precision=[]
    for i in range(num_classes):
        ps=average_precision_score(one_hot_labels[:, i], class_score_matrix[:, i])
       # if not np.isnan(ps):
        average_precision.append(ps)
    return np.array(average_precision)

def calculate_IoU(i0,i1):
    union = (min(i0[0], i1[0]), max(i0[1], i1[1]))
    inter = (max(i0[0], i1[0]), min(i0[1], i1[1]))
    iou = 1.0*(inter[1]-inter[0])/(union[1]-union[0])
    return iou

def nms_temporal(x1,x2,s, overlap):
    pick = []
    assert len(x1)==len(s)
    assert len(x2)==len(s)
    if len(x1)==0:
        return pick

    #x1 = [b[0] for b in boxes]
    #x2 = [b[1] for b in boxes]
    #s = [b[-1] for b in boxes]
    union = map(operator.sub, x2, x1) # union = x2-x1
    I = [i[0] for i in sorted(enumerate(s), key=lambda x:x[1])] # sort and get index

    while len(I)>0:
        i = I[-1]
        pick.append(i)

        xx1 = [max(x1[i],x1[j]) for j in I[:-1]]
        xx2 = [min(x2[i],x2[j]) for j in I[:-1]]
        inter = [max(0.0, k2-k1) for k1, k2 in zip(xx1, xx2)]
        o = [inter[u]/(union[i] + union[I[u]] - inter[u]) for u in range(len(I)-1)]
        I_new = []
        for j in range(len(o)):
            if o[j] <=overlap:
                I_new.append(I[j])
        I = I_new
    return pick

'''
compute recall at certain IoU
'''
def compute_IoU_recall_top_n_forreg(top_n, iou_thresh, sentence_image_mat, sentence_image_reg_mat, sclips, iclips):
    correct_num = 0.0
    for k in range(sentence_image_mat.shape[0]):
        gt = sclips[k]
        gt_start = float(gt.split("_")[1])
        gt_end = float(gt.split("_")[2])
        #print gt +" "+str(gt_start)+" "+str(gt_end)
        sim_v = [v for v in sentence_image_mat[k]]
        starts = [s for s in sentence_image_reg_mat[k,:,0]]
        ends = [e for e in sentence_image_reg_mat[k,:,1]]
        picks = nms_temporal(starts,ends, sim_v, iou_thresh-0.05)
        #sim_argsort=np.argsort(sim_v)[::-1][0:top_n]
        if top_n<len(picks): picks=picks[0:top_n]
        for index in picks:
            pred_start = sentence_image_reg_mat[k, index, 0]
            pred_end = sentence_image_reg_mat[k, index, 1]
            iou = calculate_IoU((gt_start, gt_end),(pred_start, pred_end))
            if iou>=iou_thresh:
                correct_num+=1
                break
    return correct_num

'''
evaluate the model
'''
def do_eval_slidingclips(sess, vs_eval_op, model, movie_length_info, iter_step, test_result_output):
    IoU_thresh = [0.1, 0.2, 0.3, 0.4, 0.5]
    all_correct_num_10 = [0.0]*5
    all_correct_num_5 = [0.0]*5
    all_correct_num_1 = [0.0]*5
    all_retrievd = 0.0
    for movie_name in model.test_set.movie_names:
        movie_length=movie_length_info[movie_name.split(".")[0]]
        print "Test movie: "+movie_name+"....loading movie data"
        movie_clip_featmaps, movie_clip_sentences=model.test_set.load_movie_slidingclip(movie_name, 16)
        print "sentences: "+ str(len(movie_clip_sentences))
        print "clips: "+ str(len(movie_clip_featmaps))
        sentence_image_mat=np.zeros([len(movie_clip_sentences), len(movie_clip_featmaps)])
        sentence_image_reg_mat=np.zeros([len(movie_clip_sentences), len(movie_clip_featmaps), 2])
        for k in range(len(movie_clip_sentences)):
            #sentence_clip_name=movie_clip_sentences[k][0]
            #start=float(sentence_clip_name.split("_")[1])
            #end=float(sentence_clip_name.split("_")[2].split("_")[0])
            
            sent_vec=movie_clip_sentences[k][1]
            sent_vec=np.reshape(sent_vec,[1,sent_vec.shape[0]])
            for t in range(len(movie_clip_featmaps)):
                featmap = movie_clip_featmaps[t][1]
                visual_clip_name = movie_clip_featmaps[t][0]
                start = float(visual_clip_name.split("_")[1])
                end = float(visual_clip_name.split("_")[2].split("_")[0])
                featmap = np.reshape(featmap, [1, featmap.shape[0]])
                feed_dict = {
                model.visual_featmap_ph_test: featmap,
                model.sentence_ph_test:sent_vec
                }
                outputs = sess.run(vs_eval_op,feed_dict=feed_dict)
                sentence_image_mat[k,t] = outputs[0]
                reg_clip_length = (end-start)*(10**outputs[2])
                reg_mid_point = (start+end)/2.0+movie_length*outputs[1]
                reg_end = end+outputs[2]
                reg_start = start+outputs[1]
                
                sentence_image_reg_mat[k,t,0] = reg_start
                sentence_image_reg_mat[k,t,1] = reg_end
        
        iclips = [b[0] for b in movie_clip_featmaps]
        sclips = [b[0] for b in movie_clip_sentences]
        
        # calculate Recall@m, IoU=n
        for k in range(len(IoU_thresh)):
            IoU=IoU_thresh[k]
            correct_num_10 = compute_IoU_recall_top_n_forreg(10, IoU, sentence_image_mat, sentence_image_reg_mat, sclips, iclips)
            correct_num_5 = compute_IoU_recall_top_n_forreg(5, IoU, sentence_image_mat, sentence_image_reg_mat, sclips, iclips)
            correct_num_1 = compute_IoU_recall_top_n_forreg(1, IoU, sentence_image_mat, sentence_image_reg_mat, sclips, iclips)
            print movie_name+" IoU="+str(IoU)+", R@10: "+str(correct_num_10/len(sclips))+"; IoU="+str(IoU)+", R@5: "+str(correct_num_5/len(sclips))+"; IoU="+str(IoU)+", R@1: "+str(correct_num_1/len(sclips))
            all_correct_num_10[k]+=correct_num_10
            all_correct_num_5[k]+=correct_num_5
            all_correct_num_1[k]+=correct_num_1
        all_retrievd+=len(sclips)
    for k in range(len(IoU_thresh)):
        print " IoU="+str(IoU_thresh[k])+", R@10: "+str(all_correct_num_10[k]/all_retrievd)+"; IoU="+str(IoU_thresh[k])+", R@5: "+str(all_correct_num_5[k]/all_retrievd)+"; IoU="+str(IoU_thresh[k])+", R@1: "+str(all_correct_num_1[k]/all_retrievd)
        test_result_output.write("Step "+str(iter_step)+": IoU="+str(IoU_thresh[k])+", R@10: "+str(all_correct_num_10[k]/all_retrievd)+"; IoU="+str(IoU_thresh[k])+", R@5: "+str(all_correct_num_5[k]/all_retrievd)+"; IoU="+str(IoU_thresh[k])+", R@1: "+str(all_correct_num_1[k]/all_retrievd)+"\n")

def run_training():
    initial_steps = 0
    max_steps = 20000
    batch_size = 56
    train_csv_path = "./exp_data/TACoS/train_clip-sentvec.pkl"
    test_csv_path = "./exp_data/TACoS/test_clip-sentvec.pkl"
    test_feature_dir="../TACOS/Interval128_256_overlap0.8_c3d_fc6/"
    train_feature_dir = "../TACOS/Interval64_128_256_512_overlap0.8_c3d_fc6/"
    
    model = ctrl_model.CTRL_Model(batch_size, train_csv_path, test_csv_path, test_feature_dir, train_feature_dir)
    test_result_output=open("ctrl_test_results.txt", "w")
    with tf.Graph().as_default():
		
        loss_align_reg, vs_train_op, vs_eval_op, offset_pred, loss_reg = model.construct_model()
        # Create a session for running Ops on the Graph.
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction = 0.2)
        sess = tf.Session(config = tf.ConfigProto(gpu_options = gpu_options))
        # Run the Op to initialize the variables.
        init = tf.initialize_all_variables()
        sess.run(init)
        for step in xrange(max_steps):
            start_time = time.time()
            feed_dict = model.fill_feed_dict_train_reg()
            _, loss_value, offset_pred_v, loss_reg_v = sess.run([vs_train_op, loss_align_reg, offset_pred, loss_reg], feed_dict=feed_dict)
            duration = time.time() - start_time

            if step % 5 == 0:
                # Print status to stdout.
                print('Step %d: loss = %.3f (%.3f sec)' % (step, loss_value, duration))

            if (step+1) % 2000 == 0:
                print "Start to test:-----------------\n"
                movie_length_info=pickle.load(open("./video_allframes_info.pkl"))
                do_eval_slidingclips(sess, vs_eval_op, model, movie_length_info, step+1, test_result_output)

def main(_):
    run_training()


if __name__ == '__main__':
    tf.app.run()
        	



