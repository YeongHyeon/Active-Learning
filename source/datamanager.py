import numpy as np
import source.utils as utils
import tensorflow as tf
from sklearn.utils import shuffle

class DataSet(object):

    def __init__(self, dataname):

        print("\nInitializing Dataset...")
        self.dataset = 10
        self.dataname = dataname
        self.__reset_index()
        self.__preparing()
        self.__reset_index()

    def __reset_index(self):

        self.idx_tr, self.idx_val, self.idx_te, self.idx_unlabel = 0, 0, 0, 0

    def __preparing(self):
        
        if(self.dataname.lower() == 'mnist'): 
            (x_tr, y_tr), (x_te, y_te) = tf.keras.datasets.mnist.load_data()
            x_tr = np.expand_dims(x_tr, axis=-1)
            x_te = np.expand_dims(x_te, axis=-1)
        else:
            (x_tr, y_tr), (x_te, y_te) = tf.keras.datasets.cifar10.load_data()
            y_tr = y_tr[:, 0]
            y_te = y_te[:, 0]

        self.x_tr, self.y_tr = x_tr, y_tr
        self.x_tr, self.y_tr = shuffle(self.x_tr, self.y_tr)
        num_tr = int(self.x_tr.shape[0] * 0.1)
        self.x_unlabel, self.y_unlabel = self.x_tr[num_tr:].copy(), self.y_tr[num_tr:].copy()
        self.x_tr, self.y_tr = self.x_tr[:num_tr], self.y_tr[:num_tr]

        num_val = int(self.x_tr.shape[0] * 0.1)
        self.x_val, self.y_val = self.x_tr[:num_val], self.y_tr[:num_val]
        self.x_tr, self.y_tr = self.x_tr[num_val:], self.y_tr[num_val:]
        self.x_te, self.y_te = x_te, y_te

        ftxt = open("training_summary.txt", "w")
        for i in range(10):
            text = "Class-%d: %5d samples (%d unlabeled samples)" %(i, np.sum(self.y_tr == i), np.sum(self.y_unlabel == i))
            print(text)
            ftxt.write("%s\n" %(text))
        ftxt.close()

        self.num_tr, self.num_val, self.num_te = self.x_tr.shape[0], self.x_val.shape[0], self.x_te.shape[0]
        self.num_unlabel = self.x_unlabel.shape[0]
        dict_tmp = self.next_batch(ttvu=0)
        x_tmp = dict_tmp['x']
        self.dim_h, self.dim_w, self.dim_c = x_tmp.shape[1], x_tmp.shape[2], x_tmp.shape[3]
        self.num_class = 10

        self.num_trained = self.num_tr

    def set_by_oracle(self, loss, K=1000):

        loss = np.argsort(loss)
        self.x_tr = np.concatenate((self.x_tr, self.x_unlabel[loss[:K]]), axis=0)
        self.y_tr = np.concatenate((self.y_tr, self.y_unlabel[loss[:K]]), axis=0)
        self.x_unlabel, self.y_unlabel = self.x_unlabel[loss[K:]], self.y_unlabel[loss[K:]]
        self.num_tr, self.num_unlabel = self.x_tr.shape[0], self.x_unlabel.shape[0]

        self.num_trained += self.num_tr
        self.__reset_index()

    def next_batch(self, batch_size=1, ttvu=0):

        if(ttvu == 0):
            idx_d, num_d, data, label = self.idx_tr, self.num_tr, self.x_tr, self.y_tr
        elif(ttvu == 1):
            idx_d, num_d, data, label = self.idx_te, self.num_te, self.x_te, self.y_te
        elif(ttvu == 2):
            idx_d, num_d, data, label = self.idx_val, self.num_val, self.x_val, self.y_val
        elif(ttvu == 3):
            idx_d, num_d, data, label = self.idx_unlabel, self.num_unlabel, self.x_unlabel, self.y_unlabel

        batch_x, batch_y, terminate = [], [], False
        while(True):

            try:
                tmp_x = utils.min_max_norm(data[idx_d])
                tmp_y = np.eye(10)[label[idx_d]]
            except:
                idx_d = 0
                if(ttvu == 0):
                    self.x_tr, self.y_tr = shuffle(self.x_tr, self.y_tr)
                terminate = True
                break

            batch_x.append(tmp_x)
            batch_y.append(tmp_y)
            idx_d += 1

            if(len(batch_x) >= batch_size): break

        batch_x = np.asarray(batch_x)
        batch_y = np.asarray(batch_y)

        if(ttvu == 0): self.idx_tr = idx_d
        elif(ttvu == 1): self.idx_te = idx_d
        elif(ttvu == 2): self.idx_val = idx_d
        elif(ttvu == 3): self.idx_unlabel = idx_d

        return {'x':batch_x.astype(np.float32), 'y':batch_y.astype(np.float32), 'terminate':terminate}