import numpy as np

class data(object):
    def __init__(self, features,targets, batch_size=50,counter=0):
        features = np.expand_dims(features,axis=2)
        targets = np.expand_dims(targets,axis=2)
        self.sample_num = len(features)
        self.train_features = features[0:int(0.8*len(features))]
        self.train_targets = targets[0:int(0.8*len(targets))]
        self.test_features = features[int(0.8*len(features)):]
        self.test_targets = targets[int(0.8*len(targets)):]
        self.counter = 0
        self.train_sample_num = int(0.8*len(features))
        self.test_sample_num = int(0.2*len(features))
        self.features_last_batch = False
        self.targets_sample_num = targets.shape[0]
        self.targets_width = targets.shape[1]
        self.epoch_counter = 0
        self.input_batch_default = 32
        self.train_counter=0
        self.test_counter=0
    def get_input_sample(self,stage,bs):
        begin=self.counter
        end=self.counter+bs
        if stage == 'train':
            self.train_counter+=1
            print('in training queue')
            if end > self.train_sample_num:
                end = self.train_sample_num
                self.counter = 0
                self.epoch_counter += 1
            else:
                self.counter = end
            samp_features = np.expand_dims(self.train_features[begin:end],axis=2)
            samp_targets = np.expand_dims(self.train_targets[begin:end],axis=2)
            return(samp_features, samp_targets,self.train_counter)
        elif stage == 'test':
            self.test_counter+=1
            print('in test queue, counter, epoch', self.train_counter, self.epoch_counter)
            if end > self.test_sample_num:
                end = self.test_sample_num
                self.counter = 0
                self.epoch_counter += 1
            else:
                self.counter = end
            samp_features = np.expand_dims(self.test_features[begin:end], axis=2)
            samp_targets = np.expand_dims(self.test_targets[begin:end],axis=2)
            return(samp_features, samp_targets,self.test_counter)
