class Config:
    def __init__(self):
        self.epoch = 0
        # starting epoch
        self.n_epochs = 200
        # number of epochs of training
        self.batchSize = 1
        # size of the batches
        self.dataroot = '../datasets/horse2zebra/'
        # root directory of the dataset
        self.lr = 0.0002
        # initial learning rate
        self.decay_epoch = 100
        # epoch to start linearly decaying the learning rate to 0
        self.size = 256
        # size of the data crop (squared assumed)
        self.input_nc = 3
        # number of channels of input data
        self.output_nc = 3
        # number of channels of output data
        self.cuda = True
        # use GPU computation
        self.n_cpu = 8
        # number of cpu threads to use during batch generation
        self.pre_train = 0
        self.output_path = '../../output/'
