class Path(object):
    @staticmethod
    def db_root_dir(dataset):
        if dataset == 'pascal':
            # return '/path/to/datasets/VOCdevkit/VOC2012/'  # folder that contains VOCdevkit/.
            return 'G:\\LoveDA\\'
            # return 'H:/datasets/VOCdevkit/VOC2012/'
        elif dataset == 'sbd':
            return '/path/to/datasets/benchmark_RELEASE/'  # folder that contains dataset/.
        elif dataset == 'cityscapes':
            return '/path/to/datasets/cityscapes/'     # foler that contains leftImg8bit/
        elif dataset == 'coco':
            return '/path/to/datasets/coco/'
        elif dataset == 'supervisely':
            return 'E:/CodeDownload/dataset/supervisely/'
        else:
            print('Dataset {} not available.'.format(dataset))
            raise NotImplementedError
