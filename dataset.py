import torch
from torch.utils.data.dataset import Dataset
import numpy as np
import os
import torch.nn.functional as F
from tqdm import tqdm

def norm_img(image, MIN_BOUND=0., MAX_BOUND=2000.):
    image = (image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
    image[image > 1] = 1.
    image[image < 0] = 0.
    return image

readvdnames = lambda x: open(x).read().rstrip().split('\n')


class WHSDataset_2D_scale_partSeries(Dataset):
    def __init__(self, image_multidir, crop_d=32, stride=5):
        super(WHSDataset_2D_scale_partSeries, self).__init__()

        self.image_paths = list()

        for image_dir in image_multidir:
            for filename3D in os.listdir(image_dir):
                filename3D = os.path.join(image_dir,filename3D)

                image_multipaths = [os.path.join(filename3D, x)
                                for x in os.listdir(filename3D)
                                if x.endswith('image', 0, 5)]
                image_multipaths.sort(key=lambda x: int(x[-8:-4]))

                self.get_series(image_multipaths, crop_d, stride)

        pass

    def get_series(self, patientSeries, crop_d, stride):

        seriesLength = len(patientSeries)

        if seriesLength < crop_d:
            image_paths = list()
            image_paths.extend(patientSeries)
            for i in range(crop_d-seriesLength):
                image_paths.append(patientSeries[-1])

            self.image_paths.append(image_paths)
            return None

        start_slice = 0
        end_slice = start_slice + crop_d
        while end_slice <= seriesLength:
            self.image_paths.append(patientSeries[start_slice:end_slice])
            start_slice += stride
            end_slice = start_slice + crop_d

        if end_slice > seriesLength:
            self.image_paths.append(patientSeries[-crop_d:])

        pass

    def __getitem__(self, index):
        image_serial = list()
        label_serial = list()

        for image_path in self.image_paths[index]:
            filename = os.path.basename(image_path)
            filename = filename.replace('image', 'label')
            label_path = os.path.join(os.path.dirname(image_path),filename)

            image = np.load(image_path)
            label = np.load(label_path)

            image = torch.from_numpy(image.astype('float32')).unsqueeze(0).unsqueeze(0)
            image = F.interpolate(image,[224,224], mode="bilinear",align_corners=False)
            image = image.squeeze(0)

            label = torch.from_numpy(label.astype('float32')).unsqueeze(0).unsqueeze(0)
            label = F.interpolate(label, [224,224])
            label = label.squeeze(0)

            image_serial.append(image)
            label_serial.append(label)

        return image_serial, label_serial

    def __len__(self):
        return len(self.image_paths)


