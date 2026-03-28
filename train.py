import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2"
import time, yaml
import torch
import copy
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import WHSDataset_2D_scale_partSeries
import numpy as np
import pandas as pd
import torch.nn.functional as F
from utils import logger
from loss import WCEDCELoss
from models.reslstmunet import ResLSTMUNet

if __name__ == '__main__':
    ############################
    # Parameters
    ############################
    CFG_FILE = "train_info.yaml"
    with open(CFG_FILE, "r") as f:
        cfg = yaml.safe_load(f)

    NUM_EPOCHS = 60
    BATCHSIZE = 12 # 8, 12
    NUM_WORKERS = 8
    LEARNINGRATE = 0.0001  # 0.0001

    model_freq = 3 #2

    NUM_CLASS = 8

    PRETRAINED = True
    DEEPSUPERVISION =True
    AUGMENTATION = True
    WCEDCELOSS = True
    MULTISCALEATT = True
    ############################
    # load data
    ###########################
    image_paths = list()
    image_paths.append(cfg["WHS_datasets"]["datasets_input_path"][0])
    image_paths.append(cfg["WHS_datasets"]["datasets_input_path"][2])

    test_paths = list()
    test_paths.append(cfg["WHS_datasets"]["datasets_input_path"][1])

    crop_d = 18 # 设置序列长度

    train_set = WHSDataset_2D_scale_partSeries(image_multidir=image_paths, crop_d = crop_d)
    val_set = WHSDataset_2D_scale_partSeries(image_multidir=test_paths, crop_d = crop_d)

    train_loader = DataLoader(dataset=train_set, num_workers=NUM_WORKERS, batch_size=BATCHSIZE, shuffle=True, pin_memory=True)
    val_loader = DataLoader(dataset=val_set, num_workers=NUM_WORKERS, batch_size=BATCHSIZE, shuffle=False, pin_memory=True)
    ############################
    # load the net
    ###########################
    architecture = cfg["model_2D"][0]

    model_net = ResLSTMUNet(in_channels=1, out_channels=NUM_CLASS, pretrained=PRETRAINED, deep_sup=DEEPSUPERVISION, multiscale_att=MULTISCALEATT)

    model_net = torch.nn.DataParallel(model_net).cuda()

    INIT_MODEL_PATH = os.path.join(cfg["WHS_datasets"]["results_output"]["model_state_dict"], architecture)

    print('#parameters:', sum(param.numel() for param in model_net.parameters()))
    ############################
    # loss and optimization
    ###########################
    if WCEDCELOSS:
        criterion = WCEDCELoss(intra_weights=torch.tensor([1., 3., 3., 3., 3., 3., 3., 3.]).cuda(), inter_weights=0.5)
    else:
        criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor([1., 3., 3., 3., 3., 3., 3., 3.]).cuda())
    # construct an optimizer
    optimizer = optim.Adam(model_net.parameters(), lr=LEARNINGRATE, betas=(0.9, 0.999))
    ############################
    # Train the net
    ############################
    results = {'loss': [], 'dice': [], 'iou': [], 'val_loss': [], 'val_dice': [], 'val_iou': []}
    # for saving best model
    best_model_wts = copy.deepcopy(model_net)
    best_dice = 0.0
    best_loss = 2. #2.
    best_epoch = 0

    since = time.time()
    model_dict_temp_save_path = os.path.join(cfg["WHS_datasets"]["results_output"]["model_state_dict"], architecture)
    if not os.path.exists(model_dict_temp_save_path):
        os.makedirs(model_dict_temp_save_path)

    statistics_path = os.path.join(cfg["WHS_datasets"]["results_output"]["statistics"], architecture)
    if not os.path.exists(statistics_path):
        os.makedirs(statistics_path)
    logger = logger(statistics_path + '/train_' + architecture + '.log')
    logger.info('start training!')

    for epoch in range(1, NUM_EPOCHS + 1):
        # adjust_lr(optimizer=optimizer, init_lr=0.0001, epoch=epoch, decay_rate=0.1, decay_epoch=30)
        epochresults = {'loss': [], 'dice': [], 'iou': [], 'val_loss': [], 'val_dice': [], 'val_iou': []}
        model_net.train()
        for iteration, (image_serial, label_serial) in enumerate(train_loader):
            image_serial = [item.cuda() for item in image_serial]
            label_serial = [item.cuda() for item in label_serial]

            optimizer.zero_grad()

            if DEEPSUPERVISION:
                pred_serial, pred1_serial, pred2_serial, pred3_serial, pred4_serial = model_net(image_serial)

                temporal = len(pred_serial)
                loss = 0
                for t in range(temporal):
                    pred = pred_serial[t]
                    pred1 = pred1_serial[t]
                    pred2 = pred2_serial[t]
                    pred3 = pred3_serial[t]
                    pred4 = pred4_serial[t]

                    label = label_serial[t]

                    loss0 = criterion(pred, label.squeeze(1).long())
                    loss1 = criterion(pred1, F.interpolate(label, scale_factor=1. / 2., mode='bilinear',align_corners=False).squeeze(1).long())
                    loss2 = criterion(pred2, F.interpolate(label, scale_factor=1. / 4., mode='bilinear',align_corners=False).squeeze(1).long())
                    loss3 = criterion(pred3, F.interpolate(label, scale_factor=1. / 8., mode='bilinear',align_corners=False).squeeze(1).long())
                    loss4 = criterion(pred4, F.interpolate(label, scale_factor=1. / 16., mode='bilinear',align_corners=False).squeeze(1).long())
                    loss += 0.4 * loss0 + 0.3 * loss1 + 0.2 * loss2 + 0.05 * loss3 + 0.05 * loss4

                loss /= temporal
            else:
                pred_serial = model_net(image_serial)
                temporal = pred_serial.shape[1]
                loss = 0
                for t in range(temporal):
                    pred = pred_serial[t]
                    label = label_serial[t]

                    loss += criterion(pred, label.squeeze(1).long())
                loss /= temporal

            loss.backward()
            optimizer.step()
            if iteration % int(len(train_loader)/5) == 0:
                logger.info("Train: Epoch/Epoches {}/{}\t"
                            "iteration/iterations {}/{}\t"
                            "loss {:.3f}".format(epoch, NUM_EPOCHS, iteration, len(train_loader), loss.item()))
            epochresults['loss'].append(loss.item())
        results['loss'].append(np.mean(epochresults['loss']))
        ############################
        # validate the net
        ############################
        model_net.eval()
        with torch.no_grad():
            for val_iteration, (val_image_serial, val_label_serial) in enumerate(val_loader):

                val_image_serial = [item.cuda() for item in val_image_serial]
                val_label_serial = [item.cuda() for item in val_label_serial]

                if DEEPSUPERVISION:
                    val_pred_serial, val_pred1_serial, val_pred2_serial, val_pred3_serial, val_pred4_serial = model_net(
                        val_image_serial)

                    val_temporal = len(val_pred_serial)
                    val_loss = 0
                    for t in range(val_temporal):
                        val_pred = val_pred_serial[t]
                        val_pred1 = val_pred1_serial[t]
                        val_pred2 = val_pred2_serial[t]
                        val_pred3 = val_pred3_serial[t]
                        val_pred4 = val_pred4_serial[t]

                        val_label = val_label_serial[t]

                        val_loss0 = criterion(val_pred, val_label.squeeze(1).long())
                        val_loss1 = criterion(val_pred1, F.interpolate(val_label, scale_factor=1. / 2., mode='bilinear',
                                                                       align_corners=False).squeeze(1).long())
                        val_loss2 = criterion(val_pred2, F.interpolate(val_label, scale_factor=1. / 4., mode='bilinear',
                                                                       align_corners=False).squeeze(1).long())
                        val_loss3 = criterion(val_pred3, F.interpolate(val_label, scale_factor=1. / 8., mode='bilinear',
                                                                       align_corners=False).squeeze(1).long())
                        val_loss4 = criterion(val_pred4,
                                              F.interpolate(val_label, scale_factor=1. / 16., mode='bilinear',
                                                            align_corners=False).squeeze(1).long())
                        val_loss += 0.4 * val_loss0 + 0.3 * val_loss1 + 0.2 * val_loss2 + 0.05 * val_loss3 + 0.05 * val_loss4
                    val_loss /= val_temporal
                else:
                    val_pred_serial = model_net(val_image_serial)
                    val_temporal = val_pred_serial.shape[1]
                    val_loss = 0
                    for t in range(val_temporal):
                        val_pred = val_pred_serial[t]
                        val_label = val_label_serial[t]

                        val_loss += criterion(val_pred, val_label.squeeze(1).long())
                    val_loss /= val_temporal

                if val_iteration % int(len(val_loader)/5) == 0:
                    logger.info("Val: Epoch/Epoches {}/{}\t"
                                "iteration/iterations {}/{}\t"
                                "val loss {:.3f}".format(epoch, NUM_EPOCHS, val_iteration, len(val_loader), val_loss.item()))
                epochresults['val_loss'].append(val_loss.item())
            results['val_loss'].append(np.mean(epochresults['val_loss']))

            logger.info("Average: Epoch/Epoches {}/{}\t"
                        "train epoch loss {:.3f}\t"
                        "val epoch loss {:.3f}\n".format(epoch, NUM_EPOCHS, np.mean(epochresults['loss']),
                                                         np.mean(epochresults['val_loss'])))
        # saving the best model parameters
        if np.mean(epochresults['val_loss']) < best_loss:
            best_loss = np.mean(epochresults['val_loss'])
            best_model_wts = copy.deepcopy(model_net)
            best_epoch = epoch

        if epoch % model_freq == 0 or epoch == NUM_EPOCHS:
            torch.save(model_net.state_dict(), model_dict_temp_save_path + '/' + architecture + '_' + 'epoch_%d.pth' % epoch)

    torch.save(best_model_wts.state_dict(), statistics_path + '/' + architecture + '_best_epoch_%d.pth' % best_epoch)
    logger.info(statistics_path + '/train_' + architecture + '_best_epoch_%d.pth' % best_epoch)

    time_elapsed = time.time() - since
    logger.info('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    logger.info('finish training!')

    ############################
    # save the results
    ############################
    data_frame = pd.DataFrame(
        data={'loss': results['loss'],
              'val_loss': results['val_loss']},
        index=range(1, NUM_EPOCHS + 1))
    data_frame.to_csv('statistics/' + architecture + '/train_results.csv', index_label='Epoch')
