import time
import cv2
import torch
import argparse
import demo_utils
import noise_image
import numpy as np
import pytorch_lightning as pl
from thop import profile
from config.defaultmf import get_cfg_defaults
from model.lightning_loftr import PL_LoFTR
from sklearn.metrics import mean_squared_error


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # parser.add_argument(
    #     'data_cfg_path', type=str, help='data config path')
    parser.add_argument(
        '--ckpt_path', type=str, default="./weights/outdoor-large-LA.ckpt", help='path to the checkpoint')
    parser.add_argument(
        '--dump_dir', type=str, default=None, help="if set, the matching results will be dump to dump_dir")
    parser.add_argument(
        '--profiler_name', type=str, default='inference', help='options: [inference, pytorch], or leave it unset')
    parser.add_argument(
        '--batch_size', type=int, default=1, help='batch_size per gpu')
    parser.add_argument(
        '--num_workers', type=int, default=2)
    parser.add_argument(
        '--thr', type=float, default=None, help='modify the coarse-level matching threshold.')

    parser = pl.Trainer.add_argparse_args(parser)
    return parser.parse_args()


if __name__ == '__main__':
    # parse arguments
    args = parse_args()
    # init default-cfg and merge it with the main- and data-cfg
    config = get_cfg_defaults()
    # config.merge_from_file(args.data_cfg_path)
    pl.seed_everything(config.TRAINER.SEED)  # reproducibility

    # tune when testing
    if args.thr is not None:
        config.LOFTR.MATCH_COARSE.THR = args.thr

    # lightning module
    model = PL_LoFTR(config, pretrained_ckpt=args.ckpt_path, dump_dir=args.dump_dir)
    matcher = model.matcher
    torch.cuda.empty_cache()
    matcher.cuda(), matcher.eval()
    # lightning data
    # data_module = MultiSceneDataModule(args, config)

    img0_path = "./assets/1DSMsets/pair7-2.png"
    img1_path = "./assets/1DSMsets/pair7-1.png"

    # -----------------------origin image ------------------------
    output_path = "./output/1DSMsets/pair1r.jpg"
    img0 = cv2.imread(img0_path)
    img0 = demo_utils.resize(img0, 512)

    # --------------------Additive noise image ------------------
    # output_path = "./output/1DSMsets/pair1+snr0.jpg"
    # img0 = noise_image.Additive_noise(img0_path, SNR=0)

    # --------------------stripe noise image --------------------
    # output_path = "output/1DSMsets/pair1+0p101S.jpg"
    # img0 = noise_image.stripe_noise(img0_path, 0.1)

    img1 = cv2.imread(img1_path)
    img1 = demo_utils.resize(img1, 512)

    img0_g = cv2.imread(img0_path, 0)
    img1_g = cv2.imread(img1_path, 0)

    img0_g, img1_g = demo_utils.resize(img0_g, 512), demo_utils.resize(img1_g, 512)
    batch = {'image0': torch.from_numpy(img0_g / 255.)[None, None].cuda().float(),
             'image1': torch.from_numpy(img1_g / 255.)[None, None].cuda().float()}

    # with torch.no_grad(): 只是想要网络结果的话就不需要反向传播，如果想通过网络输出的结果去进一步优化网络的话就需要反向传播了。
    # Inference with LoFTR and get prediction

    tic = time.time()
    with torch.no_grad():  # 不需要进行网络参数的更新就不用反向传播

        matcher(batch)

        mkpts0 = batch['mkpts0_f'].cpu().numpy()
        mkpts1 = batch['mkpts1_f'].cpu().numpy()

    toc = time.time()
    tt1 = toc - tic
    # --------------------------RANSAC Outlier Removal----------------------------------
    # F_hat, mask_F = cv2.findFundamentalMat(mkpts0, mkpts1, method=cv2.USAC_FAST,
    #                                        ransacReprojThreshold=1, confidence=0.999)
    # F_hat, mask_F = cv2.findFundamentalMat(mkpts0, mkpts1, method=cv2.USAC_ACCURATE,
    #                                        ransacReprojThreshold=1, confidence=0.999)
    F_hat, mask_F = cv2.findFundamentalMat(mkpts0, mkpts1, method=cv2.USAC_MAGSAC,
                                           ransacReprojThreshold=1, confidence=0.999)



    if mask_F is not None:
        mask_F = mask_F[:, 0].astype(bool)
    else:
        mask_F = np.zeros_like(mkpts0[:, 0]).astype(bool)

    # visualize match
    # display = demo_utils.draw_match(img0, img1, mkpts0, mkpts1)
    display = demo_utils.draw_match(img0, img1, mkpts0[mask_F], mkpts1[mask_F])

    putative_num = len(mkpts0)
    correct_num = len(mkpts0[mask_F])
    inliner_ratio = correct_num / putative_num
    # -------------------------------RMSE计算---------------------------------

    text1 = "putative_num:{}".format(putative_num)
    text2 = 'correct_num:{}'.format(correct_num)
    text3 = 'inliner ratio:%.3f' % inliner_ratio
    text4 = 'run time: %.3fs' % tt1

    print('putative_num:{}'.format(putative_num), '\ncorrect_num:{}'.format(correct_num),
          '\ninliner ratio:%.3f' % inliner_ratio, '\nrun time: %.3fs' % tt1)

    cv2.putText(display, str(text1), (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    cv2.putText(display, str(text2), (5, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    cv2.putText(display, str(text3), (5, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    cv2.putText(display, str(text4), (5, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    cv2.imwrite(output_path, display)

    flops, params = profile(matcher, inputs=(batch,))
    print("Params：", "%.2f" % (params / (1000 ** 2)), "M")
    print("GFLOPS：", "%.2f" % (flops / (1000 ** 3)))
