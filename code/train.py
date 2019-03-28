import time
import os
import logging

import numpy as np
from tqdm import tqdm

import torch

import utils


from metrics import AverageMeter, Success, Precision, Accuracy_Completeness
from metrics import estimateOverlap, estimateAccuracy

from searchspace import ExhaustiveSearch, GaussianMixtureModel
from searchspace import KalmanFiltering, ParticleFiltering

import torch.nn.functional as F
# TODO: remove torch dependency


def trainer(train_loader,
            val_loader,
            model,
            optimizer,
            scheduler,
            criterion_tracking,
            criterion_completion,
            max_epochs=100,
            model_name=None,
            lambda_completion=0):

    logging.info("start training")

    if model_name is None:
        model_name = "dummy_model"

    best_loss = 9e99

    for epoch in range(max_epochs):
        best_model_path = os.path.join("..", "models", model_name,
                                       "model.pth.tar")

        # train for one epoch
        loss_training = train(
            train_loader,
            model,
            criterion_tracking,
            criterion_completion,
            optimizer,
            epoch + 1,
            lambda_completion=lambda_completion)

        # # evaluate on validation set
        loss_validation = validate(
            val_loader,
            model,
            criterion_tracking,
            criterion_completion,
            epoch + 1,
            lambda_completion=lambda_completion)

        state = {
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_loss': best_loss,
            'optimizer': optimizer.state_dict(),
        }
        os.makedirs(os.path.join("..", "models", model_name), exist_ok=True)
        torch.save(
            state,
            os.path.join("..", "models", model_name,
                         "model_epoch" + str(epoch + 1) + ".pth.tar"))

        # remember best prec@1 and save checkpoint
        is_better = loss_validation < best_loss
        best_loss = min(loss_validation, best_loss)

        # test the model
        if is_better:
            torch.save(state, best_model_path)


        # update the LR scheduler
        if scheduler is not None:
            prevLR = optimizer.param_groups[0]['lr']
            scheduler.step(loss_validation)
            currLR = optimizer.param_groups[0]['lr']
            if (currLR is not prevLR and scheduler.num_bad_epochs == 0):
                logging.info("Plateau Reached!")

            if (prevLR < 2 * scheduler.eps and
                    scheduler.num_bad_epochs >= scheduler.patience):
                logging.info(
                    "Plateau Reached and no more reduction -> Exiting Loop")
                break

    return


# TODO: Combine validation and training?
def train(dataloader,
          model,
          criterion_tracking,
          criterion_completion,
          optimizer,
          epoch,
          lambda_completion=0):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss_tracking = AverageMeter()
    loss_completion = AverageMeter()
    losses = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    with tqdm(enumerate(dataloader), total=len(dataloader), ncols=120)  as t:
        for i, (this_PC, prev_PC, model_PC, target) in t:
            # measure data loading time
            data_time.update(time.time() - end)

            this_PC = this_PC.cuda()
            prev_PC = prev_PC.cuda()
            model_PC = model_PC.cuda()
            target = target.float().cuda(non_blocking=True).view(-1)

            # compute output
            output, prev_PC_AE = model(this_PC, model_PC)

            if lambda_completion < 1:
                loss1 = criterion_tracking(output, target)
            else:
                loss1 = torch.tensor([0]).float().cuda()

            if lambda_completion != 0:
                loss2 = criterion_completion(prev_PC_AE, model_PC)
            else:
                loss2 = torch.tensor([0]).float().cuda()
            loss = loss1 + lambda_completion * loss2

            # measure accuracy and record loss
            loss_tracking.update(loss1.item(), this_PC.size(0))
            loss_completion.update(loss2.item(), this_PC.size(0))
            losses.update(loss.item(), this_PC.size(0))

            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            t.set_description(f'Train {epoch}: '
                              f'Time {batch_time.avg:.3f}s '
                              f'(it:{batch_time.val:.3f}s) '
                              f'Data:{data_time.avg:.3f}s '
                              f'(it:{data_time.val:.3f}s) '
                              f'Loss {losses.avg:.4f} '
                              f'(tr:{loss_tracking.avg:.4f}, '
                              f'comp:{loss_completion.avg:.0f})')
    return losses.avg


def validate(val_loader,
             model,
             criterion_tracking,
             criterion_completion,
             epoch,
             lambda_completion=0):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss_tracking = AverageMeter()
    loss_completion = AverageMeter()
    losses = AverageMeter()

    # switch to evaluate mode
    model.eval()
    with tqdm(enumerate(val_loader), total=len(val_loader), ncols=120) as t:

        with torch.no_grad():
            end = time.time()
            for i, (this_PC, prev_PC, model_PC, target) in t:
                # measure data loading time
                data_time.update(time.time() - end)

                this_PC = this_PC.cuda()
                prev_PC = prev_PC.cuda()
                model_PC = model_PC.cuda()
                target = target.cuda(non_blocking=True).view(-1)

                output, prev_PC_AE = model(this_PC, model_PC)

                if lambda_completion < 1:
                    loss1 = criterion_tracking(output, target)
                else:
                    loss1 = torch.tensor([0]).float().cuda()

                if lambda_completion != 0:
                    loss2 = criterion_completion(prev_PC_AE, model_PC)
                else:
                    loss2 = torch.tensor([0]).float().cuda()
                loss = loss1 + lambda_completion * loss2

                # measure accuracy and record loss
                loss_tracking.update(loss1.item(), this_PC.size(0))
                loss_completion.update(loss2.item(), this_PC.size(0))
                losses.update(loss.item(), this_PC.size(0))

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                t.set_description(f'Valid {epoch}: '
                              f'Time {batch_time.avg:.3f}s '
                              f'(it:{batch_time.val:.3f}s) '
                              f'Data:{data_time.avg:.3f}s '
                              f'(it:{data_time.val:.3f}s) '
                              f'Loss {losses.avg:.4f} '
                              f'(tr:{loss_tracking.avg:.4f}, '
                              f'comp:{loss_completion.avg:.0f})')

    return losses.avg


def test(loader,
         model,
         model_name="dummy_model",
         epoch=-1,
         shape_aggregation="",
         search_space="",
         number_candidate=125,
         reference_BB="",
         model_fusion="pointcloud",
         max_iter=-1,
         IoU_Space=3,
         DetailedMetrics=False):

    batch_time = AverageMeter()
    data_time = AverageMeter()

    Success_main = Success()
    Precision_main = Precision()
    Accuracy_Completeness_main = Accuracy_Completeness()

    Precision_occluded = [Precision(), Precision()]
    Success_occluded = [Success(), Success()]

    Precision_dynamic = [Precision(), Precision()]
    Success_dynamic = [Success(), Success()]


    # SEARCH SPACE INIT
    if ("Kalman".upper() in search_space.upper()):
        search_space_sampler = KalmanFiltering()
    elif ("Particle".upper() in search_space.upper()):
        search_space_sampler = ParticleFiltering()
    elif ("GMM".upper() in search_space.upper()):
        search_space_sampler = GaussianMixtureModel(n_comp=int(search_space[3:]))
    else:
        search_space_sampler = ExhaustiveSearch()

    # switch to evaluate mode
    model.eval()

    end = time.time()

    dataset = loader.dataset
    with tqdm(enumerate(loader), total=len(loader.dataset.list_of_anno), ncols=220) as t:
        for batch in loader:
            # measure data loading time
            data_time.update((time.time() - end))
            for PCs, BBs, list_of_anno in batch: # tracklet

                search_space_sampler.reset()

                results_BBs = []
                results_scores = []
                results_latents = []

                for i, _ in enumerate(PCs):
                    this_anno = list_of_anno[i]
                    this_BB = BBs[i]
                    this_PC = PCs[i]

                    # IS THE POINT CLOUD OCCLUDED?
                    occluded = this_anno["occluded"]
                    if occluded in [0]:  # FULLY VISIBLE
                        occluded = 0
                    elif occluded in [1, 2]:  # PARTIALLY AND FULLY OCCLUDED
                        occluded = 1
                    else:
                        occluded = -1


                    # INITIAL FRAME
                    if i == 0:
                        box = BBs[i]

                        model_PC = utils.getModel([this_PC], [this_BB],
                                                  offset=dataset.offset_BB,
                                                  scale=dataset.scale_BB)

                        if "latent".upper() in model_fusion.upper():
                            this_latent = model.AE.encode(
                                utils.regularizePC(model_PC, model).cuda())[0]

                        score = 1.0
                        candidate_BBs = []
                        dynamic = -1

                    else:
                        # previous_PC = PCs[i - 1]
                        previous_BB = BBs[i - 1]
                        # previous_anno = list_of_anno[i - 1]

                        # IS THE SAMPLE dynamic?
                        if (np.linalg.norm(this_BB.center - previous_BB.center) > 0.709): # for complete set
                            dynamic = 1
                        else:
                            dynamic = 0

                        # DEFINE REFERENCE BB
                        if ("previous_result".upper() in reference_BB.upper()):
                            ref_BB = results_BBs[-1]
                        elif ("previous_gt".upper() in reference_BB.upper()):
                            ref_BB = previous_BB
                        elif ("current_gt".upper() in reference_BB.upper()):
                            ref_BB = this_BB

                        search_space = search_space_sampler.sample(
                            number_candidate)

                        candidate_BBs = utils.generate_boxes(
                            ref_BB, search_space=search_space)

                        candidate_PCs = [
                            utils.cropAndCenterPC(
                                this_PC,
                                box,
                                offset=dataset.offset_BB,
                                scale=dataset.scale_BB) for box in candidate_BBs
                        ]

                        candidate_PCs_reg = [
                            utils.regularizePC(PC, model)
                            for PC in candidate_PCs
                        ]

                        candidate_PCs_torch = torch.cat(
                            candidate_PCs_reg, dim=0).cuda()


                        # DATA FUSION: PC vs LATENT

                        if "latent".upper() in model_fusion.upper():

                            candidate_PCs_encoded = model.AE.encode(candidate_PCs_torch)

                            model_PC_encoded = torch.stack(results_latents) # stack all latent vectors

                            # AGGREGATION: IO vs ONLY0 vs ONLYI vs AVG vs MEDIAN vs MAX
                            if ("firstandprevious".upper() in shape_aggregation.upper()):
                                model_PC_encoded = (model_PC_encoded[0] + model_PC_encoded[i-1] )/ 2
                            elif "first".upper() in shape_aggregation.upper():
                                model_PC_encoded = model_PC_encoded[0]
                            elif "previous".upper() in shape_aggregation.upper():
                                model_PC_encoded = model_PC_encoded[i-1]
                            elif "MEDIAN".upper() in shape_aggregation.upper():
                                model_PC_encoded = torch.median(model_PC_encoded,0)[0]
                            elif ("MAX".upper() in shape_aggregation.upper()):
                                model_PC_encoded = torch.max(model_PC_encoded,0)[0]
                            elif ("AVG".upper() in shape_aggregation.upper()):
                                model_PC_encoded = torch.mean(model_PC_encoded,0)
                            else:
                                model_PC_encoded = torch.mean(model_PC_encoded,0)

                            # repeat model_encoded for size similarity with candidate_PCs
                            repeat_shape = np.ones(len(candidate_PCs_encoded.shape), dtype=np.int32)
                            repeat_shape[0] = len(candidate_PCs_encoded)
                            model_PC_encoded = model_PC_encoded.repeat(tuple(repeat_shape)).cuda()

                            #TODO: remove torch dependency =-> Functional
                            #         Y_AE = model.AE.forward(prev_PC)
                            output = F.cosine_similarity(candidate_PCs_encoded, model_PC_encoded, dim=1)
                            scores = output.detach().cpu().numpy()

                        elif "pointcloud".upper() in model_fusion.upper():

                            # AGGREGATION: IO vs ONLY0 vs ONLYI vs ALL
                            if ("firstandprevious".upper() in shape_aggregation.upper()):
                                model_PC = utils.getModel(
                                    [PCs[0], PCs[i - 1]],
                                    [results_BBs[0], results_BBs[i - 1]],
                                    offset=dataset.offset_BB,
                                    scale=dataset.scale_BB)
                            elif ("first".upper() in shape_aggregation.upper()):
                                model_PC = utils.getModel(
                                    [PCs[0]], [results_BBs[0]],
                                    offset=dataset.offset_BB,
                                    scale=dataset.scale_BB)
                            elif ("previous".upper() in shape_aggregation.
                                  upper()):
                                model_PC = utils.getModel(
                                    [PCs[i - 1]], [results_BBs[i - 1]],
                                    offset=dataset.offset_BB,
                                    scale=dataset.scale_BB)
                            elif ("all".upper() in shape_aggregation.upper()):
                                model_PC = utils.getModel(
                                    PCs[:i],
                                    results_BBs,
                                    offset=dataset.offset_BB,
                                    scale=dataset.scale_BB)
                            else:
                                model_PC = utils.getModel(
                                    PCs[:i],
                                    results_BBs,
                                    offset=dataset.offset_BB,
                                    scale=dataset.scale_BB)

                            repeat_shape = np.ones(
                                len(candidate_PCs_torch.shape), dtype=np.int32)
                            repeat_shape[0] = len(candidate_PCs_torch)
                            model_PC_encoded = utils.regularizePC(
                                model_PC,
                                model).repeat(tuple(repeat_shape)).cuda()

                            output, decoded_PC = model(candidate_PCs_torch,
                                              model_PC_encoded)

                            scores = output.detach().cpu().numpy()

                        elif "space".upper() in model_fusion.upper():

                            scores = np.array([
                                utils.distanceBB_Gaussian(bb, this_BB)
                                for bb in candidate_BBs
                            ])

                        search_space_sampler.addData(data=search_space, score=scores.T)

                        idx = np.argmax(scores)

                        score = scores[idx]
                        box = candidate_BBs[idx]
                        if "latent".upper() in model_fusion.upper():
                            this_latent = candidate_PCs_encoded[idx]

                        if(DetailedMetrics):
                            #  Construct GT model
                            gt_model_PC_start_idx = max(0,i-10)
                            gt_model_PC_end_idx = min(i+10,len(PCs))
                            gt_model_PC = utils.getModel(
                                PCs[gt_model_PC_start_idx:gt_model_PC_end_idx],
                                BBs[gt_model_PC_start_idx:gt_model_PC_end_idx],
                                offset=dataset.offset_BB,
                                scale=dataset.scale_BB)

                            if(gt_model_PC.points.shape[1]>0):
                                gt_model_PC = gt_model_PC.convertToPytorch().float().unsqueeze(2).permute(2,0,1)
                                gt_candidate_PC = utils.regularizePC(
                                    utils.cropAndCenterPC(
                                        this_PC,
                                        this_BB,
                                        offset=dataset.offset_BB,
                                        scale=dataset.scale_BB), model).cuda()
                                decoded_PC = model.AE.decode(
                                    model.AE.encode(
                                        gt_candidate_PC)).detach().cpu()
                                Accuracy_Completeness_main.update(
                                    decoded_PC, gt_model_PC)

                    results_BBs.append(box)
                    results_scores.append(score)

                    if "latent".upper() in model_fusion.upper():
                        results_latents.append(this_latent.detach().cpu())

                    # estimate overlap/accuracy fro current sample
                    this_overlap = estimateOverlap(this_BB, box, dim=IoU_Space)
                    this_accuracy = estimateAccuracy(this_BB, box, dim=IoU_Space)

                    Success_main.add_overlap(this_overlap)
                    Precision_main.add_accuracy(this_accuracy)

                    if (dynamic >= 0):
                        Success_dynamic[dynamic].add_overlap(this_overlap)
                        Precision_dynamic[dynamic].add_accuracy(this_accuracy)

                    if (occluded >= 0):
                        Success_occluded[occluded].add_overlap(this_overlap)
                        Precision_occluded[occluded].add_accuracy(this_accuracy)

                    # measure elapsed time
                    batch_time.update(time.time() - end)
                    end = time.time()

                    t.update(1)

                    if Success_main.count >= max_iter and max_iter >= 0:
                        return Success_main.average, Precision_main.average


                t.set_description(f'Test {epoch}: '
                                  f'Time {batch_time.avg:.3f}s '
                                  f'(it:{batch_time.val:.3f}s) '
                                  f'Data:{data_time.avg:.3f}s '
                                  f'(it:{data_time.val:.3f}s), '
                                  f'Succ/Prec:'
                                  f'{Success_main.average:.1f}/'
                                  f'{Precision_main.average:.1f}')

    if DetailedMetrics:
        logging.info(f"Succ/Prec fully visible({Success_occluded[0].count}):")
        logging.info(f"{Success_occluded[0].average:.1f}/{Precision_occluded[0].average:.1f}")

        logging.info(f"Succ/Prec occluded({Success_occluded[1].count}):")
        logging.info(f"{Success_occluded[1].average:.1f}/{Precision_occluded[1].average:.1f}")

        logging.info(f"Succ/Prec dynamic({Success_dynamic[0].count}):")
        logging.info(f"{Success_dynamic[0].average:.1f}/{Precision_dynamic[0].average:.1f}")

        logging.info(f"Succ/Prec static({Success_dynamic[1].count}):")
        logging.info(f"{Success_dynamic[1].average:.1f}/{Precision_dynamic[1].average:.1f}")

        logging.info(f"Acc/Comp ({Accuracy_Completeness_main.count}):")
        logging.info(f"{Accuracy_Completeness_main.average[0]:.4f}/{Accuracy_Completeness_main.average[1]:.4f}")

    return Success_main.average, Precision_main.average
