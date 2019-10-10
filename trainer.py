import torch
import torch.nn.functional as F

from torch.autograd import Variable
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

import os
import time
import shutil
import pickle

from tqdm import tqdm
from utils import AverageMeter
from model import RecurrentAttention

import wandb
wandb.init("RVA")


class Trainer(object):
    """
    Trainer encapsulates all the logic necessary for
    training the Recurrent Attention Model.

    All hyperparameters are provided by the user in the
    config file.
    """
    def __init__(self, config, data_loader):
        """
        Construct a new Trainer instance.

        Args
        ----
        - config: object containing command line arguments.
        - data_loader: data iterator
        """
        self.config = config

        # glimpse network params
        self.patch_size = config.patch_size
        self.glimpse_scale = config.glimpse_scale
        self.num_patches = config.num_patches
        self.loc_hidden = config.loc_hidden
        self.glimpse_hidden = config.glimpse_hidden

        # core network params
        self.num_glimpses = config.num_glimpses
        self.hidden_size = config.hidden_size

        # reinforce params
        self.std = config.std
        self.M = config.M

        # data params
        if config.is_train:
            self.train_loader = data_loader[0]
            self.valid_loader = data_loader[1]
            self.num_train = len(self.train_loader.sampler.indices)
            self.num_valid = len(self.valid_loader.sampler.indices)
        else:
            self.test_loader = data_loader
            self.num_test = len(self.test_loader.dataset)
        self.num_classes = 2
        self.num_channels = 1

        # training params
        self.epochs = config.epochs
        self.start_epoch = 0
        self.momentum = config.momentum
        self.lr = config.init_lr
        self.loss_balance = config.loss_balance

        # misc params
        self.use_gpu = config.use_gpu
        self.best = config.best
        self.ckpt_dir = config.ckpt_dir
        self.logs_dir = config.logs_dir
        self.best_valid_acc = 0.
        self.counter = 0
        self.lr_patience = config.lr_patience
        self.train_patience = config.train_patience
        self.use_tensorboard = config.use_tensorboard
        self.resume = config.resume
        self.print_freq = config.print_freq
        self.plot_freq = config.plot_freq
        self.model_name = 'ram_{}_{}x{}_{}'.format(
            config.num_glimpses, config.patch_size,
            config.patch_size, config.glimpse_scale
        )

        self.plot_dir = './plots/' + self.model_name + '/'
        if not os.path.exists(self.plot_dir):
            os.makedirs(self.plot_dir)

        # configure tensorboard logging
        # if self.use_tensorboard:
        #     tensorboard_dir = self.logs_dir + self.model_name
        #     print('[*] Saving tensorboard logs to {}'.format(tensorboard_dir))
        #     if not os.path.exists(tensorboard_dir):
        #         os.makedirs(tensorboard_dir)
        #     configure(tensorboard_dir)

        # build RAM model
        self.model = RecurrentAttention(
            self.patch_size, self.num_patches, self.glimpse_scale,
            self.num_channels, self.loc_hidden, self.glimpse_hidden,
            self.std, self.hidden_size, self.num_classes,
        )
        if self.use_gpu:
            self.model.cuda()

        print('[*] Number of model parameters: {:,}'.format(
            sum([p.data.nelement() for p in self.model.parameters()])))

        # # initialize optimizer and scheduler
        # self.optimizer = optim.SGD(
        #     self.model.parameters(), lr=self.lr, momentum=self.momentum,
        # )
        # self.scheduler = ReduceLROnPlateau(
        #     self.optimizer, 'min', patience=self.lr_patience
        # )
        self.optimizer = optim.Adam(
            self.model.parameters(), lr=1e-4,
        )

        wandb.watch(self.model, log="all")

    def reset(self, batch_size):
        """
        Initialize the hidden state of the core network
        and the location vector.

        This is called once every time a new minibatch
        `x` is introduced.
        """
        dtype = (
            torch.cuda.FloatTensor if self.use_gpu else torch.FloatTensor
        )

        h_t = torch.zeros(batch_size, self.hidden_size)
        h_t = Variable(h_t).type(dtype)

        l_t = None

        return h_t, l_t

    def train(self):
        """
        Train the model on the training set.

        A checkpoint of the model is saved after each epoch
        and if the validation accuracy is improved upon,
        a separate ckpt is created for use on the test set.
        """
        # load the most recent checkpoint
        if self.resume:
            self.load_checkpoint(best=False)

        print("\n[*] Train on {} samples, validate on {} samples".format(
            self.num_train, self.num_valid)
        )

        for epoch in range(self.start_epoch, self.epochs):
            self.curr_epoch = epoch
            print(
                '\nEpoch: {}/{} - LR: {:.6f}'.format(
                    epoch+1, self.epochs, self.lr)
            )



            # train for 1 epoch
            train_loss, train_acc, glimpses = self.train_one_epoch(epoch)
            # evaluate on validation set
            valid_loss, valid_acc, val_glimpses = self.validate(epoch)
            # # reduce lr if validation loss plateaus
            # self.scheduler.step(valid_loss)

            is_best = valid_acc > self.best_valid_acc
            msg1 = "train loss: {:.3f} - train acc: {:.3f} - train glm {:.3f}"
            msg2 = "- val loss: {:.3f} - val acc: {:.3f} - val glm {:.3f}"
            if is_best:
                self.counter = 0
                msg2 += " [*]"
            msg = msg1 + msg2
            print(msg.format(train_loss, train_acc, glimpses, valid_loss, valid_acc, val_glimpses))
            wandb.log({
                "test_accuracy": valid_acc,
                "train_accuracy": train_acc,
                'train_glimpses': glimpses,
                'val_glimpses': val_glimpses
            })

            # check for improvement
            if not is_best:
                self.counter += 1
            if self.counter > self.train_patience:
                print("[!] No improvement in a while, stopping training.")
                return
            self.best_valid_acc = max(valid_acc, self.best_valid_acc)
            self.save_checkpoint(
                {'epoch': epoch + 1,
                 'model_state': self.model.state_dict(),
                 'optim_state': self.optimizer.state_dict(),
                 'best_valid_acc': self.best_valid_acc,
                 }, is_best
            )
            # decay
            # for param_group in self.optimizer.param_groups:
            #     old_lr = param_group['lr']
            #     param_group['lr'] = param_group['lr']*0.98
            # print(f"Reducing LR from {old_lr} to {old_lr*0.98}")

    def train_one_epoch(self, epoch):
        """
        Train the model for 1 epoch of the training set.

        An epoch corresponds to one full pass through the entire
        training set in successive mini-batches.

        This is used by train() and should not be called manually.
        """
        batch_time = AverageMeter()
        losses = AverageMeter()
        accs = AverageMeter()
        glimpses = AverageMeter()

        tic = time.time()
        with tqdm(total=self.num_train) as pbar:
            for i, (x, y) in enumerate(self.train_loader):
                if self.use_gpu:
                    x, y = x.cuda(), y.cuda()
                x, y = Variable(x), Variable(y)

                import numpy as np
                for b in range(x.shape[0]):
                    random_rotation = np.random.randint(0, 4)
                    if random_rotation == 0:
                        pass
                    elif random_rotation == 1:
                        x[b] = x[b].transpose(0, 1)
                    elif random_rotation == 2:
                        x[b] = x[b].flip(1)
                    else:
                        x[b] = x[b].transpose(0, 1).flip(1)


                plot = False
                if (epoch % self.plot_freq == 0) and (i == 0):
                    plot = True
                loss, glm, acc = self.rollout(x, y)
                glimpses.update(glm)
                # store
                try:
                    losses.update(loss.data[0], x.size()[0])
                    accs.update(acc.data[0], x.size()[0])
                except:
                    losses.update(loss.data.item(), x.size()[0])
                    accs.update(acc.data.item(), x.size()[0])
                # compute gradients and update SGD
                self.optimizer.zero_grad()
                loss = loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)

                self.optimizer.step()

                # measure elapsed time
                toc = time.time()
                batch_time.update(toc-tic)

                try:
                    loss_data =loss.data[0]
                    acc_data = acc.data[0]
                except IndexError:
                    loss_data = loss.data.item()
                    acc_data = acc.data.item()
                pbar.set_description(
                    (
                        "{:.1f}s - loss: {:.3f} - acc: {:.3f}".format(
                            (toc-tic), loss_data, acc_data
                        )
                    )
                )
                batch_size = x.shape[0]
                pbar.update(batch_size)

            return losses.avg, accs.avg, glimpses.avg

    def rollout(self, x, y):
        batch_size = x.shape[0]
        h_t, l_t = self.reset(batch_size)
        h_t = None

        # extract the glimpses
        total_glimpses = 0
        # we want to run this loop UNTIL they are all done,
        # that will involve some "dummy" forward passes
        # need to track when each element of the batch is actually
        # done to do proper masking

        # use None so everything errors out if I don't explicitly set it
        prob_as = [None for _ in range(batch_size)]
        log_ds = [None for _ in range(batch_size)]
        done_indices = [-1 for _ in range(batch_size)]
        timeouts = [False for _ in range(batch_size)]
        glimpse_totals = [None for _ in range(batch_size)]
        for t in range(self.num_glimpses):
            total_glimpses += 1
            # forward pass through model
            h_t, l_t, prob_a, d_t, log_probds = self.model(x, l_t, h_t)
            for batch_ind in range(batch_size):
                if done_indices[batch_ind] > -1:
                    # already done
                    continue
                elif d_t[batch_ind] == 1:
                    glimpse_totals[batch_ind] = t + 1
                    # mark as done
                    done_indices[batch_ind] = t
                    # save the log_d
                    log_ds[batch_ind] = log_probds[batch_ind]
                    # save the prob_a
                    prob_as[batch_ind] = prob_a[batch_ind]
                elif t == self.num_glimpses - 1:
                    # glimpses are timing out
                    timeouts[batch_ind] = True
                    glimpse_totals[batch_ind] = t + 1
                    # mark as done
                    done_indices[batch_ind] = t
                    # save the log_d
                    log_ds[batch_ind] = log_probds[batch_ind]
                    # save the prob_a
                    prob_as[batch_ind] = prob_a[batch_ind]
            if all([done_index > -1 for done_index in done_indices]):
                break
        prob_as = torch.stack(prob_as)
        log_ds = torch.stack(log_ds)
        # calculate reward
        predicted = torch.max(prob_as, 1)[1]
        R = (predicted.detach() == y.long()).float()
        R = R.unsqueeze(1).repeat(1, self.num_glimpses)

        # compute losses for differentiable modules
        loss_action = F.nll_loss(prob_as, y)
        decision_target = []
        decision_scaling = []
        for batch_ind in range(batch_size):
            if timeouts[batch_ind]:
                decision_target.append(1)
                decision_scaling.append(1.0)
            elif R[batch_ind][0] == 1:
                decision_target.append(1)
                decision_scaling.append(self.config.loss_balance)
            elif R[batch_ind][0] == 0:
                decision_target.append(0)
                decision_scaling.append(1.0)
            else:
                raise RuntimeError("how did we get here?")
        decision_target = torch.tensor(decision_target)
        decision_scaling = torch.tensor(decision_scaling)
        # now take the error between our decision target and log_ds
        loss_decision = (F.nll_loss(log_ds, decision_target,
                                    reduction='none') * decision_scaling).mean()
        # sum up into a hybrid loss
        loss = loss_action + loss_decision
        # compute accuracy
        correct = (predicted == y).float()
        acc = 100 * (correct.sum() / len(y))

        return loss, sum(glimpse_totals) / batch_size, acc

    def validate(self, epoch):
        """
        Evaluate the model on the validation set.
        """
        losses = AverageMeter()
        accs = AverageMeter()
        glimpses = AverageMeter()

        for i, (x, y) in enumerate(self.valid_loader):
            if self.use_gpu:
                x, y = x.cuda(), y.cuda()
            x, y = Variable(x), Variable(y)
            loss, glm, acc = self.rollout(x,y)

            glimpses.update(glm)
            # store
            try:
                losses.update(loss.data[0], x.size()[0])
                accs.update(acc.data[0], x.size()[0])
            except:
                losses.update(loss.data.item(), x.size()[0])
                accs.update(acc.data.item(), x.size()[0])

        return losses.avg, accs.avg, glimpses.avg

    def test(self):
        """
        Test the model on the held-out test data.
        This function should only be called at the very
        end once the model has finished training.
        """
        correct = 0

        # load the best checkpoint
        self.load_checkpoint(best=self.best)

        for i, (x, y) in enumerate(self.test_loader):
            if self.use_gpu:
                x, y = x.cuda(), y.cuda()
            x, y = Variable(x, volatile=True), Variable(y)

            # duplicate 10 times
            # x = x.repeat(self.M, 1, 1, 1)

            # initialize location vector and hidden state
            self.batch_size = x.shape[0]
            h_t, l_t = self.reset()
            h_t = None

            # extract the glimpses
            for t in range(self.num_glimpses - 1):
                # forward pass through model
                h_t, l_t, b_t, p = self.model(x, l_t, h_t, valid=True)

            # last iteration
            h_t, l_t, b_t, log_probas, p = self.model(
                x, l_t, h_t, last=True, valid=True
            )

            log_probas = log_probas.view(
                -1, log_probas.shape[-1]
            )
            log_probas = torch.mean(log_probas, dim=0)

            pred = log_probas.data.max(1, keepdim=True)[1]
            correct += pred.eq(y.data.view_as(pred)).cpu().sum()

        perc = (100. * correct) / (self.num_test)
        error = 100 - perc
        print(
            '[*] Test Acc: {}/{} ({:.2f}% - {:.2f}%)'.format(
                correct, self.num_test, perc, error)
        )

    def save_checkpoint(self, state, is_best):
        """
        Save a copy of the model so that it can be loaded at a future
        date. This function is used when the model is being evaluated
        on the test data.

        If this model has reached the best validation accuracy thus
        far, a seperate file with the suffix `best` is created.
        """
        # print("[*] Saving model to {}".format(self.ckpt_dir))

        filename = self.model_name + '_ckpt.pth.tar'
        ckpt_path = os.path.join(self.ckpt_dir, filename)
        torch.save(state, ckpt_path)

        if is_best:
            filename = self.model_name + '_model_best.pth.tar'
            shutil.copyfile(
                ckpt_path, os.path.join(self.ckpt_dir, filename)
            )

    def load_checkpoint(self, best=False):
        """
        Load the best copy of a model. This is useful for 2 cases:

        - Resuming training with the most recent model checkpoint.
        - Loading the best validation model to evaluate on the test data.

        Params
        ------
        - best: if set to True, loads the best model. Use this if you want
          to evaluate your model on the test data. Else, set to False in
          which case the most recent version of the checkpoint is used.
        """
        print("[*] Loading model from {}".format(self.ckpt_dir))

        filename = self.model_name + '_ckpt.pth.tar'
        if best:
            filename = self.model_name + '_model_best.pth.tar'
        ckpt_path = os.path.join(self.ckpt_dir, filename)
        ckpt = torch.load(ckpt_path)

        # load variables from checkpoint
        self.start_epoch = ckpt['epoch']
        self.best_valid_acc = ckpt['best_valid_acc']
        self.model.load_state_dict(ckpt['model_state'])
        self.optimizer.load_state_dict(ckpt['optim_state'])

        if best:
            print(
                "[*] Loaded {} checkpoint @ epoch {} "
                "with best valid acc of {:.3f}".format(
                    filename, ckpt['epoch'], ckpt['best_valid_acc'])
            )
        else:
            print(
                "[*] Loaded {} checkpoint @ epoch {}".format(
                    filename, ckpt['epoch'])
            )
