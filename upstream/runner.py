import collections
import datetime
from torch import nn
import os
import math
import torch
from tqdm import tqdm
from tensorboardX import SummaryWriter
from upstream.model import TransformerConfig, UpstreamModel
from upstream.optimization import BertAdam
from upstream.utils import plot_spectrogram_to_numpy

class Runner():
    ''' Handler for complete pre-training progress of upstream models '''

    def __init__(self, args, teacher_config, student_config, dataloader, teacher_ckpdir, logger):

        self.device = torch.device('cuda') if (args.gpu and torch.cuda.is_available()) else torch.device('cpu')
        if torch.cuda.is_available(): print('[Runner] - CUDA is available!')
        self.args = args
        self.dataloader = dataloader
        self.log = SummaryWriter(teacher_ckpdir)

        # Distiller
        self.temperature = args.temperature
        self.ce_loss_fct = torch.nn.KLDivLoss(reduction="batchmean")

        #
        # Teacher config and args
        #
        self.teacher_global_step = 1

        self.teacher_model_kept = []
        self.teacher_config = teacher_config
        self.teacher_ckpdir = teacher_ckpdir

        # optimizer
        self.teacher_learning_rate = float(teacher_config['optimizer']['learning_rate'])
        self.teacher_warmup_proportion = teacher_config['optimizer']['warmup_proportion']
        self.teacher_gradient_accumulation_steps = teacher_config['optimizer']['gradient_accumulation_steps']
        self.teacher_gradient_clipping = teacher_config['optimizer']['gradient_clipping']

        # Training details
        self.teacher_apex = teacher_config['runner']['apex']
        self.teacher_total_steps = teacher_config['runner']['total_steps']
        self.teacher_log_step = teacher_config['runner']['log_step']
        self.teacher_save_step = teacher_config['runner']['save_step']
        self.teacher_duo_feature = teacher_config['runner']['duo_feature']
        self.teacher_max_keep = teacher_config['runner']['max_keep']

        # model
        self.teacher_transformer_config = teacher_config['transformer']
        self.teacher_dr = teacher_config['transformer']['downsample_rate']
        self.teacher_dual_transformer = teacher_config['transformer']['dual_transformer'] if 'dual_transformer' in \
                                                                                             teacher_config[
                                                                                                 'transformer'] else False
        self.teacher_wave_transformer = teacher_config['transformer']['wave_transformer'] if 'wave_transformer' in \
                                                                                             teacher_config[
                                                                                                 'transformer'] else False
        print(f'[Runner] - Using features pre-extracted and saved')
        self.teacher_input_dim = self.teacher_transformer_config['input_dim']

        #
        # Student config and args
        #
        self.student_global_step = 1

        self.student_model_kept = []
        self.student_config = student_config
        self.student_ckpdir = teacher_ckpdir

        # optimizer
        self.student_learning_rate = float(student_config['optimizer']['learning_rate'])
        self.student_warmup_proportion = student_config['optimizer']['warmup_proportion']
        self.student_gradient_accumulation_steps = student_config['optimizer']['gradient_accumulation_steps']
        self.student_gradient_clipping = student_config['optimizer']['gradient_clipping']

        # Training details
        self.student_apex = student_config['runner']['apex']
        self.student_total_steps = student_config['runner']['total_steps']
        self.student_log_step = student_config['runner']['log_step']
        self.student_save_step = student_config['runner']['save_step']
        self.student_duo_feature = student_config['runner']['duo_feature']
        self.student_max_keep = student_config['runner']['max_keep']

        # model
        self.student_transformer_config = student_config['transformer']
        self.student_dr = student_config['transformer']['downsample_rate']
        self.student_dual_transformer = student_config['transformer']['dual_transformer'] if 'dual_transformer' in \
                                                                                             student_config[
                                                                                                 'transformer'] else False
        self.student_wave_transformer = student_config['transformer']['wave_transformer'] if 'wave_transformer' in \
                                                                                             student_config[
                                                                                                 'transformer'] else False
        self.student_input_dim = self.student_transformer_config['input_dim']

        # loss
        self.loss_funtion = nn.L1Loss()

        # logger
        self.logger = logger

    def set_model(self):
        # build the Teacher Transformer model with speech prediction head
        if self.teacher_wave_transformer:
            print('[Runner] - Initializing Wave Transformer model...')
            assert self.teacher_input_dim == 1
        else:
            print('[Runner] - Initializing Transformer model...')
        teacher_model_config = TransformerConfig(self.teacher_config)

        self.teacher_model = UpstreamModel(teacher_model_config, self.teacher_input_dim,
                                                               None).to(self.device)
        if self.args.parent:
            self.teacher_model.train()
        else:
            self.teacher_model.eval()

        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        if self.args.parent:
            # Setup optimizer
            teacher_param_optimizer = list(self.teacher_model.named_parameters())

            teacher_optimizer_grouped_parameters = [
                {'params': [p for n, p in teacher_param_optimizer if not any(nd in n for nd in no_decay)],
                 'weight_decay': 0.01},
                {'params': [p for n, p in teacher_param_optimizer if any(nd in n for nd in no_decay)],
                 'weight_decay': 0.0}
            ]

            if 'type' not in self.teacher_config['optimizer']:
                self.teacher_config['optimizer']['type'] = 'adam'
            print('[Runner] - Optimizer: ' + (
                'apex Fused Adam' if self.teacher_apex else str(self.teacher_config['optimizer']['type'])))

            if self.teacher_config['optimizer']['type'] == 'adam':
                self.teacher_optimizer = BertAdam(teacher_optimizer_grouped_parameters,
                                                  lr=self.teacher_learning_rate,
                                                  warmup=self.teacher_warmup_proportion,
                                                  t_total=self.teacher_total_steps,
                                                  schedule='warmup_linear')
            else:
                raise NotImplementedError()

        if self.args.teacher_resume is not None:
            self.load_model(self.args.teacher_resume, 'teacher')

        #
        # build the Student Transformer model with speech prediction head
        #
        if self.student_wave_transformer:
            print('[Runner] - Initializing Wave Transformer model...')
            assert self.student_input_dim == 1
        else:
            print('[Runner] - Initializing Transformer model...')
        student_model_config = TransformerConfig(self.student_config)

        self.student_model = UpstreamModel(student_model_config, self.teacher_input_dim, None).to(
            self.device)
        self.student_model.train()

        student_param_optimizer = list(self.student_model.named_parameters())

        student_optimizer_grouped_parameters = [
            {'params': [p for n, p in student_param_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay': 0.01},
            {'params': [p for n, p in student_param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

        if 'type' not in self.student_config['optimizer']:
            self.student_config['optimizer']['type'] = 'adam'
        print('[Runner] - Optimizer: ' + (
            'apex Fused Adam' if self.teacher_apex else str(self.student_config['optimizer']['type'])))

        if self.student_config['optimizer']['type'] == 'adam':
            self.student_optimizer = BertAdam(student_optimizer_grouped_parameters,
                                              lr=self.student_learning_rate,
                                              warmup=self.student_warmup_proportion,
                                              t_total=self.student_total_steps,
                                              schedule='warmup_linear')
        else:
            raise NotImplementedError()

        if self.args.student_resume is not None:
            self.load_model(self.args.student_resume, 'student')

        if self.args.multi_gpu:
            self.teacher_model = torch.nn.DataParallel(self.teacher_model)
            print('[Runner] - Multi-GPU training Enabled: ' + str(torch.cuda.device_count()))
        print('[Runner] - Number of teacher parameters: ' + str(
            sum(p.numel() for p in self.teacher_model.parameters() if p.requires_grad)))
        print('[Runner] - Number of student parameters: ' + str(
            sum(p.numel() for p in self.student_model.parameters() if p.requires_grad)))

    def load_model_from_teacher(self, ckptpth):
        # Only select part of ckpt
        print('[Runner] - Load Parent Model From ' + ckptpth)
        ckpt = torch.load(ckptpth)
        tmp_transformer = collections.OrderedDict()
        for key, value in ckpt['Transformer'].items():
            if not 'encoder.layer.2' in key:
                tmp_transformer[key] = value
        self.student_model.Transformer.load_state_dict(tmp_transformer)
        self.student_model.SpecHead.load_state_dict(ckpt['SpecHead'])

    def load_model(self, ckptpth, kind):
        if kind == 'teacher':
            ckpt = torch.load(ckptpth)
            self.teacher_model.Transformer.load_state_dict(ckpt['Transformer'])
            self.teacher_model.SpecHead.load_state_dict(ckpt['SpecHead'])
            # self.teacher_optimizer.load_state_dict(ckpt['Optimizer'])
            self.teacher_global_step = ckpt['Global_step']
        elif kind == 'student':
            # Only select part of ckpt
            ckpt = torch.load(ckptpth)
            '''tmp_transformer = collections.OrderedDict()
            for key, value in ckpt['Transformer'].items():
                if not 'encoder.layer.2' in key:
                    tmp_transformer[key] = value
            tmp_optimizer = {}
            tmp_optimizer['param_groups'] = ckpt['Optimizer']['param_groups']
            tmp_optimizer['state'] = {}
            count = 0
            for key, item in ckpt['Optimizer']['state'].items():
                if count == 42:
                    break
                tmp_optimizer['state'][key] = item
                count = count + 1'''
            self.student_model.Transformer.load_state_dict(ckpt['Transformer'])
            self.student_model.SpecHead.load_state_dict(ckpt['SpecHead'])
            self.student_optimizer.load_state_dict(ckpt['Optimizer'])
            self.student_global_step = ckpt['Global_step']

    def up_sample_frames(self, spec, return_first=False):
        if len(spec.shape) != 3:
            spec = spec.unsqueeze(0)
            assert (len(spec.shape) == 3), 'Input should have acoustic feature of shape BxTxD'
        # spec shape: [batch_size, sequence_length // downsample_rate, output_dim * downsample_rate]
        spec_flatten = spec.view(spec.shape[0], spec.shape[1] * self.teacher_dr, spec.shape[2] // self.teacher_dr)
        if return_first: return spec_flatten[0]
        return spec_flatten  # spec_flatten shape: [batch_size, sequence_length * downsample_rate, output_dim // downsample_rate]

    def down_sample_frames(self, spec):
        left_over = spec.shape[1] % self.teacher_dr
        if left_over != 0: spec = spec[:, :-left_over, :]
        spec_stacked = spec.view(spec.shape[0], spec.shape[1] // self.teacher_dr, spec.shape[2] * self.teacher_dr)
        return spec_stacked

    def process_data(self, spec):
        """Process training data for the masked acoustic model"""
        with torch.no_grad():

            assert (len(
                spec) == 5), 'dataloader should return (spec_masked, pos_enc, mask_label, attn_mask, spec_stacked)'
            # Unpack and Hack bucket: Bucketing should cause acoustic feature to have shape 1xBxTxD'
            spec_masked = spec[0].squeeze(0)
            pos_enc = spec[1].squeeze(0)
            mask_label = spec[2].squeeze(0)
            attn_mask = spec[3].squeeze(0)
            spec_stacked = spec[4].squeeze(0)

            spec_masked = spec_masked.to(device=self.device)
            if pos_enc.dim() == 3:
                # pos_enc: (batch_size, seq_len, hidden_size)
                # GPU memory need (batch_size * seq_len * hidden_size)
                pos_enc = pos_enc.float().to(device=self.device)
            elif pos_enc.dim() == 2:
                # pos_enc: (seq_len, hidden_size)
                # GPU memory only need (seq_len * hidden_size) even after expanded
                pos_enc = pos_enc.float().to(device=self.device).expand(spec_masked.size(0), *pos_enc.size())
            mask_label = mask_label.bool().to(device=self.device)
            attn_mask = attn_mask.float().to(device=self.device)
            spec_stacked = spec_stacked.to(device=self.device)

        return spec_masked, pos_enc, mask_label, attn_mask, spec_stacked  # (x, pos_enc, mask_label, attention_mask. y)

    def save_model(self, name='states', to_path=None, type='teacher'):
        # Save Teacher
        if type == 'teacher':
            all_states = {
                'SpecHead': self.teacher_model.SpecHead.state_dict() if not self.args.multi_gpu else self.teacher_model.module.SpecHead.state_dict(),
                'Transformer': self.teacher_model.Transformer.state_dict() if not self.args.multi_gpu else self.teacher_model.module.Transformer.state_dict(),
            }

            all_states['Optimizer'] = self.teacher_optimizer.state_dict()
            all_states['Global_step'] = self.teacher_global_step
            all_states['Settings'] = {'Config': self.teacher_config, 'Paras': self.args}

            if to_path is None:
                new_model_path = '{}/{}-{}.ckpt'.format(self.teacher_ckpdir, 'exp/teacher_' + name,
                                                        self.teacher_global_step)
            else:
                new_model_path = to_path + '/exp_11_6/'

            torch.save(all_states, new_model_path)
            self.teacher_model_kept.append(new_model_path)

            if len(self.teacher_model_kept) >= self.teacher_max_keep:
                os.remove(self.teacher_model_kept[0])
                self.teacher_model_kept.pop(0)
        else:
            # Save student
            all_states = {
                'SpecHead': self.student_model.SpecHead.state_dict() if not self.args.multi_gpu else self.student_model.module.SpecHead.state_dict(),
                'Transformer': self.student_model.Transformer.state_dict() if not self.args.multi_gpu else self.student_model.module.Transformer.state_dict(),
            }

            all_states['Optimizer'] = self.student_optimizer.state_dict()
            all_states['Global_step'] = self.student_global_step
            all_states['Settings'] = {'Config': self.student_config, 'Paras': self.args}

            if to_path is None:
                new_model_path = '{}/{}-{}.ckpt'.format(self.student_ckpdir, 'student_' + name,
                                                        self.student_global_step)
            else:
                new_model_path = to_path

            torch.save(all_states, new_model_path)
            self.student_model_kept.append(new_model_path)

            if len(self.student_model_kept) >= self.student_max_keep:
                os.remove(self.student_model_kept[0])
                self.student_model_kept.pop(0)
        # logger
        self.end = datetime.datetime.now()
        self.logger.info(str(self.student_global_step) + ' start time : ' + str(self.start))
        self.logger.info(str(self.student_global_step) + ' end time : ' + str(self.end))
        self.logger.info(str(self.student_global_step) + ' time interval : ' + str(self.end))
        self.start = self.end

    def train(self):
        ''' Self-Supervised Pre-Training of Transformer Model'''

        if self.args.multi_gpu:
            self.student_model = torch.nn.DataParallel(self.student_model)
            print('[Runner] - Multi-GPU training Enabled: ' + str(torch.cuda.device_count()))

        ''' Distiller Pre-Training of Transformer Model'''
        pbar = tqdm(total=self.student_total_steps)
        pbar.n = self.student_global_step - 1

        self.start = datetime.datetime.now()
        while self.student_global_step <= self.student_total_steps:

            progress = tqdm(self.dataloader, desc="Iteration")

            step = 0
            step = 0
            loss_val = 0
            loss_tol = 0
            for batch in progress:
                batch_is_valid, *batch = batch
                try:
                    if self.student_global_step > self.student_total_steps: break
                    if not batch_is_valid: continue
                    step += 1

                    spec_masked, pos_enc, mask_label, attn_mask, spec_stacked = self.process_data(batch)
                    with torch.no_grad():
                        useless_loss, t_pred = self.teacher_model(spec_masked, pos_enc, mask_label, attn_mask,
                                                                  spec_stacked)

                    old_loss, pred_spec = self.student_model(spec_masked, pos_enc, mask_label, attn_mask, spec_stacked)

                    t_pred_slct, pred_spec_slct = t_pred.masked_select(mask_label), pred_spec.masked_select(mask_label)
                    t_pred_slct, pred_spec_slct = t_pred_slct.view(-1, t_pred_slct.size(-1)), \
                        pred_spec_slct.view(-1, pred_spec_slct.size(-1))

                    loss_ce = self.loss_funtion(pred_spec_slct, t_pred_slct)
                    loss = (5.0 * loss_ce + 2.0 * old_loss) / 7.0
                    # loss = old_loss

                    # Accumulate Loss
                    if self.student_gradient_accumulation_steps > 1:
                        loss = loss / self.student_gradient_accumulation_steps
                    if self.args.multi_gpu:
                        loss = loss.sum()
                        loss.backward()
                    else:
                        loss.backward()
                    loss_val += loss.item()

                    # print loss
                    loss_tol = loss_tol + loss.item()
                    if step % 100 == 0:
                        loss_tol = loss_tol / 100
                        self.logger.info('The ' + str(self.student_global_step) + ' loss is: ' + str(loss_tol))
                        loss_tol = 0

                    # Update
                    if (step + 1) % self.student_gradient_accumulation_steps == 0:

                        # Step
                        grad_norm = torch.nn.utils.clip_grad_norm_(self.student_model.parameters(),
                                                                   self.student_gradient_clipping)
                        if math.isnan(grad_norm):
                            print('[Runner] - Error : grad norm is NaN @ step ' + str(self.student_global_step))
                        else:
                            self.student_optimizer.step()
                        self.student_optimizer.zero_grad()

                        if self.student_global_step % self.student_log_step == 0:
                            # Log
                            self.log.add_scalar('lr', self.student_optimizer.get_lr()[0], self.student_global_step)
                            self.log.add_scalar('loss', (loss_val), self.student_global_step)
                            self.log.add_scalar('gradient norm', grad_norm, self.student_global_step)
                            progress.set_description("Loss %.4f" % (loss_val))

                            # tensorboard log
                            spec_list = [spec_masked, pred_spec, spec_stacked]
                            name_list = ['mask_spec', 'pred_spec', 'true_spec']

                            for i in range(len(spec_list)):
                                spec = self.up_sample_frames(spec_list[i][0], return_first=True)
                                spec = plot_spectrogram_to_numpy(spec.data.cpu().numpy())
                                self.log.add_image(name_list[i], spec, self.student_global_step)

                        if self.student_global_step % self.student_save_step == 0:
                            self.save_model('states', None, 'student')

                            # tensorboard log
                            spec_list = [spec_masked, pred_spec, spec_stacked]
                            name_list = ['mask_spec', 'pred_spec', 'true_spec']

                            for i in range(len(spec_list)):
                                if i == 0 and self.student_wave_transformer:
                                    self.log.add_audio(name_list[0], spec_list[0][0].data.cpu().numpy(),
                                                       self.student_global_step,
                                                       self.student_config['online']['sample_rate'])
                                    continue
                                spec = self.up_sample_frames(spec_list[i][0], return_first=True)
                                spec = plot_spectrogram_to_numpy(spec.data.cpu().numpy())
                                self.log.add_image(name_list[i], spec, self.student_global_step)

                        loss_val = 0
                        pbar.update(1)
                        self.student_global_step += 1

                except RuntimeError as e:
                    if 'CUDA out of memory' in str(e):
                        print('CUDA out of memory at step: ', self.student_global_step)
                        torch.cuda.empty_cache()
                        self.student_optimizer.zero_grad()
                    else:
                        raise

        pbar.close()
        self.log.close()
