from contextlib import contextmanager
import copy
import torch
import torch.nn as nn
import numpy as np
from matplotlib import pyplot as plt

from spirl.components.base_model import BaseModel
from spirl.components.logger import Logger
from spirl.components.checkpointer import CheckpointHandler
from spirl.modules.losses import BCELogitsLoss
from spirl.modules.subnetworks import Predictor, Encoder
from spirl.utils.general_utils import AttrDict, ParamDict, map_dict
from spirl.utils.vis_utils import fig2img
from spirl.utils.pytorch_utils import RemoveSpatial, ResizeSpatial
from spirl.modules.layers import LayerBuilderParams


class DemoDiscriminator(BaseModel):
    """Simple feed forward predictor network that distinguishes demo and non-demo states."""
    def __init__(self, params, logger=None):
        BaseModel.__init__(self, logger)
        self._hp = self._default_hparams()
        self._hp.overwrite(params)  # override defaults with config file
        self._hp.builder = LayerBuilderParams(self._hp.use_convs, self._hp.normalization)
        self.device = self._hp.device

        # set up demo dataset
        if self._hp.demo_data_path is not None:
            self._hp.demo_data_conf.device = self.device
            self._demo_data_loader = self._hp.demo_data_conf.dataset_spec.dataset_class(
                self._hp.demo_data_path, self._hp.demo_data_conf, resolution=self._hp.demo_data_conf.dataset_spec.res,
                phase="train", shuffle=True).get_data_loader(self._hp.batch_size,
                                                             n_repeat=10000)  # making new iterators is slow, so repeat often
            self._demo_data_iter = iter(self._demo_data_loader)

        self.build_network()

    def _default_hparams(self):
        # put new parameters in here:
        return super()._default_hparams().overwrite(ParamDict({
            'state_dim': None,
            'action_dim': None,
            'use_convs': False,
            'device': None,
            'nz_enc': 32,               # number of dimensions in encoder-latent space
            'nz_mid': 32,               # number of dimensions for internal feature spaces
            'n_processing_layers': 3,   # number of layers in MLPs
            'action_input': False,      # if True, conditions on action in addition to state
            'demo_data_conf': {},       # data configuration for demo dataset
            'demo_data_path': None,     # path to demo data directory
        }))

    def build_network(self):
        assert not self._hp.use_convs   # currently only supports non-image inputs
        self.demo_discriminator = self.build_discriminator()

    def forward(self, inputs):
        """forward pass at training time"""
        # run discriminator in test-mode if no dataset for training is given
        if self._hp.demo_data_path is None:
            return self.demo_discriminator(inputs)

        output = AttrDict()

        # sample demo inputs
        demo_inputs = self._get_demo_batch()

        # run discriminator on demo and non-demo data
        output.demo_logits = self.demo_discriminator(self._discriminator_input(demo_inputs))
        output.nondemo_logits = self.demo_discriminator(self._discriminator_input(inputs))
        output.logits = torch.cat((output.demo_logits, output.nondemo_logits))

        # compute targets for discriminator outputs
        output.targets = torch.cat((torch.ones_like(output.demo_logits), torch.zeros_like(output.nondemo_logits)))

        return output

    def loss(self, model_output, inputs):
        losses = AttrDict()

        # discriminator loss
        losses.discriminator_loss = BCELogitsLoss(1.)(model_output.logits, model_output.targets)

        losses.total = self._compute_total_loss(losses)
        return losses

    def _log_outputs(self, model_output, inputs, losses, step, log_images, phase, logger, **logging_kwargs):
        # log videos/gifs in tensorboard
        if 'demo_logits' in model_output:
            logger.log_scalar(torch.sigmoid(model_output.demo_logits).mean(), "p_demo", step, phase)
            logger.log_scalar(torch.sigmoid(model_output.nondemo_logits).mean(), "p_nondemo", step, phase)

    def build_discriminator(self):
        return Predictor(self._hp, input_size=self.discriminator_input_size, output_size=1,
                         num_layers=self._hp.n_processing_layers, mid_size=self._hp.nz_mid)

    def _discriminator_input(self, inputs):
        if not self._hp.action_input:
            return inputs.states[:, 0]
        else:
            return torch.cat((inputs.states[:, 0], inputs.actions[:, 0]), dim=-1)

    def _get_demo_batch(self):
        try:
            demo_batch = next(self._demo_data_iter)
        except StopIteration:
            self._demo_data_iter = iter(self._demo_data_loader)
            demo_batch = next(self._demo_data_iter)
        return AttrDict(map_dict(lambda x: x.to(self.device), demo_batch))

    def evaluate_discriminator(self, state):
        """Evaluates discriminator probability."""
        return nn.Sigmoid()(self.demo_discriminator(state))

    @property
    def resolution(self):
        return 64       # return dummy resolution, images are not used by this model

    @property
    def discriminator_input_size(self):
        return self._hp.state_dim if not self._hp.action_input else self._hp.state_dim + self._hp.action_dim

    @contextmanager
    def val_mode(self):
        pass
        yield
        pass


class ImageDemoDiscriminator(DemoDiscriminator):
    """Implements demo discriminator with image input."""
    def _default_hparams(self):
        default_dict = ParamDict({
            'discriminator_input_res': 32,      # input resolution of prior images
            'encoder_ngf': 8,           # number of feature maps in shallowest level of encoder
            'n_input_frames': 1,        # number of prior input frames
        })
        # add new params to parent params
        return super()._default_hparams().overwrite(default_dict)

    def _updated_encoder_params(self):
        params = copy.deepcopy(self._hp)
        return params.overwrite(AttrDict(
            use_convs=True,
            use_skips=False,                  # no skip connections needed bc we are not reconstructing
            img_sz=self._hp.discriminator_input_res,  # image resolution
            input_nc=3*self._hp.n_input_frames,  # number of input feature maps
            ngf=self._hp.encoder_ngf,         # number of feature maps in shallowest level
            nz_enc=self.discriminator_input_size,     # size of image encoder output feature
            builder=LayerBuilderParams(use_convs=True, normalization=self._hp.normalization)
        ))

    def build_discriminator(self):
        return nn.Sequential(
            ResizeSpatial(self._hp.discriminator_input_res),
            Encoder(self._updated_encoder_params()),
            RemoveSpatial(),
            super().build_discriminator(),
        )

    def _discriminator_input(self, inputs):
        assert not self._hp.action_input        # action input currently not supported for image discriminator
        return inputs.images[:, :self._hp.n_input_frames]\
            .reshape(inputs.images.shape[0], -1, self.resolution, self.resolution)

    def filter_input(self, raw_input):
        assert raw_input.shape[-1] == raw_input.shape[-2] == self.resolution
        assert len(raw_input.shape) == 4        # [batch, channels, res, res]
        return raw_input[:, :self._hp.n_input_frames*3]

    @property
    def discriminator_input_size(self):
        return self._hp.nz_mid

    @property
    def resolution(self):
        return self._hp.discriminator_input_res


class DemoDiscriminatorLogger(Logger):
    """
    Logger for Skill Space model. No extra methods needed to implement by
    environment-specific logger implementation.
    """
    N_LOGGING_SAMPLES = 5000        # number of samples from demo / non-demo used for logging

    def visualize(self, model_output, inputs, losses, step, phase, logger):
        pass

    @staticmethod
    def plot_discriminator_samples(demo_samples, non_demo_samples, logger, step, phase):
        # plot histogram of demo and non-demo sample probabilities
        bins = np.linspace(0, 1, 50)
        fig = plt.figure()
        plt.hist(demo_samples.p_demo, bins, alpha=0.5, label='demo')
        plt.hist(non_demo_samples.p_demo, bins, alpha=0.5, label='nondemo')
        plt.legend(loc='upper right')
        logger.log_images([fig2img(fig)], "p_demo_hist", step, phase)
        plt.close(fig)

        # plot 2D map of states with color-coded demo probabilities
        fig = plt.figure()
        plt.scatter(np.concatenate((demo_samples.states[:, 0], non_demo_samples.states[:, 0])),
                    np.concatenate((demo_samples.states[:, 1], non_demo_samples.states[:, 1])), s=5,
                    c=np.concatenate((demo_samples.p_demo, non_demo_samples.p_demo)), cmap='RdYlGn')
        plt.axis("equal")
        logger.log_images([fig2img(fig)], "maze_p_demo_vis", step, phase)
        plt.close(fig)
