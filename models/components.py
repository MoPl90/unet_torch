import torch
from torch import nn
from torch.nn.modules.batchnorm import BatchNorm2d
from torch.nn.modules.pooling import MaxPool2d
from collections import OrderedDict


class Conv2DBlock(nn.Module):
    def __init__(self, c_in, c_out, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels=c_in, out_channels=c_out, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=False, padding_mode='zeros'),   
            nn.BatchNorm2d(c_in),
            nn.ReLU,
            nn.Dropout(dropout),
            nn.Conv2d(in_channels=c_out, out_channels=c_out, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=False, padding_mode='zeros'),   
            nn.BatchNorm2d(c_out),
            nn.ReLU,
            nn.Dropout(dropout),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

    def forward(self, x):
        return self.net(x) 

class Conv3DBlock(nn.Module):
    def __init__(self, c_in, c_out, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(in_channels=c_in, out_channels=c_out, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=False, padding_mode='zeros'),   
            nn.BatchNorm3d(c_in),
            nn.ReLU,
            nn.Dropout(dropout),
            nn.Conv3d(in_channels=c_out, out_channels=c_out, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=False, padding_mode='zeros'),   
            nn.BatchNorm3d(c_out),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.MaxPool3d(kernel_size=2, stride=2)
        )

    def forward(self, x):
        return self.net(x)


class Deconv2DBlock(nn.Module):
    def __init__(self, c_in, c_out, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels=c_in, out_channels=c_in, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=False, padding_mode='zeros'),   
            nn.BatchNorm2d(c_out),
            nn.ReLU,
            nn.Dropout(dropout),
            nn.Conv2d(in_channels=c_in, out_channels=c_in, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=False, padding_mode='zeros'),   
            nn.BatchNorm2d(c_out),
            nn.Dropout(dropout),
            nn.ConvTranspose2d(in_channels=c_in, out_channels=c_out, kernel_size=2, stride=2, padding=1, dilation=1, groups=1, bias=False, padding_mode='zeros'),   
            nn.BatchNorm2d(c_in),
            nn.ReLU,
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x) 

class Deconv3DBlock(nn.Module):
    def __init__(self, c_in, c_out, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(in_channels=c_in, out_channels=c_in, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=False, padding_mode='zeros'),   
            nn.BatchNorm3d(c_in),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv3d(in_channels=c_in, channels=c_in, kernel_size=3, stride=1, padding=1, dilation=1, groups=1, bias=False, padding_mode='zeros'),   
            nn.BatchNorm3d(c_out),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.ConvTranspose3d(in_channels=c_in, out_channels=c_out, kernel_size=2, stride=2, padding=1, dilation=1, groups=1, bias=False, padding_mode='zeros'),   
            nn.BatchNorm3d(c_in),
            nn.ReLU,
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Encoder(nn.Module):
    def __init__(self, dim, c_in, depth, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])

        #start with 1-channel image and initial channel dimension
        c_out = c_in
        c_in = 1
        for _ in range(depth):
            if dim == 2:
               self.layers.append(Conv2DBlock(c_in=c_in, c_out=c_out, dropout=dropout))
            elif dim == 3:
               self.layers.append(Conv3DBlock(c_in=c_in, c_out=c_out, dropout=dropout))
            else: 
                raise ValueError("Dimesion must be 2D or 3D")
            c_in = c_out
            c_out *= 2


    def forward(self, x):
        skip = []
        for l in self.layers:
            x = l[0](x)
            skip.append(x)

        return x, skip[:-1]

class Decoder(nn.Module):
    def __init__(self, dim, c_in, skip_layers, depth, c_out, dropout = 0.):
        
        assert depth == skip_layers, "Number of skip connections needs to match network depth!"

        super().__init__()
        self.layers = nn.ModuleList([])

        # for i in range(depth):

    def forward(self, x):
        return self.net(x)


class UnetSkipConnectionBlock(nn.Module):
    """Defines the Unet submodule with skip connection.
        X -------------------identity----------------------
        |-- downsampling -- |submodule| -- upsampling --|
    """

    def __init__(self, outer_nc, inner_nc, input_nc=None, dim=3,
                 submodule=None, outermost=False, innermost=False, 
                 norm='batch_norm', use_dropout=False):
        """Construct a Unet submodule with skip connections.
        Parameters:
            outer_nc (int) -- the number of filters in the outer conv layer
            inner_nc (int) -- the number of filters in the inner conv layer
            input_nc (int) -- the number of channels in input images/features
            dim (int) -- spatial data dimension
            submodule (UnetSkipConnectionBlock) -- previously defined submodules
            outermost (bool)    -- if this module is the outermost module
            innermost (bool)    -- if this module is the innermost module
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
        """
        super().__init__()
        self.outermost = outermost
        
        if dim == 2:
            if 'batch' in norm.lower():
                norm_layer = nn.BatchNorm2d
            elif 'instance' in norm.lower():
                norm_layer = nn.InstanceNorm2d
            else:
                raise ValueError("Norm layer '" + norm + "' unknown.")
        elif dim == 3:
            if 'batch' in norm.lower():
                norm_layer = nn.BatchNorm3d
            elif 'instance' in norm.lower():
                norm_layer = nn.InstanceNorm3d
            else:
                raise ValueError("Norm layer '" + norm + "' unknown.")
        else:
            raise ValueError("Dimension must be 2 or 3.")

        use_bias = 'instance' in norm.lower() 
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=3,
                             stride=1, padding=1, bias=use_bias) if dim == 2 else \
                   nn.Conv3d(input_nc, inner_nc, kernel_size=3,
                             stride=1, padding=1, bias=use_bias)

        downrelu = nn.LeakyReLU(0.2, inplace=True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(inplace=True)
        upnorm = norm_layer(outer_nc)
        downsample = nn.MaxPool2d(kernel_size=2) if dim == 2 else \
                     nn.MaxPool3d(kernel_size=2) 

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1) if dim == 2 else \
                     nn.ConvTranspose3d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            # down = [downconv, downrelu, downconv, downsample]
            # up = [uprelu, upconv, nn.Softmax(dim=1)]
            down = [self._block(input_nc, inner_nc, 'outer_block', norm_layer, downrelu, dim), downsample]
            up = [upconv, self._block(outer_nc, outer_nc, 'outer_block', norm_layer, nn.Softmax(dim=1), dim)]

            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias) if dim == 2 else \
                     nn.ConvTranspose3d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            # down = [downrelu, downconv, downsample]
            # up = [uprelu, upconv, upnorm]
            down = [self._block(input_nc, inner_nc, 'inner_block1', norm_layer, downrelu, dim)]
            up = [self._block(inner_nc, outer_nc, 'inner_block2', norm_layer, downrelu, dim)]

            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)  if dim == 2 else \
                     nn.ConvTranspose3d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            # down = [downrelu, downconv, downnorm,  downsample]
            # up = [uprelu, upconv, upnorm]

            down = [self._block(input_nc, inner_nc, 'inner_block', norm_layer, downrelu, dim), downsample]
            up = [upconv, self._block(outer_nc, outer_nc, 'outer_block', norm_layer, downrelu, dim)]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):

        if self.outermost:
            return self.model(x)
        else:   # add skip connections
            return torch.cat([x, self.model(x)], 1)


    @staticmethod
    def _block(in_channels, features, name, norm_layer, act_layer, dim):
        if dim == 2:
            return nn.Sequential(
                OrderedDict(
                    [
                        (
                            name + "conv1",
                            nn.Conv2d(
                                in_channels=in_channels,
                                out_channels=features,
                                kernel_size=3,
                                padding=1,
                                bias=False,
                            ),
                        ),
                        (name + "norm1", norm_layer(num_features=features)),
                        (name + "relu1", act_layer),
                        (
                            name + "conv2",
                            nn.Conv2d(
                                in_channels=features,
                                out_channels=features,
                                kernel_size=3,
                                padding=1,
                                bias=False,
                            ),
                        ),
                        (name + "norm2", norm_layer(num_features=features)),
                        (name + "relu2", act_layer),
                    ]
                )
            )
        else:
            return nn.Sequential(
                OrderedDict(
                    [
                        (
                            name + "conv1",
                            nn.Conv3d(
                                in_channels=in_channels,
                                out_channels=features,
                                kernel_size=3,
                                padding=1,
                                bias=False,
                            ),
                        ),
                        (name + "norm1", norm_layer(num_features=features)),
                        (name + "relu1", act_layer),
                        (
                            name + "conv2",
                            nn.Conv3d(
                                in_channels=features,
                                out_channels=features,
                                kernel_size=3,
                                padding=1,
                                bias=False,
                            ),
                        ),
                        (name + "norm2", norm_layer(num_features=features)),
                        (name + "relu2", act_layer),
                    ]
                )
            )