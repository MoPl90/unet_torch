from .components import *


class UNet(nn.Module):
    """Create a Unet-based generator"""

    def __init__(self, channels=1, classes=2, depth=4, filters=64, norm='batchnorm', dim=3, dropout=0.5, **kwargs):
        """Construct a Unet generator
        Parameters:
            channels (int)  -- the number of channels in input images
            classes (int)   -- the number of channels in output images
            depth (int)     -- network depth
            filters (int)   -- the number of filters in the last conv layer
            norm_layer      -- normalization layer
        We construct the U-Net from the innermost layer to the outermost layer.
        It is a recursive process.
        """

        super().__init__()
        # construct unet structure
        unet_block = UnetSkipConnectionBlock(filters * 8, filters * 8, input_nc=None, submodule=None, norm=norm, dim=dim, innermost=True)  # add the innermost layer
        factor = 4
        for i in range(depth-1):
            if factor > 0:
                unet_block = UnetSkipConnectionBlock(filters * (factor), filters * (factor * 2), input_nc=None, submodule=unet_block, norm=norm, dim=dim)
            factor = factor // 2

        # unet_block = UnetSkipConnectionBlock(filters * 2, filters * 4, input_nc=None, submodule=unet_block, norm=norm, dim=dim)
        # unet_block = UnetSkipConnectionBlock(filters, filters * 2, input_nc=None, submodule=unet_block, norm=norm, dim=dim)
        self.model = UnetSkipConnectionBlock(classes, filters, input_nc=channels, submodule=unet_block, outermost=True, norm=norm, dim=dim)  # add the outermost layer

    def forward(self, input):
        """Standard forward"""
        return self.model(input)