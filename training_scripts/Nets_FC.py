import torch.nn as nn
import torch.nn.functional as F

# activation function
act_f = nn.ELU()


class Encoder_2D(nn.Module):

    def __init__(self, channels_list, ksize, strides_list):

        super(Encoder_2D, self).__init__()
        n_layers = len(channels_list) - 1

        # set up convolutional layers
        self.f_conv = nn.ModuleList(
            [
                nn.Conv2d(
                    channels_list[i],
                    channels_list[i + 1],
                    kernel_size=ksize[i],
                    stride = strides_list[i],
                    padding="same",
                )
                for i in range(n_layers)
            ]
        )
        for conv_i in self.f_conv:
            nn.init.xavier_uniform_(conv_i.weight)

        # setting batch normalization layers
        self.conv2_bn = nn.ModuleList(
            [nn.BatchNorm2d(channels_list[i + 1]) for i in range(n_layers)]
        )

    def forward(self, x):

        for i, conv_i in enumerate(self.f_conv):
            x = conv_i(x)
            x = self.conv2_bn[i](x)
            x = act_f(x)
        return x


class Decoder_2D(nn.Module):

    def __init__(
        self, channels_list, ksize, strides_list
    ): 

        super(Decoder_2D, self).__init__()

        n_layers = len(channels_list) - 1

        def compute_padding(k):
            return k // 2 if isinstance(k, int) else tuple(kk // 2 for kk in k)
        
        self.f_conv = nn.ModuleList(
            [
                nn.ConvTranspose2d(
                    channels_list[i],
                    channels_list[i + 1],
                    kernel_size=ksize[i],
                    stride = strides_list[i],
                    #padding=ksize[i]//2,
                    #   output_padding = output_padding_list[i]
                    padding=compute_padding(ksize[i])
                )
                for i in range(n_layers)
            ]
        )

        for conv_i in self.f_conv:
            nn.init.xavier_uniform_(conv_i.weight)

        self.conv2_bn = nn.ModuleList(
            [nn.BatchNorm2d(channels_list[i + 1]) for i in range(n_layers)]
        )

    def forward(self, x):

        for i, conv_i in enumerate(self.f_conv[:-1]):
            x = conv_i(x)
            x = self.conv2_bn[i](x)
            x = act_f(x)
        # not apply batch normalization or activation function to the last layer
        x = self.f_conv[-1](x)

        return x


class AutoEncoder_2D(nn.Module):

    def __init__(self, channels_list, ksize, strides_list):

        super(AutoEncoder_2D, self).__init__()

        self.encoder = Encoder_2D(
            channels_list=channels_list, ksize=ksize, strides_list=strides_list
        )

        self.decoder = Decoder_2D(
            channels_list=channels_list[::-1],
            ksize=ksize[::-1],
            strides_list=strides_list[::-1],
        )

    def forward(self, x):

        return self.decoder(self.encoder(x))

    def get_latent_space_coordinates(self, x):
        """
        Returns latent vector of shape [B, C], regardless of input size.
        """
        encoded = self.encoder(x)  # shape: [B, C, H, W]
        pooled = F.adaptive_avg_pool2d(encoded, 1)  # shape: [B, C, 1, 1]
        latent = pooled.view(pooled.size(0), -1)  # shape: [B, C]
        return latent
