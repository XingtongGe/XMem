from .basics import *
# import pickle
# import os
# import codecs

class Bitparm(nn.Module):
    '''
    save params
    '''
    def __init__(self, channel, final=False):
        super(Bitparm, self).__init__()
        self.final = final
        self.h = nn.Parameter(torch.nn.init.normal_(torch.empty(channel).view(1, -1, 1, 1), 0, 0.01))
        self.b = nn.Parameter(torch.nn.init.normal_(torch.empty(channel).view(1, -1, 1, 1), 0, 0.01))
        if not final:
            self.a = nn.Parameter(torch.nn.init.normal_(torch.empty(channel).view(1, -1, 1, 1), 0, 0.01))
        else:
            self.a = None

    def forward(self, x):
        if self.final:
            return torch.sigmoid(x * F.softplus(self.h) + self.b)
        else:
            x = x * F.softplus(self.h) + self.b
            return x + torch.tanh(x) * torch.tanh(self.a)

class ICLR17BitEstimator(nn.Module):
    '''
    Estimate bit used in ICLR17, directly predict prob
    '''
    def __init__(self, channel):
        super(ICLR17BitEstimator, self).__init__()
        self.f1 = Bitparm(channel)
        self.f2 = Bitparm(channel)
        self.f3 = Bitparm(channel)
        self.f4 = Bitparm(channel, True)

    def calprob(self, x):
        x = self.f1(x)
        x = self.f2(x)
        x = self.f3(x)
        x = self.f4(x)
        return x

    def forward(self, x):
        prob = self.calprob(x + 0.5) - self.calprob(x - 0.5)
        total_bits = torch.sum(torch.clamp(-1.0 * torch.log(prob + 1e-5) / math.log(2.0), 0, 50))
        return total_bits, prob

class ICLR18BitEstimator(nn.Module):
    '''
    Estimate bit used in ICLR18, use sigma to predict prob
    '''
    def __init__(self):
        super(ICLR18BitEstimator, self).__init__()

    def forward(self, x, sigma):
        mu = torch.zeros_like(sigma)
        sigma = sigma.clamp(1e-5, 1e10)
        gaussian = torch.distributions.laplace.Laplace(mu, sigma)
        probs = gaussian.cdf(x + 0.5) - gaussian.cdf(x - 0.5)
        total_bits = torch.sum(torch.clamp(-1.0 * torch.log(probs + 1e-5) / math.log(2.0), 0, 50))
        return total_bits, probs

class Parameter_net(nn.Module):
    def __init__(self):
        super(Parameter_net, self).__init__()
        self.conv1 = nn.Conv2d(256, 512, 1, stride=1, padding=0)
        self.relu1 = nn.LeakyReLU(negative_slope=0.1)
        self.conv2 = nn.Conv2d(512, 512, 1, stride=1, padding=0)
        self.relu2 = nn.LeakyReLU(negative_slope=0.1)
        self.res0 = Resblocks(512, 1)
        self.conv3 = nn.Conv2d(512, 256, 1, stride=1, padding=0)


    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.res0(x)
        return self.conv3(x)


class NIPS18nocBitEstimator(nn.Module):
    '''
    Estimate bit used in NIPS18 without context, use mean and sigma to predict prob
    '''
    def __init__(self):
        super(NIPS18nocBitEstimator, self).__init__()
        self.parameters_net = Parameter_net()

    def forward(self, x, musigma):
        musigma = self.parameters_net(musigma)
        channel = x.shape[1]
        mu = musigma[:, 0:channel, :, :]
        sigma = musigma[:, channel:, :, :]
        sigma = sigma.pow(2)
        sigma = sigma.clamp(1e-5, 1e10)
        gaussian = torch.distributions.normal.Normal(mu, sigma)
        probs = gaussian.cdf(x + 0.5) - gaussian.cdf(x - 0.5)
        total_bits = torch.sum(torch.clamp(-1.0 * torch.log(probs + 1e-5) / math.log(2.0), 0, 50))
        return total_bits, probs


class Context_prediction_net(nn.Module):
    '''
    Compress residual prior
    '''
    def __init__(self):
        super(Context_prediction_net, self).__init__()
        self.conv1 = MaskedConvolution2D(out_channel_resM, 256, 5, stride=1, padding=2)


    def forward(self, x):
        x = self.conv1(x)
        return x

class Entropy_parameter_net(nn.Module):
    '''
    Compress residual prior
    '''
    def __init__(self):
        super(Entropy_parameter_net, self).__init__()
        self.conv1 = nn.Conv2d(512, 384, 1, stride=1, padding=0)
        self.relu1 = nn.LeakyReLU(negative_slope=0.1)
        self.conv2 = nn.Conv2d(384, 384, 1, stride=1, padding=0)
        self.relu2 = nn.LeakyReLU(negative_slope=0.1)
        self.conv3 = nn.Conv2d(384, 256, 1, stride=1, padding=0)


    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        return self.conv3(x)

class NIPS18BitEstimator(nn.Module):
    '''
    Estimate bit used in NIPS18 without context, use mean and sigma to predict prob
    '''
    def __init__(self):
        super(NIPS18BitEstimator, self).__init__()
        self.context_model = Context_prediction_net()
        self.entropy_model = Entropy_parameter_net()

    def forward(self, x, musigma):
        musigma = self.entropy_model(torch.cat((self.context_model(x), musigma), 1))
        channel = x.shape[1]
        mu = musigma[:, 0:channel, :, :]
        sigma = musigma[:, channel:, :, :]
        sigma = sigma.pow(2)
        sigma = sigma.clamp(1e-5, 1e10)
        gaussian = torch.distributions.normal.Normal(mu, sigma)
        probs = gaussian.cdf(x + 0.5) - gaussian.cdf(x - 0.5)
        total_bits = torch.sum(torch.clamp(-1.0 * torch.log(probs + 1e-5) / math.log(2.0), 0, 50))
        return total_bits, probs
