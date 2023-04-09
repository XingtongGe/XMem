from yolox.models.darknet import *

class featureDarknet(CSPDarknet):
    def __init__(self, dep_mul, wid_mul, out_features=("dark3", "dark4", "dark5"), depthwise=False, act="silu"):
        super().__init__(dep_mul, wid_mul, out_features, depthwise, act)
    
    def forward(self, x):
        # 去掉stem操作即可
        outputs = {}
        # x = self.stem(x)
        # print(x.shape)
        outputs["stem"] = x
        x = self.dark2(x)
        outputs["dark2"] = x
        x = self.dark3(x)
        outputs["dark3"] = x
        x = self.dark4(x)
        outputs["dark4"] = x
        x = self.dark5(x)
        outputs["dark5"] = x
        return {k: v for k, v in outputs.items() if k in self.out_features}
        # return super().forward(x)

