
import torch
import torch.nn as nn
# -------------------------- MetaModule & VNet --------------------------
class MetaModule(nn.Module):
    """基础 meta-learning 模块，可递归更新参数"""
    def params(self):
        for name, param in self.named_params():
            yield param

    def named_leaves(self):
        return []

    def named_submodules(self):
        return []

    def named_params(self, curr_module=None, memo=None, prefix=''):
        if memo is None:
            memo = set()
        if curr_module is None:
            curr_module = self

        if hasattr(curr_module, 'named_leaves'):
            for name, p in curr_module.named_leaves():
                if p is not None and p not in memo:
                    memo.add(p)
                    yield prefix + ('.' if prefix else '') + name, p
        else:
            for name, p in curr_module._parameters.items():
                if p is not None and p not in memo:
                    memo.add(p)
                    yield prefix + ('.' if prefix else '') + name, p

        for mname, module in curr_module.named_children():
            submodule_prefix = prefix + ('.' if prefix else '') + mname
            for name, p in self.named_params(module, memo, submodule_prefix):
                yield name, p


class MetaLinear(MetaModule):
    def __init__(self, input_size, output_size, bias=True):
        super(MetaLinear, self).__init__()
        self.linear = nn.Linear(input_size, output_size, bias=bias)

    def forward(self, x):
        return self.linear(x)


class VNet(MetaModule):
    """简单三层全连接网络，用于生成样本权重"""
    def __init__(self, input=1, hidden1=100, hidden2=None, output=1, num_classes=1):
        super(VNet, self).__init__()
        self.input = input
        self.output = output
        if hidden2 is None:
            hidden2 = hidden1
        self.fc1 = MetaLinear(input, hidden1)
        self.relu1 = nn.ReLU(True)
        if hidden2 > 0:
            self.fc2 = MetaLinear(hidden1, hidden2)
            self.relu2 = nn.ReLU(True)
            self.fc3 = MetaLinear(hidden2, output)
        else:
            self.fc2 = None
            self.relu2 = None
            self.fc3 = MetaLinear(hidden1, output)

    def forward(self, x, num=None, c=None):
        out = self.fc1(x)
        out = self.relu1(out)
        if self.fc2 is not None:
            out = self.fc2(out)
            out = self.relu2(out)
        out = self.fc3(out)
        # 输出权重限制在 [0,1]
        out = torch.sigmoid(out)
        return out