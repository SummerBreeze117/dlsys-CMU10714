# dlsys-CMU10714（部分笔记）

## homework2

### weight initialization

**Kaiming uniform/normal**

`kaiming_uniform(fan_in, fan_out, nonlinearity="relu", **kwargs)`

Tensor从均匀分布 $U(-bound, bound)$中采样，其中
	$$bound = \text{gain} \times \sqrt{\frac{3}{\text{fan}_{in}}}$$
`kaiming_normal(fan_in, fan_out, nonlinearity="relu", **kwargs)`

Tensor从高斯分布 $N(0, std^2)$中采样，其中
$$std = \frac{\text{gain}}{\sqrt{\text{fan}_{in}}}$$
`ReLU` 的推荐使用缩放因子：$gain=\sqrt{2}$

- `fan_in` - 输入维度
- `fan_out` - 输出维度
- `gain` - 可选的缩放因子

### nn
#### Linear

`needle.nn.Linear(in_features, out_features, bias=True, device=None, dtype="float32")`

对于输入数据进行线性变换: $y = xA^T + b$. 输入形状是 $(N, H_{in})$. 其中 $H_{in}=\text{in\_features}$. 输出形状是$(N, H_{out})$，其中$H_{out}=\text{out\_features}$. $N$是批次数据个数

> 注意使用broadcast的显式调用来修饰偏置项的形状，也就是从$(1, out\_features)$到$(N, out\_features)$. Needle不支持broadcast的隐式调用.
> 你需要先初始化权重项，再初始化偏置项

Parameters
- `in_features` - size of each input sample
- `out_features` - size of each output sample
- `bias` - If set to `False`, the layer will not learn an additive bias.
Variables
- `weight` - the learnable weights of shape (`in_features`, `out_features`). The values should be initialized with the Kaiming Uniform initialization with `fan_in = in_features`
- `bias` - the learnable bias of shape (`out_features`). The values should be initialized with the Kaiming Uniform initialize with `fan_in = out_features`. 
**注意这里我们要得到$(1, out\_features)$的bias向量，但是只有$fan\_in$才能作为初始化参数，所以我们要通过初始化$(out\_features,1)$的方式并且加上转置操作来得到目标向量。**
```python
self.bias = Parameter(init.kaiming_uniform(out_features, 1).transpose()) if bias else None
```

#### Sequential

`needle.nn.Sequential(*modules)`

将一系列模块应用于输入（按照它们传递给构造函数的顺序）并返回最后一个模块的输出。

### LogSumExp

`needle.ops.LogSumExp(axes)`

通过减去最大元素，将数值稳定的 log-sum-exp 函数应用于输入。
$$\text{LogSumExp}(z) = \log (\sum_{i} \exp (z_i - \max{z})) + \max{z}$$
#### **前向过程**

**如果`axes=(1,)`，则表示按行求最大值，即遍历游标在第一个维度变化.**

将输入Z按`axes`维度求得最大值，即

```python
z_max_origindim = array_api.max(Z, self.axes, keepdims=True)
z_max_reducedim = array_api.max(Z, self.axes)
```

> 注意这里两个max数组的维度不相同，第一个是保留维度的（这里隐含广播语义，也即是将每行求得的最大值广播到整行），用于跟原数组相减.

将$\exp (z_i - \max{z})$按`axes`维度进行求和，即

```python
array_api.sum(array_api.exp(Z - z_max_origindim), self.axes)
```

最后求得结果

```python
array_api.log(array_api.sum(array_api.exp(Z - z_max_origindim), self.axes)) + z_max_reducedim
```

#### **反向传播**

$$\frac{\partial \text{LogSumExp}(z)}{\partial z} =\frac{\exp (z_i - \max{z})}{\sum_{i} \exp (z_i - \max{z})}$$
这里我们求出相应的z的梯度值以后，还需要将维度变回之前的维度，因为前向过程是降低了维度的.

对于`axes=None`的情况，我们直接将结果广播回原维度即可.

```python
grad_sum_exp_z = out_grad / sum_exp_z
if self.axes is None:  
    return grad_sum_exp_z.broadcast_to(z.shape) * exp_z
```

对于`axes!=None`的情况，我们会将已经reduce过的维度reshape扩展回来，再广播回原维度.

```python
expand_shape = list(z.shape)  
for axis in self.axes:  
    expand_shape[axis] = 1  
grad_exp_z = grad_sum_exp_z.reshape(expand_shape).broadcast_to(z.shape)  
return grad_exp_z * exp_z
```