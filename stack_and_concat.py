import torch
t1 = torch.tensor([1, 1, 1]) #一维
t1.unsqueeze(dim=0)

t1.unsqueeze(dim=1)

print(t1.shape)
print(t1.unsqueeze(dim=0).shape) # torch.Size([1, 3])
print(t1.unsqueeze(dim=1).shape) # torch.Size([3, 1])

# concatenating joins a sequence of tensors along an eisting axis, concatenating沿着一个已经存在的轴将一系列tensors组合在一起
# and stacking joins a sequence of tensors along a new axis. stacking 沿着一个新轴组合一系列的tensor
t1 = torch.tensor([1, 1, 1])
t2 = torch.tensor([2, 2, 2])
t3 = torch.tensor([3, 3, 3])

cat_tensor = torch.cat(
    (t1, t2, t3),
    dim=0
)

"""
cat_tensor_dim1 = torch.cat(
    (t1, t2, t3),
    dim=1
)
提示dimension out of range(expected to be in range of [-1, 0], but got 1)

"""
stack_tensor = torch.stack(
    (t1, t2, t3),
    dim=0
)

stack_tensor_dim1 = torch.stack(
    (t1, t2, t3),
    dim=1
)

#print(cat_tensor_dim1)
print(stack_tensor)
print(stack_tensor_dim1)

cat_unsqueezed_tensor = torch.cat(
    (t1.unsqueeze(0),
    t2.unsqueeze(0),
    t3.unsqueeze(0)),
    dim=0
)

print(cat_unsqueezed_tensor) # 这个效果与stack一样了。但是这种方法明显更麻烦，stack简单的多

#从上面我们可以看到stack_tensor_dim1的结果与stack的结果是一样的。如何理解这个结果呢：
cat_unsqueezed_dim1 = torch.cat(
    (
        t1.unsqueeze(1),
        t2.unsqueeze(1),
        t3.unsqueeze(1)
    ),
    dim=1
)

print(t1.unsqueeze(1))
print(cat_unsqueezed_dim1) # 可以把torch.stack((t1, t2, t3), dim=1)的结果看成是t1, t2, t3 unsqueezed之后，沿着dim=1拼接

#tensorflow: stack and concat
import tensorflow as tf
t1 = tf.constant([1, 1, 1])
t2 = tf.constant([2, 2, 2])
t3 = tf.constant([3, 3, 3])
print(tf.concat(
    (t1, t2, t3),
    axis=0
))

print(tf.stack(
    (t1, t2, t3),
    axis=0
))

print(tf.concat(
    (
        tf.expand_dims(t1, 0)
        ,tf.expand_dims(t2, 0)
        ,tf.expand_dims(t3, 0)
    ),
    axis=0
))

print(tf.stack(
    (t1, t2, t3),
    axis=1
))

print(tf.concat(
    (
        tf.expand_dims(t1, 1),
        tf.expand_dims(t2, 1),
        tf.expand_dims(t3, 1)
    ),
    axis=1
))

#从上面可以看出tf.stack和tf.concat与pytorch的stack和cat的功能是完全一样的


# numpy： stack vs concatenate
import numpy as np

t1 = np.array([1, 1, 1])
t2 = np.array([2, 2, 2])
t3 = np.array([3, 3, 3])

print(np.concatenate(
    (t1, t2, t3),
    axis=0
))

print(np.stack(
    (t1, t2, t3),
    axis=0
))

print(np.concatenate(
    (np.expand_dims(t1, 0),
     np.expand_dims(t2, 0),
     np.expand_dims(t3, 0)),
    axis=0
))

print(np.stack(
    (t1, t2, t3),
    axis=1
))

print(np.concatenate(
    (
        np.expand_dims(t1, 1),
        np.expand_dims(t2, 1),
        np.expand_dims(t3, 1)
    ),
    axis=1
))