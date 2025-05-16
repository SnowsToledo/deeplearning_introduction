import torch


def sum_tensor(a, b):
    return a + b


def set_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def reshape_tensor(t):
    print(t.shape)
    print(t.ndim)


def broadcast_tensor(a, b):
    return a + b


def autograd_tensor():
    x = torch.tensor(2.0, requires_grad=True)
    y = x ** 4
    y.backward()
    print(x.grad)


if __name__ == '__main__':
    a = torch.tensor([1.0, 2.0])
    b = torch.tensor([3.0, 4.0])
    c = torch.tensor([[5.0, 6.0], [7.0, 8.0]])

    # Tensor a partir de lista
    t1 = torch.tensor([1.0, 2.0, 3.0])

    # Tensor com valores aleatórios
    t2 = torch.randn(2, 3, 4, 5)  # 2 linhas, 3 colunas
    # print(t2)
    # Tensor com todos os zeros
    z = torch.zeros(4, 4)
    print(z)
    # Tensor de uns
    o = torch.ones(2, 2)
    print(o)
    # print(a + c)  # Soma
    # print(a * b)  # Multiplicação, elemento a elemento
    # print(a @ b)  # Produto escalar
    #
    # print(sum_tensor(a, b))
    print(set_device())
    # reshape_tensor(a)
    # print(broadcast_tensor(a, c))
    autograd_tensor()
