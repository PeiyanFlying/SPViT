
def gumbel_softmax(logits: Tensor, tau: float = 1, hard: bool = False, eps: float = 1e-10, dim: int = -1) -> Tensor:
    r"""
    Samples from the Gumbel-Softmax distribution (`Link 1`_  `Link 2`_) and optionally discretizes.
    Args:
      logits: `[..., num_features]` unnormalized log probabilities
      tau: non-negative scalar temperature
      hard: if ``True``, the returned samples will be discretized as one-hot vectors,
            but will be differentiated as if it is the soft sample in autograd
      dim (int): A dimension along which softmax will be computed. Default: -1.
    Returns:
      Sampled tensor of same shape as `logits` from the Gumbel-Softmax distribution.
      If ``hard=True``, the returned samples will be one-hot, otherwise they will
      be probability distributions that sum to 1 across `dim`.
    .. note::
      This function is here for legacy reasons, may be removed from nn.Functional in the future.
    .. note::
      The main trick for `hard` is to do  `y_hard - y_soft.detach() + y_soft`
      It achieves two things:
      - makes the output value exactly one-hot
      (since we add then subtract y_soft value)
      - makes the gradient equal to y_soft gradient
      (since we strip all other gradients)
    Examples::
        >>> logits = torch.randn(20, 32)
        >>> # Sample soft categorical using reparametrization trick:
        >>> F.gumbel_softmax(logits, tau=1, hard=False)
        >>> # Sample hard categorical using "Straight-through" trick:
        >>> F.gumbel_softmax(logits, tau=1, hard=True)
    .. _Link 1:
        https://arxiv.org/abs/1611.00712
    .. _Link 2:
        https://arxiv.org/abs/1611.01144
    """
    if has_torch_function_unary(logits):
        return handle_torch_function(gumbel_softmax, (logits,), logits, tau=tau, hard=hard, eps=eps, dim=dim)
    if eps != 1e-10:
        warnings.warn("`eps` parameter is deprecated and has no effect.")

    gumbels = (
        -torch.empty_like(logits, memory_format=torch.legacy_contiguous_format).exponential_().log()
    )  # ~Gumbel(0,1)
    gumbels = (logits + gumbels) / tau  # ~Gumbel(logits,tau)
    y_soft = gumbels.softmax(dim)

    if hard:
        # Straight through.
        index = y_soft.max(dim, keepdim=True)[1]
        y_hard = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format).scatter_(dim, index, 1.0)
        ret = y_hard - y_soft.detach() + y_soft
    else:
        # Reparametrization trick.
        ret = y_soft
    return ret

#hard=True 的时候返回的就是one-hot向量。其中y_soft 是采样出来的概率分布，
#y_hard是根据这个概率分布得到求出来的one-hot向量，detach()这个方法实际上是把一个张量移除计算图变成常量，
#这样反向传播的时候就不会计算它的梯度。所以这个东西：  ret = y_hard - y_soft.detach() + y_soft
#就是构造了一个数值上等于one-hot向量的张量，但实际上反向传播的时候梯度是顺着y_soft传回去的。


if i in self.pruning_loc: 
  spatial_x = x[:, 1:] 
  pred_score = self.score_predictor[p_count](spatial_x, prev_decision).reshape(B, -1, 2)#pai, two columns,keep score and remove score, all fu shu 
  print('pred_score',pred_score)# Fu shu 
  hard_keep_decision = F.gumbel_softmax(pred_score, hard=False)#[:, :, 0:1] * prev_decision #remove this because use after the whole gumbel operation
  print('hard_keep_decision',hard_keep_decision) #after gumble, become zheng shu, two columns sum =1, stands for keep remove probability 
  index_test = hard_keep_decision.max(-1,keepdim=True)[1] #2 columns, which column is larger, output tensor and index, so only need the second which is [1]--> index #
  print('index_test',index_test.shape) #
  print(index_test) 
  y_hard = torch.zeros_like(pred_score, memory_format=torch.legacy_contiguous_format).scatter_(-1, index_test, 1.0) 
  ret = y_hard - hard_keep_decision.detach() + hard_keep_decision 
  print('ret',ret[:, :, 0:1]) #prob of keep 
  exit()










