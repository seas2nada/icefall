import torch

def OrthogonalMP_REG_Parallel(A, b, tol=1E-4, nnz=None, positive=False, lam=1, device="cpu"):
    '''approximately solves min_x |x|_0 s.t. Ax=b using Orthogonal Matching Pursuit
    Args:
      A: design matrix of size (d, n)
      b: measurement vector of length d
      tol: solver tolerance
      nnz = maximum number of nonzero coefficients (if None set to n)
      positive: only allow positive nonzero coefficients
    Returns:
       vector of length n
    '''
    AT = torch.transpose(A, 0, 1)
    d, n = A.shape
    if nnz is None:
        nnz = n
    x = torch.zeros(n, device=device)  # ,dtype=torch.float64)
    resid = b.detach().clone()
    normb = b.norm().item()
    indices = []
    argmin = torch.tensor([-1])
    for i in range(nnz):
        if resid.norm().item() / normb < tol:
            break
        projections = torch.matmul(AT, resid)  # AT.dot(resid)
        # print("Projections",projections.shape)
        if positive:
            index = torch.argmax(projections)
        else:
            index = torch.argmax(torch.abs(projections))
        if index in indices:
            break
        indices.append(index)
        if len(indices) == 1:
            A_i = A[:, index]
            x_i = projections[index] / torch.dot(A_i, A_i).view(-1)  # A_i.T.dot(A_i)
            A_i = A[:, index].view(1, -1)
        else:
            # print(indices)
            A_i = torch.cat((A_i, A[:, index].view(1, -1)), dim=0)  # np.vstack([A_i, A[:,index]])
            temp = torch.matmul(A_i, torch.transpose(A_i, 0, 1)) + lam * torch.eye(A_i.shape[0], device=device)
            x_i, _, _, _ = torch.linalg.lstsq(temp, torch.matmul(A_i, b).view(-1, 1))
            # print(x_i.shape)
            if positive:

                while min(x_i) < 0.0:
                    # print("Negative",b.shape,torch.transpose(A_i, 0, 1).shape,x_i.shape)
                    argmin = torch.argmin(x_i)
                    indices = indices[:argmin] + indices[argmin + 1:]
                    A_i = torch.cat((A_i[:argmin], A_i[argmin + 1:]),
                                    dim=0)  # np.vstack([A_i[:argmin], A_i[argmin+1:]])
                    if argmin.item() == A_i.shape[0]:
                        break
                    # print(argmin.item(),A_i.shape[0],index.item())
                    temp = torch.matmul(A_i, torch.transpose(A_i, 0, 1)) + lam * torch.eye(A_i.shape[0], device=device)
                    x_i, _, _, _ = torch.linalg.lstsq(temp, torch.matmul(A_i, b).view(-1, 1))
        if argmin.item() == A_i.shape[0]:
            break
        # print(b.shape,torch.transpose(A_i, 0, 1).shape,x_i.shape,\
        #  torch.matmul(torch.transpose(A_i, 0, 1), x_i).shape)
        resid = b - torch.matmul(torch.transpose(A_i, 0, 1), x_i).view(-1)  # A_i.T.dot(x_i)
        # print("REsID",resid.shape)

    x_i = x_i.view(-1)
    # print(x_i.shape)
    # print(len(indices))
    for i, index in enumerate(indices):
        # print(i,index,end="\t")
        try:
            x[index] += x_i[i]
        except IndexError:
            x[index] += x_i
    # print(x[indices])
    return x

class DataSelectionStrategy(object):

    def __init__(self, trainloader, valloader, model, tea_model, ssl_alg, num_classes, linear_layer, loss, device, logger):
        """
        Constructor method
        """
        self.trainloader = trainloader  # assume its a sequential loader.
        self.valloader = valloader
        self.model = model
        self.N_trn = len(trainloader.sampler)
        self.N_val = len(valloader.sampler)
        self.grads_per_elem = None
        self.val_grads_per_elem = None
        self.numSelected = 0
        self.linear_layer = linear_layer
        self.num_classes = num_classes
        self.trn_lbls = None
        self.val_lbls = None
        self.loss = loss
        self.device = device
        self.tea_model = tea_model
        self.ssl_alg = ssl_alg
        self.logger = logger

    def select(self, budget, model_params, tea_model_params):
        """
        Abstract select function that is overloaded by the child classes
        """
        pass

    def ssl_loss(self, ul_weak_data, ul_strong_data, labels=False):
        """
        Function that computes contrastive semi-supervised loss

        Parameters
        -----------
        ul_weak_data: 
            Weak agumented version of unlabeled data
        ul_strong_data:
            Strong agumented version of unlabeled data
        labels: bool
            if labels, just return hypothesized labels of the unlabeled data
        
        Returns
        --------
        L_consistency: Consistency loss
        y: Actual labels(Not used anywhere)
        l1_strong: Penultimate layer outputs for strongly augmented version of unlabeled data
        targets: Hypothesized labels
        mask: mask vector of the unlabeled data

        """
        self.logger.debug("SSL loss computation initiated")
        all_data = torch.cat([ul_weak_data, ul_strong_data], 0)
        forward_func = self.model.forward
        stu_logits, l1 = forward_func(all_data, last=True, freeze=True)
        stu_unlabeled_weak_logits, stu_unlabeled_strong_logits = torch.chunk(stu_logits, 2, dim=0)
        _, l1_strong = torch.chunk(l1, 2, dim=0)

        if self.tea_model is not None: # get target values from teacher model
            t_forward_func = self.tea_model.forward
            tea_logits = t_forward_func(all_data)
            tea_unlabeled_weak_logits, _ = torch.chunk(tea_logits, 2, dim=0)
        else:
            t_forward_func = forward_func
            tea_unlabeled_weak_logits = stu_unlabeled_weak_logits

        self.model.update_batch_stats(False)
        if self.ssl_alg.__class__.__name__ in ['PseudoLabel', 'ConsistencyRegularization']:
            y, targets, mask = self.ssl_alg(
                stu_preds=stu_unlabeled_strong_logits,
                tea_logits=tea_unlabeled_weak_logits.detach(),
                w_data=ul_strong_data,
                stu_forward=forward_func,
                tea_forward=t_forward_func
            )
        else:
            y, l1_strong, targets, mask = self.ssl_alg(
                stu_preds=stu_unlabeled_strong_logits,
                tea_logits=tea_unlabeled_weak_logits.detach(),
                w_data=ul_strong_data,
                subset=True,
                stu_forward=forward_func,
                tea_forward=t_forward_func
            )
        self.logger.debug("SSL loss computation finished")
        if labels:
            if targets.ndim == 1:
                return targets
            else:
                return targets.argmax(dim=1)
        else:
            self.model.update_batch_stats(True)
            #mask = torch.ones(len(mask), device=self.device)
            L_consistency = self.loss(y, targets, mask, weak_prediction=stu_unlabeled_weak_logits.softmax(1))
            return L_consistency, y, l1_strong, targets, mask

    def get_labels(self, valid=False):
        """
        Function that iterates over labeled or unlabeled data and returns target or hypothesized labels.

        Parameters
        -----------
        valid: bool
            If True, iterate over the labeled set
        """
        self.logger.debug("Get labels function Initiated")
        for batch_idx, (ul_weak_aug, ul_strong_aug, _) in enumerate(self.trainloader):
            ul_weak_aug, ul_strong_aug = ul_weak_aug.to(self.device), ul_strong_aug.to(self.device)
            targets = self.ssl_loss(ul_weak_data=ul_weak_aug, ul_strong_data=ul_strong_aug, labels=True)
            if batch_idx == 0:
                self.trn_lbls = targets.view(-1, 1)
            else:
                self.trn_lbls = torch.cat((self.trn_lbls, targets.view(-1, 1)), dim=0)
        self.trn_lbls = self.trn_lbls.view(-1)

        if valid:
            for batch_idx, (inputs, targets) in enumerate(self.valloader):
                if batch_idx == 0:
                    self.val_lbls = targets.view(-1, 1)
                else:
                    self.val_lbls = torch.cat((self.val_lbls, targets.view(-1, 1)), dim=0)
            self.val_lbls = self.val_lbls.view(-1)
        self.logger.debug("Get labels function finished")

    def compute_gradients(self, valid=False, perBatch=False, perClass=False, store_t=False):
        """
        Computes the gradient of each element.

        Here, the gradients are computed in a closed form using CrossEntropyLoss with reduction set to 'none'.
        This is done by calculating the gradients in last layer through addition of softmax layer.

        Using different loss functions, the way we calculate the gradients will change.

        For LogisticLoss we measure the Mean Absolute Error(MAE) between the pairs of observations.
        With reduction set to 'none', the loss is formulated as:

        .. math::
            \\ell(x, y) = L = \\{l_1,\\dots,l_N\\}^\\top, \\quad
            l_n = \\left| x_n - y_n \\right|,

        where :math:`N` is the batch size.


        For MSELoss, we measure the Mean Square Error(MSE) between the pairs of observations.
        With reduction set to 'none', the loss is formulated as:

        .. math::
            \\ell(x, y) = L = \\{l_1,\\dots,l_N\\}^\\top, \\quad
            l_n = \\left( x_n - y_n \\right)^2,

        where :math:`N` is the batch size.
        Parameters
        ----------
        valid: bool
            if True, the function also computes the validation gradients
        batch: bool
            if True, the function computes the gradients of each mini-batch
        perClass: bool
            if True, the function computes the gradients using perclass dataloaders
        store_t: bool
            if True, the function stores the hypothesized weak augmentation targets and masks for unlabeled set.
        """
        if (perBatch and perClass):
            raise ValueError("batch and perClass are mutually exclusive. Only one of them can be true at a time")

        self.logger.debug("Per-sampele gradient computation Initiated")
        embDim = self.model.get_embedding_dim()
        if perClass:
            trainloader = self.pctrainloader
            if valid:
                valloader = self.pcvalloader
        else:
            trainloader = self.trainloader
            if valid:
                valloader = self.valloader
        

        if store_t:
            targets = []
            masks = []

        for batch_idx, (ul_weak_aug, ul_strong_aug, _) in enumerate(trainloader):
            ul_weak_aug, ul_strong_aug = ul_weak_aug.to(self.device), ul_strong_aug.to(self.device)
            if store_t:
                loss, out, l1, t, m = self.ssl_loss(ul_weak_data=ul_weak_aug, ul_strong_data=ul_strong_aug)
                targets.append(t)
                masks.append(m)
            else:
                loss, out, l1, _, _ = self.ssl_loss(ul_weak_data=ul_weak_aug, ul_strong_data=ul_strong_aug)
            loss = loss.sum()
            if batch_idx == 0:
                l0_grads = torch.autograd.grad(loss, out)[0]
                if self.linear_layer:
                    l0_expand = torch.repeat_interleave(l0_grads, embDim, dim=1)
                    l1_grads = l0_expand * l1.repeat(1, self.num_classes)
                if perBatch:
                    l0_grads = l0_grads.mean(dim=0).view(1, -1)
                    if self.linear_layer:
                        l1_grads = l1_grads.mean(dim=0).view(1, -1)
            else:
                batch_l0_grads = torch.autograd.grad(loss, out)[0]
                if self.linear_layer:
                    batch_l0_expand = torch.repeat_interleave(batch_l0_grads, embDim, dim=1)
                    batch_l1_grads = batch_l0_expand * l1.repeat(1, self.num_classes)
                if perBatch:
                    batch_l0_grads = batch_l0_grads.mean(dim=0).view(1, -1)
                    if self.linear_layer:
                        batch_l1_grads = batch_l1_grads.mean(dim=0).view(1, -1)
                l0_grads = torch.cat((l0_grads, batch_l0_grads), dim=0)
                if self.linear_layer:
                    l1_grads = torch.cat((l1_grads, batch_l1_grads), dim=0)

        torch.cuda.empty_cache()
        if store_t:
            self.weak_targets = targets
            self.weak_masks = masks
        
        if self.linear_layer:
            self.grads_per_elem = torch.cat((l0_grads, l1_grads), dim=1)
        else:
            self.grads_per_elem = l0_grads
            
        if valid:
            for batch_idx, (inputs, targets) in enumerate(valloader):
                inputs, targets = inputs.to(self.device), targets.to(self.device, non_blocking=True)
                if batch_idx == 0:
                    out, l1 = self.model(inputs, last=True, freeze=True)
                    loss = cross_entropy(out, targets, reduction='none').sum()
                    l0_grads = torch.autograd.grad(loss, out)[0]
                    if self.linear_layer:
                        l0_expand = torch.repeat_interleave(l0_grads, embDim, dim=1)
                        l1_grads = l0_expand * l1.repeat(1, self.num_classes)
                    if perBatch:
                        l0_grads = l0_grads.mean(dim=0).view(1, -1)
                        if self.linear_layer:
                            l1_grads = l1_grads.mean(dim=0).view(1, -1)
                else:
                    out, l1 = self.model(inputs, last=True, freeze=True)
                    loss = cross_entropy(out, targets, reduction='none').sum()
                    batch_l0_grads = torch.autograd.grad(loss, out)[0]
                    if self.linear_layer:
                        batch_l0_expand = torch.repeat_interleave(batch_l0_grads, embDim, dim=1)
                        batch_l1_grads = batch_l0_expand * l1.repeat(1, self.num_classes)

                    if perBatch:
                        batch_l0_grads = batch_l0_grads.mean(dim=0).view(1, -1)
                        if self.linear_layer:
                            batch_l1_grads = batch_l1_grads.mean(dim=0).view(1, -1)
                    l0_grads = torch.cat((l0_grads, batch_l0_grads), dim=0)
                    if self.linear_layer:
                        l1_grads = torch.cat((l1_grads, batch_l1_grads), dim=0)
            torch.cuda.empty_cache()
            if self.linear_layer:
                self.val_grads_per_elem = torch.cat((l0_grads, l1_grads), dim=1)
            else:
                self.val_grads_per_elem = l0_grads
        self.logger.debug("Per-sample gradient computation Finished")

        
    def update_model(self, model_params, tea_model_params):
        """
        Update the models parameters

        Parameters
        ----------
        model_params: OrderedDict
            Python dictionary object containing model's parameters
        tea_model_params: OrderedDict
            Python dictionary object containing teacher model's parameters
        """
        self.model.load_state_dict(model_params)
        if self.tea_model is not None:
            self.tea_model.load_state_dict(tea_model_params)

class GradMatchStrategy(DataSelectionStrategy):
    """
    Implementation of OMPGradMatch Strategy from the paper :footcite:`pmlr-v139-killamsetty21a` for supervised learning frameworks.

    OMPGradMatch strategy tries to solve the optimization problem given below:

    .. math::
        \\underset{\\mathcal{S} \\subseteq \\mathcal{U}:|\\mathcal{S}| \\leq k, \{\\mathbf{w}_j\}_{j \\in [1, |\\mathcal{S}|]}:\\forall_{j} \\mathbf{w}_j \\geq 0}{\\operatorname{argmin\\hspace{0.7mm}}} \\left \\Vert \\underset{i \\in \\mathcal{U}}{\\sum} \\mathbf{m}_i \\nabla_{\\theta}l_u(x_i, \\theta) - \\underset{j \\in \\mathcal{S}}{\\sum} \\mathbf{m}_j \\mathbf{w}_j \\nabla_{\\theta} l_u(x_j, \\theta)\\right \\Vert

    In the above equation, :math:`\\mathbf{w}` denotes the weight vector that contains the weights for each data instance, :math:`\\mathcal{U}` denotes the unlabeled set 
    where :math:`(x^i, y^i)` denotes the :math:`i^{th}` training data point and label respectively, :math:`l_u` denotes the unlabeled loss, :math:`\\mathcal{S}` denotes the
    data subset selected at each round, and :math:`k` is the budget for the subset.

    The above optimization problem is solved using the Orthogonal Matching Pursuit(OMP) algorithm.

    Parameters
	----------
    trainloader: class
        Loading the training data using pytorch DataLoader
    valloader: class
        Loading the validation data using pytorch DataLoader
    model: class
        Model architecture used for training
    tea_model: class
        Teacher model architecture used for training
    ssl_alg: class
        SSL algorithm class
    loss: class
        Consistency loss function for unlabeled data with no reduction
    eta: float
        Learning rate. Step size for the one step gradient update
    device: str
        The device being utilized - cpu | cuda
    num_classes: int
        The number of target classes in the dataset
    linear_layer: bool
        Apply linear transformation to the data
    selection_type: str
        Type of selection -
        - 'PerClass': PerClass method is where OMP algorithm is applied on each class data points seperately.
        - 'PerBatch': PerBatch method is where OMP algorithm is applied on each minibatch data points.
        - 'PerClassPerGradient': PerClassPerGradient method is same as PerClass but we use the gradient corresponding to classification layer of that class only.
    logger : class
        logger file for printing the info
    valid : bool, optional
        If valid==True we use validation dataset gradient sum in OMP otherwise we use training dataset (default: False)
    v1 : bool
        If v1==True, we use newer version of OMP solver that is more accurate
    lam : float
        Regularization constant of OMP solver
    eps : float
        Epsilon parameter to which the above optimization problem is solved using OMP algorithm
    """

    def __init__(self, trainloader, valloader, model, tea_model, ssl_alg, loss,
                 eta, device, num_classes, linear_layer, selection_type, logger, 
                 valid=False, v1=True, lam=0, eps=1e-4):
        """
        Constructor method
        """

        super().__init__(trainloader, valloader, model, tea_model, ssl_alg, num_classes, linear_layer, loss, device, logger)
        self.eta = eta  # step size for the one step gradient update
        self.device = device
        self.selection_type = selection_type
        self.valid = valid
        self.lam = lam
        self.eps = eps
        self.v1 = v1

    def ompwrapper(self, X, Y, bud):
        """
        Wrapper function that instantiates the OMP algorithm 

        Parameters
	    ----------
        X: 
            Individual datapoint gradients 
        Y: 
            Gradient sum that needs to be matched to.
        bud:
            Budget of datapoints that needs to be sampled from the unlabeled set

        Returns
        ----------
        idxs: list
            List containing indices of the best datapoints,
        gammas: weights tensors
            Tensor containing weights of each instance
        """

        if self.device == "cpu":
            reg = OrthogonalMP_REG(X.numpy(), Y.numpy(), nnz=bud, positive=True, lam=0)
            ind = np.nonzero(reg)[0]
        else:
            if self.v1:
                reg = OrthogonalMP_REG_Parallel_V1(X, Y, nnz=bud,
                                                 positive=True, lam=self.lam,
                                                 tol=self.eps, device=self.device)
            else:
                reg = OrthogonalMP_REG_Parallel(X, Y, nnz=bud,
                                                positive=True, lam=self.lam,
                                                tol=self.eps, device=self.device)
            ind = torch.nonzero(reg).view(-1)
        return ind.tolist(), reg[ind].tolist()

    def select(self, budget, model_params, tea_model_params):
        """
        Apply OMP Algorithm for data selection

        Parameters
        ----------
        budget: int
            The number of data points to be selected
        model_params: OrderedDict
            Python dictionary object containing model's parameters
        tea_model_params: OrderedDict
            Python dictionary object containing teacher model's parameters

        Returns
        --------
        idxs: list
            List containing indices of the best datapoints,
        gammas: weights tensors
            Tensor containing weights of each instance
        """
        omp_start_time = time.time()
        self.update_model(model_params, tea_model_params)
        if self.selection_type == 'PerClass':
            self.get_labels(valid=self.valid)
            idxs = []
            gammas = []
            for i in range(self.num_classes):
                trn_subset_idx = torch.where(self.trn_lbls == i)[0].tolist()
                trn_data_sub = Subset(self.trainloader.dataset, trn_subset_idx)
                self.pctrainloader = DataLoader(trn_data_sub, batch_size=self.trainloader.batch_size,
                                          shuffle=False, pin_memory=True)
                if self.valid:
                    val_subset_idx = torch.where(self.val_lbls == i)[0].tolist()
                    val_data_sub = Subset(self.valloader.dataset, val_subset_idx)
                    self.pcvalloader = DataLoader(val_data_sub, batch_size=self.trainloader.batch_size,
                                                    shuffle=False, pin_memory=True)

                self.compute_gradients(self.valid, perBatch=False, perClass=True)
                trn_gradients = self.grads_per_elem
                if self.valid:
                    sum_val_grad = torch.sum(self.val_grads_per_elem, dim=0)
                else:
                    sum_val_grad = torch.sum(trn_gradients, dim=0)
                idxs_temp, gammas_temp = self.ompwrapper(torch.transpose(trn_gradients, 0, 1),
                                          sum_val_grad, math.ceil(budget * len(trn_subset_idx) / self.N_trn))
                idxs.extend(list(np.array(trn_subset_idx)[idxs_temp]))
                gammas.extend(gammas_temp)

        elif self.selection_type == 'PerBatch':
            self.compute_gradients(self.valid, perBatch=True, perClass=False)
            idxs = []
            gammas = []
            trn_gradients = self.grads_per_elem
            if self.valid:
                sum_val_grad = torch.sum(self.val_grads_per_elem, dim=0)
            else:
                sum_val_grad = torch.sum(trn_gradients, dim=0)
            idxs_temp, gammas_temp = self.ompwrapper(torch.transpose(trn_gradients, 0, 1),
                                                     sum_val_grad, math.ceil(budget/self.trainloader.batch_size))
            batch_wise_indices = list(self.trainloader.batch_sampler)
            for i in range(len(idxs_temp)):
                tmp = batch_wise_indices[idxs_temp[i]]
                idxs.extend(tmp)
                gammas.extend(list(gammas_temp[i] * np.ones(len(tmp))))

        elif self.selection_type == 'PerClassPerGradient':
            self.get_labels(valid=self.valid)
            idxs = []
            gammas = []
            embDim = self.model.get_embedding_dim()
            for i in range(self.num_classes):
                trn_subset_idx = torch.where(self.trn_lbls == i)[0].tolist()
                trn_data_sub = Subset(self.trainloader.dataset, trn_subset_idx)
                self.pctrainloader = DataLoader(trn_data_sub, batch_size=self.trainloader.batch_size,
                                          shuffle=False, pin_memory=True)
                if self.valid:
                    val_subset_idx = torch.where(self.val_lbls == i)[0].tolist()
                    val_data_sub = Subset(self.valloader.dataset, val_subset_idx)
                    self.pcvalloader = DataLoader(val_data_sub, batch_size=self.trainloader.batch_size,
                                                    shuffle=False, pin_memory=True)
                self.compute_gradients(self.valid, perBatch=False, perClass=True)
                trn_gradients = self.grads_per_elem
                tmp_gradients = trn_gradients[:, i].view(-1, 1)
                tmp1_gradients = trn_gradients[:,
                                 self.num_classes + (embDim * i): self.num_classes + (embDim * (i + 1))]
                trn_gradients = torch.cat((tmp_gradients, tmp1_gradients), dim=1)

                if self.valid:
                    val_gradients = self.val_grads_per_elem
                    tmp_gradients = val_gradients[:, i].view(-1, 1)
                    tmp1_gradients = val_gradients[:,
                                     self.num_classes + (embDim * i): self.num_classes + (embDim * (i + 1))]
                    val_gradients = torch.cat((tmp_gradients, tmp1_gradients), dim=1)
                    sum_val_grad = torch.sum(val_gradients, dim=0)
                else:
                    sum_val_grad = torch.sum(trn_gradients, dim=0)

                idxs_temp, gammas_temp = self.ompwrapper(torch.transpose(trn_gradients, 0, 1),
                                          sum_val_grad, math.ceil(budget * len(trn_subset_idx) / self.N_trn))
                idxs.extend(list(np.array(trn_subset_idx)[idxs_temp]))
                gammas.extend(gammas_temp)

        omp_end_time = time.time()
        diff = budget - len(idxs)

        if diff > 0:
            remainList = set(np.arange(self.N_trn)).difference(set(idxs))
            new_idxs = np.random.choice(list(remainList), size=diff, replace=False)
            idxs.extend(new_idxs)
            gammas.extend([1 for _ in range(diff)])
            idxs = np.array(idxs)
            gammas = np.array(gammas)

        if self.selection_type in ["PerClass", "PerClassPerGradient"]:
            rand_indices = np.random.permutation(len(idxs))
            idxs = list(np.array(idxs)[rand_indices])
            gammas = list(np.array(gammas)[rand_indices])

        self.logger.debug("OMP algorithm Subset Selection time is: %f", omp_end_time - omp_start_time)
        return idxs, gammas