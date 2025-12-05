def get_predictor(model, inputs, targets, device='cpu', **kwargs):

    if model == 'rbf':
        from predictor.rbf import RBF
        predictor = RBF(**kwargs)
        predictor.fit(inputs, targets)

    elif model == 'mlp':
        from predictor.mlp import MLP
        predictor = MLP(n_feature=inputs.shape[1], device=device)
        predictor.fit(x=inputs, y=targets, device=device)

    else:
        raise NotImplementedError

    return predictor

