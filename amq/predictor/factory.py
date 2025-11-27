def get_predictor(model, inputs, targets, device='cpu', **kwargs):

    if model == 'rbf':
        from amq.predictor.rbf import RBF
        predictor = RBF(**kwargs)
        predictor.fit(inputs, targets)

    elif model == 'carts':
        from amq.predictor.carts import CART
        predictor = CART(n_tree=5000)
        predictor.fit(inputs, targets)

    elif model == 'gp':
        from amq.predictor.gp import GP
        predictor = GP()
        predictor.fit(inputs, targets)

    elif model == 'mlp':
        from amq.predictor.mlp import MLP
        predictor = MLP(n_feature=inputs.shape[1], device=device)
        predictor.fit(x=inputs, y=targets, device=device)

    elif model == 'as':
        from amq.predictor.adaptive_switching import AdaptiveSwitching
        predictor = AdaptiveSwitching()
        predictor.fit(inputs, targets)

    else:
        raise NotImplementedError

    return predictor

