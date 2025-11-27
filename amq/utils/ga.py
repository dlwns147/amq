import numpy as np
from pymoo.core.mutation import Mutation
from pymoo.core.sampling import Sampling
from pymoo.core.crossover import Crossover
from pymoo.operators.mutation.pm import PolynomialMutation
from pymoo.core.crossover import Crossover


def apply_float_operation(problem, fun):

    # save the original bounds of the problem
    _xl, _xu = problem.xl, problem.xu

    # copy the arrays of the problem and cast them to float
    xl, xu = problem.xl.astype(float), problem.xu.astype(float)

    # modify the bounds to match the new crossover specifications and set the problem
    problem.xl = xl - (0.5 - 1e-7)
    problem.xu = xu + (0.5 - 1e-7)

    # perform the crossover
    off = fun()

    # now round to nearest integer for all offsprings
    off = np.rint(off).astype(int)

    # reset the original bounds of the problem and design space values
    problem.xl = _xl
    problem.xu = _xu

    return off


class IntegerFromFloatMutation(Mutation):

    def __init__(self, clazz=None, **kwargs):
        if clazz is None:
            raise Exception("Please define the class of the default mutation to use IntegerFromFloatMutation.")

        self.mutation = clazz(**kwargs)
        super().__init__()

    def _do(self, problem, X, **kwargs):
        def fun():
            return self.mutation._do(problem, X, **kwargs)

        return apply_float_operation(problem, fun)


class IntMutation(Mutation):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _do(self, problem, X, **kwargs):
        # import pdb; pdb.set_trace()
        xl, xu = problem.xl, problem.xu
        B = X.shape[0]
        return np.random.randint(xl, xu + 1, (B, problem.n_var)).astype(int)
        # mut = np.random.randint(xl, xu + 1, (B, problem.n_var)).astype(int)
        # mask = np.random.binomial(1, self.prob.value, (B, problem.n_var))
        # return mask * mut + (1 - mask) * X

class IntPolynomialMutation(PolynomialMutation):

    def _do(self, problem, X, params=None, **kwargs):
        return super()._do(problem, X, params, **kwargs).round().astype(int)


class MyTwoPointCrossover(Crossover):

    def __init__(self, n_offsprings, **kwargs):
        super().__init__(2, n_offsprings, **kwargs)
        self.n_points = 2

    def _do(self, _, X, **kwargs):

        # get the X of parents and count the matings
        _, n_matings, n_var = X.shape

        # start point of crossover
        r = np.row_stack([np.random.permutation(n_var - 1) + 1 for _ in range(n_matings)])[:, :self.n_points]
        r.sort(axis=1)
        r = np.column_stack([r, np.full(n_matings, n_var)])

        # the mask do to the crossover
        M = np.full((n_matings, n_var), False)

        # create for each individual the crossover range
        for i in range(n_matings):

            j = 0
            while j < r.shape[1] - 1:
                a, b = r[i, j], r[i, j + 1]
                M[i, a:b] = True
                j += 2

        if self.n_offsprings == 1:
            Xp = X[0].copy()
            Xp[~M] = X[1][~M]
            Xp = Xp[None, ...]
        elif self.n_offsprings == 2:
            Xp = np.copy(X)
            Xp[0][~M] = X[1][~M]
            Xp[1][~M] = X[0][~M]
        else:
            raise Exception

        return Xp

class MyUniformCrossover(Crossover):

    def __init__(self, n_offsprings, **kwargs):
        super().__init__(2, n_offsprings, **kwargs)

    def _do(self, _, X, **kwargs):
        _, n_matings, n_var = X.shape
        M = np.random.random((n_matings, n_var)) < 0.5
        # _X = crossover_mask(X, M)
        if self.n_offsprings == 1:
            _X = X[0].copy()
            _X[~M] = X[1][~M]
            _X = _X[None, ...]
        elif self.n_offsprings == 2:
            _X = np.copy(X)
            _X[0][~M] = X[1][~M]
            _X[1][~M] = X[0][~M]
        else:
            raise Exception
        return _X


class MySampling(Sampling):

    def _do(self, problem, n_samples, **kwargs):
        X = np.full((n_samples, problem.n_var), False, dtype=bool)

        for k in range(n_samples):
            I = np.random.permutation(problem.n_var)[:problem.n_max]
            X[k, I] = True

        return X


class BinaryCrossover(Crossover):
    def __init__(self):
        super().__init__(2, 1)

    def _do(self, problem, X, **kwargs):
        n_objs, n_matings, n_var = X.shape

        _X = np.full((self.n_offsprings, n_matings, problem.n_var), False)

        for k in range(n_matings):
            p1, p2 = X[0, k], X[1, k]

            both_are_true = np.logical_and(p1, p2)
            _X[0, k, both_are_true] = True

            n_remaining = problem.n_max - np.sum(both_are_true)

            I = np.where(np.logical_xor(p1, p2))[0]

            S = I[np.random.permutation(len(I))][:n_remaining]
            _X[0, k, S] = True

        return _X


class MyMutation(Mutation):
    def _do(self, problem, X, **kwargs):
        for i in range(X.shape[0]):
            X[i, :] = X[i, :]
            is_false = np.where(np.logical_not(X[i, :]))[0]
            is_true = np.where(X[i, :])[0]
            try:
                X[i, np.random.choice(is_false)] = True
                X[i, np.random.choice(is_true)] = False
            except ValueError:
                pass

        return X


# def mut_pm(X, xl, xu, eta, prob, at_least_once, problem):
#     n, n_var = X.shape
#     assert len(eta) == n
#     assert len(prob) == n

#     Xp = np.full(X.shape, np.inf)
#     # import pdb; pdb.set_trace()

#     # for i in tqdm(range(X.shape[0]), desc='mutation'):
#     for i in range(X.shape[0]):

#         while True:
#             # import pdb; pdb.set_trace()
#             # prob = np.random.rand(1)
#             Xp[i, :] = X[i]
#             mut = mut_binomial(1, n_var, prob, at_least_once=at_least_once)[0]
#             mut[xl == xu] = False

#             if mut.any():
#                 _xl = xl[mut]
#                 _xu = xu[mut]

#                 X_mut = X[i][mut]
#                 _eta = np.repeat(eta[i], n_var)[mut]
                
#                 delta1 = (X_mut - _xl) / (_xu - _xl)
#                 delta2 = (_xu - X_mut) / (_xu - _xl)

#                 mut_pow = 1.0 / (_eta + 1.0)

#                 rand = np.random.random((X_mut.shape))
#                 mask = rand <= 0.5
#                 mask_not = np.logical_not(mask)

#                 deltaq = np.zeros(X_mut.shape)

#                 xy = 1.0 - delta1
#                 val = 2.0 * rand + (1.0 - 2.0 * rand) * (np.power(xy, (_eta + 1.0)))
#                 d = np.power(val, mut_pow) - 1.0
#                 deltaq[mask] = d[mask]

#                 xy = 1.0 - delta2
#                 val = 2.0 * (1.0 - rand) + 2.0 * (rand - 0.5) * (np.power(xy, (_eta + 1.0)))
#                 d = 1.0 - (np.power(val, mut_pow))
#                 deltaq[mask_not] = d[mask_not]

#                 # mutated values
#                 _Y = X_mut + deltaq * (_xu - _xl)

#                 # back in bounds if necessary (floating point issues)
#                 _Y[_Y < _xl] = _xl[_Y < _xl]
#                 _Y[_Y > _xu] = _xu[_Y > _xu]

#                 # set the values for output
#                 Xp[i, mut] = _Y

#                 # Xp[i, :] = set_to_bounds_if_outside(Xp[i], xl, xu)
            
#             complexity = get_net_info(problem.ss.decode(Xp[i].astype(int)), problem.config)[problem.sec_obj]
#             if problem.ss.sec_obj_range[0] <= complexity and problem.ss.sec_obj_range[1] >= complexity:
#                 break
#     Xp = set_to_bounds_if_outside(Xp, xl, xu)

#     return Xp

# class BoundedIntPolynomialMutation(PolynomialMutation):

#     def __init__(self, **kwargs):
#         super().__init__(**kwargs)

#     def _do(self, problem, X, params=None, **kwargs):
#         X = X.astype(float)

#         eta = get(self.eta, size=len(X))
#         prob_var = self.get_prob_var(problem, size=len(X))

#         Xp = mut_pm(X, problem.xl, problem.xu, eta, prob_var, at_least_once=self.at_least_once, problem=problem)

#         # return Xp
#         return Xp.round().astype(int)

# class BoundedBinomialCrossover(Crossover):

#     def __init__(self, bias=0.5, n_offsprings=2, **kwargs):
#         super().__init__(2, n_offsprings, **kwargs)
#         self.bias = Real(bias, bounds=(0.1, 0.9), strict=(0.0, 1.0))

#     def _do(self, problem, X, **kwargs):
#         _, n_matings, n_var = X.shape
#         Xp = np.full((self.n_offsprings, n_matings, n_var), np.inf)

#         for i in range(n_matings):
#             while True:
#                 bias = get(self.bias, size=1)
#                 M = mut_binomial(1, n_var, bias, at_least_once=True)[0]
                
#                 if self.n_offsprings == 1:
#                     Xp[:, i] = X[0, i][None, :].copy()
#                     Xp[:, i, ~M] = X[1, i, ~M][None, :]
#                 elif self.n_offsprings == 2:
#                     Xp[:, i] = X[:, i].copy()
#                     Xp[0, i, ~M] = X[1, i, ~M]
#                     Xp[1, i, ~M] = X[0, i, ~M]
#                 else:
#                     raise Exception

#                 complexity = [get_net_info(problem.ss.decode(_Xp.astype(int)), problem.config)[problem.sec_obj] for _Xp in Xp[:, i]]
#                 if all([problem.ss.sec_obj_range[0] <= complexity[j] and problem.ss.sec_obj_range[1] >= complexity[j] for j in range(self.n_offsprings)]):
#                     break

#         return Xp
    
# class BoundedTwoPointCrossover(Crossover):
        
#     def __init__(self, n_points=2, n_offsprings=2, **kwargs):
#         super().__init__(n_parents=2, n_offsprings=n_offsprings, **kwargs)
#         self.n_points = n_points

#     def _do(self, problem, X, **kwargs):

#         # get the X of parents and count the matings
#         _, n_matings, n_var = X.shape
#         Xp = np.full((self.n_offsprings, n_matings, n_var), np.inf)

#         for i in range(n_matings):
#             while True:
#                 # start point of crossover
#                 r = (np.random.permutation(n_var - 1) + 1)[:self.n_points]
#                 r.sort()
#                 r = np.concatenate([r, np.array((n_var,))])

#                 # the mask do to the crossover
#                 M = np.full((n_var), False)

#                 # create for each individual the crossover range
#                 j = 0
#                 while j < len(r) - 1:
#                     a, b = r[j], r[j + 1]
#                     M[a:b] = True
#                     j += 2
                    
#                 if self.n_offsprings == 1:
#                     Xp[:, i] = X[0, i][None, :].copy()
#                     Xp[:, i, ~M] = X[1, i, ~M][None, :]
#                 elif self.n_offsprings == 2:
#                     Xp[:, i] = X[:, i].copy()
#                     Xp[0, i, ~M] = X[1, i, ~M]
#                     Xp[1, i, ~M] = X[0, i, ~M]
#                 else:
#                     raise Exception

#                 complexity = [get_net_info(problem.ss.decode(_Xp.astype(int)), problem.config)[problem.sec_obj] for _Xp in Xp[:, i]]
#                 if all([problem.ss.sec_obj_range[0] <= complexity[j] and problem.ss.sec_obj_range[1] >= complexity[j] for j in range(self.n_offsprings)]):
#                     break

#         return Xp

# def apply_float_operation(problem, fun):

#     # save the original bounds of the problem
#     _xl, _xu = problem.xl, problem.xu

#     # copy the arrays of the problem and cast them to float
#     xl, xu = problem.xl.astype(float), problem.xu.astype(float)

#     # modify the bounds to match the new crossover specifications and set the problem
#     problem.xl = xl - (0.5 - 1e-7)
#     problem.xu = xu + (0.5 - 1e-7)

#     # perform the crossover
#     off = fun()

#     # now round to nearest integer for all offsprings
#     off = np.rint(off).astype(int)

#     # reset the original bounds of the problem and design space values
#     problem.xl = _xl
#     problem.xu = _xu

#     return off

# class IntegerFromFloatMutation(Mutation):

#     def __init__(self, clazz=None, **kwargs):
#         if clazz is None:
#             raise Exception("Please define the class of the default mutation to use IntegerFromFloatMutation.")

#         self.mutation = clazz(**kwargs)
#         super().__init__()

#     def _do(self, problem, X, **kwargs):
#         def fun():
#             return self.mutation._do(problem, X, **kwargs)

#         return apply_float_operation(problem, fun)
