# Import Utils
import sys, os
import warnings
import numpy as np
import pandas as pd
import multiprocessing as mp
from numpy.random import default_rng
from tqdm import tqdm, _tqdm
from functools import lru_cache, partial
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import chain, compress, product, repeat
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Type, Union


# Local Imports
from ...qrde5_backtesting import Backtest


class Optimizer(Backtest):
    """ 
    Extends class Backtest to add optimization tools
    """
    def optimize(self, *,
                    maximize: Union[str, Callable[[pd.Series], float]] = 'SQN',
                    method: str = 'grid',
                    max_tries: Optional[Union[int, float]] = None,
                    constraint: Optional[Callable[[dict], bool]] = None,
                    return_heatmap: bool = False,
                    return_optimization: bool = False,
                    random_state: Optional[int] = None,
                    **kwargs) -> Union[pd.Series,
                                        Tuple[pd.Series, pd.Series],
                                        Tuple[pd.Series, pd.Series, dict]]:
            """
            Optimize strategy parameters to an optimal combination.
            Returns result `pd.Series` of the best run.

            `maximize` is a string key from the
            `backtesting.backtesting.Backtest.run`-returned results series,
            or a function that accepts this series object and returns a number;
            the higher the better. By default, the method maximizes
            Van Tharp's [System Quality Number](https://google.com/search?q=System+Quality+Number).

            `method` is the optimization method. Currently two methods are supported:

            * `"grid"` which does an exhaustive (or randomized) search over the
            cartesian product of parameter combinations, and
            * `"skopt"` which finds close-to-optimal strategy parameters using
            [model-based optimization], making at most `max_tries` evaluations.

            [model-based optimization]: \
                https://scikit-optimize.github.io/stable/auto_examples/bayesian-optimization.html

            `max_tries` is the maximal number of strategy runs to perform.
            If `method="grid"`, this results in randomized grid search.
            If `max_tries` is a floating value between (0, 1], this sets the
            number of runs to approximately that fraction of full grid space.
            Alternatively, if integer, it denotes the absolute maximum number
            of evaluations. If unspecified (default), grid search is exhaustive,
            whereas for `method="skopt"`, `max_tries` is set to 200.

            `constraint` is a function that accepts a dict-like object of
            parameters (with values) and returns `True` when the combination
            is admissible to test with. By default, any parameters combination
            is considered admissible.

            If `return_heatmap` is `True`, besides returning the result
            series, an additional `pd.Series` is returned with a multiindex
            of all admissible parameter combinations, which can be further
            inspected or projected onto 2D to plot a heatmap
            (see `backtesting.lib.plot_heatmaps()`).

            If `return_optimization` is True and `method = 'skopt'`,
            in addition to result series (and maybe heatmap), return raw
            [`scipy.optimize.OptimizeResult`][OptimizeResult] for further
            inspection, e.g. with [scikit-optimize]\
            [plotting tools].

            [OptimizeResult]: \
                https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.OptimizeResult.html
            [scikit-optimize]: https://scikit-optimize.github.io
            [plotting tools]: https://scikit-optimize.github.io/stable/modules/plots.html

            If you want reproducible optimization results, set `random_state`
            to a fixed integer random seed.

            Additional keyword arguments represent strategy arguments with
            list-like collections of possible values. For example, the following
            code finds and returns the "best" of the 7 admissible (of the
            9 possible) parameter combinations:

                backtest.optimize(sma1=[5, 10, 15], sma2=[10, 20, 40],
                                constraint=lambda p: p.sma1 < p.sma2)

            .. TODO::
                Improve multiprocessing/parallel execution on Windos with start method 'spawn'.
            """
            if not kwargs:
                raise ValueError('Need some strategy parameters to optimize')

            maximize_key = None
            if isinstance(maximize, str):
                maximize_key = str(maximize)
                stats = self._results if self._results is not None else self.run()
                if maximize not in stats:
                    raise ValueError('`maximize`, if str, must match a key in pd.Series '
                                    'result of backtest.run()')

                def maximize(stats: pd.Series, _key=maximize):
                    return stats[_key]

            elif not callable(maximize):
                raise TypeError('`maximize` must be str (a field of backtest.run() result '
                                'Series) or a function that accepts result Series '
                                'and returns a number; the higher the better')
            assert callable(maximize), maximize

            have_constraint = bool(constraint)
            if constraint is None:

                def constraint(_):
                    return True

            elif not callable(constraint):
                raise TypeError("`constraint` must be a function that accepts a dict "
                                "of strategy parameters and returns a bool whether "
                                "the combination of parameters is admissible or not")
            assert callable(constraint), constraint

            if return_optimization and method != 'skopt':
                raise ValueError("return_optimization=True only valid if method='skopt'")

            def _tuple(x):
                return x if isinstance(x, Sequence) and not isinstance(x, str) else (x,)

            for k, v in kwargs.items():
                if len(_tuple(v)) == 0:
                    raise ValueError(f"Optimization variable '{k}' is passed no "
                                    f"optimization values: {k}={v}")

            class AttrDict(dict):
                def __getattr__(self, item):
                    return self[item]

            def _grid_size():
                size = int(np.prod([len(_tuple(v)) for v in kwargs.values()]))
                if size < 10_000 and have_constraint:
                    size = sum(1 for p in product(*(zip(repeat(k), _tuple(v))
                                                    for k, v in kwargs.items()))
                            if constraint(AttrDict(p)))
                return size

            def _optimize_grid() -> Union[pd.Series, Tuple[pd.Series, pd.Series]]:
                rand = default_rng(random_state).random
                grid_frac = (1 if max_tries is None else
                            max_tries if 0 < max_tries <= 1 else
                            max_tries / _grid_size())
                param_combos = [dict(params)  # back to dict so it pickles
                                for params in (AttrDict(params)
                                            for params in product(*(zip(repeat(k), _tuple(v))
                                                                    for k, v in kwargs.items())))
                                if constraint(params)  # type: ignore
                                and rand() <= grid_frac]
                if not param_combos:
                    raise ValueError('No admissible parameter combinations to test')

                if len(param_combos) > 300:
                    warnings.warn(f'Searching for best of {len(param_combos)} configurations.',
                                stacklevel=2)

                heatmap = pd.Series(np.nan,
                                    name=maximize_key,
                                    index=pd.MultiIndex.from_tuples(
                                        [p.values() for p in param_combos],
                                        names=next(iter(param_combos)).keys()))

                def _batch(seq):
                    n = np.clip(int(len(seq) // (os.cpu_count() or 1)), 1, 300)
                    for i in range(0, len(seq), n):
                        yield seq[i:i + n]

                # Save necessary objects into "global" state; pass into concurrent executor
                # (and thus pickle) nothing but two numbers; receive nothing but numbers.
                # With start method "fork", children processes will inherit parent address space
                # in a copy-on-write manner, achieving better performance/RAM benefit.
                backtest_uuid = np.random.random()
                param_batches = list(_batch(param_combos))
                Backtest._mp_backtests[backtest_uuid] = (self, param_batches, maximize)  # type: ignore
                try:
                    # If multiprocessing start method is 'fork' (i.e. on POSIX), use
                    # a pool of processes to compute results in parallel.
                    # Otherwise (i.e. on Windos), sequential computation will be "faster".
                    if mp.get_start_method(allow_none=False) == 'fork':
                        with ProcessPoolExecutor() as executor:
                            futures = [executor.submit(Backtest._mp_task, backtest_uuid, i)
                                    for i in range(len(param_batches))]
                            for future in _tqdm(as_completed(futures), total=len(futures),
                                                desc='Backtest.optimize'):
                                batch_index, values = future.result()
                                for value, params in zip(values, param_batches[batch_index]):
                                    heatmap[tuple(params.values())] = value
                    else:
                        if os.name == 'posix':
                            warnings.warn("For multiprocessing support in `Backtest.optimize()` "
                                        "set multiprocessing start method to 'fork'.")
                        for batch_index in _tqdm(range(len(param_batches))):
                            _, values = Backtest._mp_task(backtest_uuid, batch_index)
                            for value, params in zip(values, param_batches[batch_index]):
                                heatmap[tuple(params.values())] = value
                finally:
                    del Backtest._mp_backtests[backtest_uuid]

                best_params = heatmap.idxmax()

                if pd.isnull(best_params):
                    # No trade was made in any of the runs. Just make a random
                    # run so we get some, if empty, results
                    stats = self.run(**param_combos[0])
                else:
                    stats = self.run(**dict(zip(heatmap.index.names, best_params)))

                if return_heatmap:
                    return stats, heatmap
                return stats

            def _optimize_skopt() -> Union[pd.Series,
                                        Tuple[pd.Series, pd.Series],
                                        Tuple[pd.Series, pd.Series, dict]]:
                try:
                    from skopt import forest_minimize
                    from skopt.callbacks import DeltaXStopper
                    from skopt.learning import ExtraTreesRegressor
                    from skopt.space import Categorical, Integer, Real
                    from skopt.utils import use_named_args
                except ImportError:
                    raise ImportError("Need package 'scikit-optimize' for method='skopt'. "
                                    "pip install scikit-optimize") from None

                nonlocal max_tries
                max_tries = (200 if max_tries is None else
                            max(1, int(max_tries * _grid_size())) if 0 < max_tries <= 1 else
                            max_tries)

                dimensions = []
                for key, values in kwargs.items():
                    values = np.asarray(values)
                    if values.dtype.kind in 'mM':  # timedelta, datetime64
                        # these dtypes are unsupported in skopt, so convert to raw int
                        # TODO: save dtype and convert back later
                        values = values.astype(int)

                    if values.dtype.kind in 'iumM':
                        dimensions.append(Integer(low=values.min(), high=values.max(), name=key))
                    elif values.dtype.kind == 'f':
                        dimensions.append(Real(low=values.min(), high=values.max(), name=key))
                    else:
                        dimensions.append(Categorical(values.tolist(), name=key, transform='onehot'))

                # Avoid recomputing re-evaluations:
                # "The objective has been evaluated at this point before."
                # https://github.com/scikit-optimize/scikit-optimize/issues/302
                memoized_run = lru_cache()(lambda tup: self.run(**dict(tup)))

                # np.inf/np.nan breaks sklearn, np.finfo(float).max breaks skopt.plots.plot_objective
                INVALID = 1e300
                progress = iter(_tqdm(repeat(None), total=max_tries, desc='Backtest.optimize'))

                @use_named_args(dimensions=dimensions)
                def objective_function(**params):
                    next(progress)
                    # Check constraints
                    # TODO: Adjust after https://github.com/scikit-optimize/scikit-optimize/pull/971
                    if not constraint(AttrDict(params)):
                        return INVALID
                    res = memoized_run(tuple(params.items()))
                    value = -maximize(res)
                    if np.isnan(value):
                        return INVALID
                    return value

                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        'ignore', 'The objective has been evaluated at this point before.')

                    res = forest_minimize(
                        func=objective_function,
                        dimensions=dimensions,
                        n_calls=max_tries,
                        base_estimator=ExtraTreesRegressor(n_estimators=20, min_samples_leaf=2),
                        acq_func='LCB',
                        kappa=3,
                        n_initial_points=min(max_tries, 20 + 3 * len(kwargs)),
                        initial_point_generator='lhs',  # 'sobel' requires n_initial_points ~ 2**N
                        callback=DeltaXStopper(9e-7),
                        random_state=random_state)

                stats = self.run(**dict(zip(kwargs.keys(), res.x)))
                output = [stats]

                if return_heatmap:
                    heatmap = pd.Series(dict(zip(map(tuple, res.x_iters), -res.func_vals)),
                                        name=maximize_key)
                    heatmap.index.names = kwargs.keys()
                    heatmap = heatmap[heatmap != -INVALID]
                    heatmap.sort_index(inplace=True)
                    output.append(heatmap)

                if return_optimization:
                    valid = res.func_vals != INVALID
                    res.x_iters = list(compress(res.x_iters, valid))
                    res.func_vals = res.func_vals[valid]
                    output.append(res)

                return stats if len(output) == 1 else tuple(output)

            if method == 'grid':
                output = _optimize_grid()
            elif method == 'skopt':
                output = _optimize_skopt()
            else:
                raise ValueError(f"Method should be 'grid' or 'skopt', not {method!r}")
            return output

    
