from typing import Callable

import torch

import pandas as pd


class StratificationGenerator():
    def __init__(self, model: Callable[[torch.Tensor], torch.Tensor],
                 df_points: pd.core.frame.DataFrame,
                 n_classes: int, order: int = 2):
        """Class that generated stratification types and normal vectors for
        the decision boundary of a neural network for multiclass classification

        Args:
            model (Callable[[torch.Tensor], torch.Tensor]): Neural network of
            type [*, 3]->[*, n_classes] returning the logit probability of a
            3d point belonging to a class
            n_classes (int): number of classes
            df_points (pd.core.frame.DataFrame): DataFrame of points lying in
            stratum of order=order of the decision boundary
        """
        assert order in [2, 3], "Stratification only works for orders in [2, 3] \
            up to now"
        self.order = order
        self.model = model
        self.n_classes = n_classes
        assert type(df_points.index) == pd.core.indexes.range.RangeIndex,\
            "Dataframe index must be RangeIndex"
        assert df_points.index.step == 1,\
            "Dataframe index must have step equal to 1"

        self.df_points = df_points

    def create_stratification_type_df(self,
                                      batch_size: int = 128,
                                      verbose=False):
        if verbose:
            print("Starting computation of stratification type")

        def batch_stratication_creation(df_batch):
            t_points = torch.tensor(df_batch.values, dtype=torch.float32)
            t_logits = self.model(t_points)
            t_top_order = torch.topk(t_logits, k=self.order).indices
            top_orders = [sorted(tuple(top_order.detach().numpy()))
                          for top_order in t_top_order]
            return pd.DataFrame({"strat_type": top_orders},
                                index=df_batch.index)

        # Create DataFrame of strat_type batchwise
        df_strat_type = pd.DataFrame(columns=["strat_type"])
        for batch_num, df_batch in enumerate(
                self.create_df_generator(self.df_points, batch_size)):
            stratification_batch = batch_stratication_creation(df_batch)
            df_strat_type = df_strat_type.append(stratification_batch)
            if verbose and batch_num % 50 == 0:
                print('batch number:', batch_num, ' / ',
                      int(self.df_points.shape[0] / batch_size))
        # inner join with df_points
        df_strat_type = pd.concat([self.df_points, df_strat_type], axis=1,
                                  join='inner')
        # Return joint DataFrame sorted by strat_type to make the computation
        # of the normals easier
        return df_strat_type.sort_values(by=["strat_type"])

    def create_normal_df(self, batch_size: int = 128,
                         verbose=False):
        assert self.order == 2, "Normal vectors can only be computed for \
            order 2 strata."

        df_strat = self.create_stratification_type_df(batch_size=batch_size,
                                                      verbose=verbose)
        if verbose:
            print("Starting computation of normal vectors")
        def batch_normal_creation(df_batch):
            list_strat_types = list(df_batch['strat_type'])
            unique_strat_types = [list(x) for x in
                                  set(tuple(x) for x in list_strat_types)]
            df_normal_batch = pd.DataFrame(columns=["n0", "n1", "n2"])
            for strat_type in unique_strat_types:
                # filter all points that are of stratification type strat_type

                # workaround since numpy.int64 cause errors in isin
                strat_type = [int(el) for el in strat_type]

                df_batch_st = df_batch[
                    df_batch['strat_type'].isin([strat_type])
                    ]
                top_class = strat_type[-1]
                # Create tensor of points lying in stratum strat_type
                t_points_st = torch.tensor(
                    df_batch_st[["x0", "x1", "x2"]].values,
                    dtype=torch.float32,
                    requires_grad=False)
                assert t_points_st.shape[0] != 0, f"There are no points of \
                    stratification type: {strat_type}"
                # Computing the gradient of the top_class component of
                # self.model
                delta = torch.zeros_like(t_points_st, requires_grad=True)
                loss = torch.sum(self.model(t_points_st + delta)[:, top_class])
                loss.backward()
                t_points_st_gradient = delta.grad
                # Normalize gradient
                t_st_norm = t_points_st_gradient \
                    / torch.sqrt((t_points_st_gradient**2)
                                 .sum(axis=1, keepdims=True))

                df_strat_type = pd.DataFrame(
                    data=t_st_norm.detach().numpy(),
                    columns=["n0", "n1", "n2"],
                    index=df_batch_st.index)
                df_normal_batch = df_normal_batch.append(df_strat_type)
            return df_normal_batch

        df_normal = pd.DataFrame(columns=["n0", "n1", "n2"])
        for batch_num, df_batch in enumerate(
                self.create_df_generator(df_strat, batch_size)):
            batch_normal = batch_normal_creation(df_batch)
            df_normal = df_normal.append(batch_normal)
            if verbose and batch_num % 50 == 0:
                print('batch number:', batch_num, ' / ',
                      int(df_strat.shape[0] / batch_size))

        # inner join with df_strat
        df_normal = pd.concat([df_strat, df_normal], axis=1,
                              join='inner')
        return df_normal

    def create_df_generator(self, df: pd.core.frame.DataFrame,
                            batch_size: int = 128):
        """Create generator of Dataframe

        Args:
            df (pd.core.frame.DataFrame): dataframe to load from
            batch_size (int, optional): batch_size. Defaults to 128.

        Yields:
            pd.core.frame.DataFrame: Dataframe batch
        """
        df_size = df.shape[0]
        idx = 0
        while idx + batch_size < df_size:
            yield df[idx:idx + batch_size]
            idx += batch_size
        yield df[idx:]
