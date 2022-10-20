import torch
import pandas as pd
from torch import nn
from typing import List, Dict, Tuple, Iterator
from torch.utils.data import DataLoader

from submodules import NBEATSGenericBlock, NBEATSSeasonalBlock, NBEATSTrendBlock, MultiEmbedding

TIME_IDX = "_time_idx_"


class NbeatsNetwork(nn.Module):
    def __init__(
        self,
        targets: str,
        model_type: str = "G",  # 'I'
        num_stack: int = 1,
        num_block: int = 1,
        width: int = 8,  # [2**9]
        expansion_coe: int = 5,  # [2**5]
        num_block_layer: int = 4,
        prediction_length: int = 0,
        context_length: int = 0,
        dropout: float = 0.1,
        backcast_loss_ratio: float = 0.1,
        covariate_number: int = 0,
        encoder_cont: List[str] = [],
        decoder_cont: List[str] = [],
        embedding_sizes: Dict[str, Tuple[int, int]] = {},
        x_categoricals: List[str] = [],
        output_size=1,
    ):

        super().__init__()
        self._targets = targets
        self._encoder_cont = encoder_cont
        self._decoder_cont = decoder_cont
        self.dropout = dropout
        self.backcast_loss_ratio = backcast_loss_ratio
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.target_number = output_size
        self.covariate_number = covariate_number

        self.encoder_embeddings = MultiEmbedding(
            embedding_sizes=embedding_sizes,
            embedding_paddings=[],
            categorical_groups={},
            x_categoricals=x_categoricals,
        )

        if model_type == "I":
            width = [2**width, 2 ** (width + 2)]
            self.stack_types = ["trend", "seasonality"] * num_stack
            self.expansion_coefficient_lengths = [item for i in range(num_stack) for item in [3, 7]]
            self.num_blocks = [num_block for i in range(2 * num_stack)]
            self.num_block_layers = [num_block_layer for i in range(2 * num_stack)]
            self.widths = [item for i in range(num_stack) for item in width]
        elif model_type == "G":
            self.stack_types = ["generic"] * num_stack
            self.expansion_coefficient_lengths = [2**expansion_coe for i in range(num_stack)]
            self.num_blocks = [num_block for i in range(num_stack)]
            self.num_block_layers = [num_block_layer for i in range(num_stack)]
            self.widths = [2**width for i in range(num_stack)]
        #
        # setup stacks
        self.net_blocks = nn.ModuleList()

        for stack_id, stack_type in enumerate(self.stack_types):
            for _ in range(self.num_blocks[stack_id]):
                if stack_type == "generic":
                    net_block = NBEATSGenericBlock(
                        units=self.widths[stack_id],
                        thetas_dim=self.expansion_coefficient_lengths[stack_id],
                        num_block_layers=self.num_block_layers[stack_id],
                        backcast_length=self.context_length,
                        forecast_length=self.prediction_length,
                        dropout=self.dropout,
                        tar_num=self.target_number,
                        cov_num=self.covariate_number + self.encoder_embeddings.total_embedding_size(),
                        tar_pos=self.target_positions,
                    )
                elif stack_type == "seasonality":
                    net_block = NBEATSSeasonalBlock(
                        units=self.widths[stack_id],
                        num_block_layers=self.num_block_layers[stack_id],
                        backcast_length=self.context_length,
                        forecast_length=self.prediction_length,
                        min_period=self.expansion_coefficient_lengths[stack_id],
                        dropout=self.dropout,
                        tar_num=self.target_number,
                        cov_num=self.covariate_number + self.encoder_embeddings.total_embedding_size(),
                        tar_pos=self.target_positions,
                    )
                elif stack_type == "trend":
                    net_block = NBEATSTrendBlock(
                        units=self.widths[stack_id],
                        thetas_dim=self.expansion_coefficient_lengths[stack_id],
                        num_block_layers=self.num_block_layers[stack_id],
                        backcast_length=self.context_length,
                        forecast_length=self.prediction_length,
                        dropout=self.dropout,
                        tar_num=self.target_number,
                        cov_num=self.covariate_number + self.encoder_embeddings.total_embedding_size(),
                        tar_pos=self.target_positions,
                    )
                else:
                    raise ValueError(f"Unknown stack type {stack_type}")

                self.net_blocks.append(net_block)

    @property
    def target_positions(self):
        return [self._encoder_cont.index(tar) for tar in self._targets]

    def forward(self, x: Dict[str, torch.Tensor], **kwargs) -> Dict[str, torch.Tensor]:
        """
        Pass forward of network.

        Args:
            x (Dict[str, torch.Tensor]): input from dataloader generated from
                :py:class:`~pytorch_forecasting.data.timeseries.TimeSeriesDataSet`.

        Returns:
            Dict[str, torch.Tensor]: output of model
        """
        # batch_size * look_back * features
        encoder_cont = torch.cat(
            [x["encoder_cont"], x["encoder_time_features"], x["encoder_lag_features"]], dim=-1
        )
        # `target` can only be continuous, so position inside `encoder_cat` is irrelevant
        encoder_cat = (
            torch.cat([v for _, v in self.encoder_embeddings(x["encoder_cat"]).items()], dim=-1)
            if self.encoder_embeddings.total_embedding_size() != 0
            else torch.zeros(
                encoder_cont.size(0),
                self.context_length,
                self.encoder_embeddings.total_embedding_size(),
                device=encoder_cont.device,
            )
        )
        # self.hparams.prediction_length=gap+real_predict
        timesteps = self.context_length + self.prediction_length
        # encoder_cont.size(2) + self.encoder_embeddings.total_embedding_size(),
        generic_forecast = [
            torch.zeros(
                (encoder_cont.size(0), timesteps, len(self.target_positions)),
                dtype=torch.float32,
                device=encoder_cont.device,
            )
        ]

        trend_forecast = [
            torch.zeros(
                (encoder_cont.size(0), timesteps, len(self.target_positions)),
                dtype=torch.float32,
                device=encoder_cont.device,
            )
        ]
        seasonal_forecast = [
            torch.zeros(
                (encoder_cont.size(0), timesteps, len(self.target_positions)),
                dtype=torch.float32,
                device=encoder_cont.device,
            )
        ]

        forecast = torch.zeros(
            (encoder_cont.size(0), self.prediction_length, len(self.target_positions)),
            dtype=torch.float32,
            device=encoder_cont.device,
        )

        # make sure `encoder_cont` is followed by `encoder_cat`

        backcast = torch.cat([encoder_cont, encoder_cat], dim=-1)

        for i, block in enumerate(self.net_blocks):
            # evaluate block
            backcast_block, forecast_block = block(backcast)
            # add for interpretation
            full = torch.cat([backcast_block.detach(), forecast_block.detach()], dim=1)
            if isinstance(block, NBEATSTrendBlock):
                trend_forecast.append(full)
            elif isinstance(block, NBEATSSeasonalBlock):
                seasonal_forecast.append(full)
            else:
                generic_forecast.append(full)
            # update backcast and forecast
            backcast = backcast.clone()
            backcast[..., self.target_positions] = backcast[..., self.target_positions] - backcast_block
            # do not use backcast -= backcast_block as this signifies an inline operation
            forecast = forecast + forecast_block

        # `encoder_cat` always at the end of sequence, so it will not affect `self.target_positions`
        # backcast, forecast is of batch_size * context_length/prediction_length * tar_num
        return {
            "prediction": forecast,
            "backcast": (
                encoder_cont[..., self.target_positions] - backcast[..., self.target_positions],
                self.backcast_loss_ratio,
            ),
        }

    @classmethod
    def from_dataset(cls, dataset: TimeSeriesDataset, **kwargs) -> "NbeatsNetwork":
        """

        Args:
            dataset (TimeSeriesDataSet): dataset where sole predictor is the target.
            **kwargs: additional arguments to be passed to ``__init__`` method.

        Returns:
            NBeats
        """
        # assert dataset.max_encoder_length%dataset.max_prediction_length==0 and dataset.max_enco
        # der_length<=10*dataset.max_prediction_length,"look back length should be 1-10 t
        # imes of prediction length"
        desired_embedding_sizes = kwargs.pop("embedding_sizes", {})
        embedding_sizes = {}
        for k, v in dataset.embedding_sizes.items():
            if k in dataset.encoder_cat:
                embedding_sizes[k] = v
        for name, size in desired_embedding_sizes.items():
            cat_size, _ = embedding_sizes[name]
            embedding_sizes[name] = (cat_size, size)

        return cls(
            dataset.targets,
            encoder_cont=dataset.encoder_cont + dataset.time_features + dataset.encoder_lag_features,
            decoder_cont=dataset.decoder_cont + dataset.time_features + dataset.decoder_lag_features,
            embedding_sizes=embedding_sizes,
            # only for cont, cat will be added in __init__
            covariate_number=len(dataset.encoder_cont + dataset.time_features + dataset.encoder_lag_features)
            - dataset.n_targets,
            x_categoricals=dataset.categoricals,
            context_length=dataset.get_parameters().get("indexer").get("params").get("look_back"),
            prediction_length=dataset.get_parameters().get("indexer").get("params").get("look_forward"),
            **kwargs,
        )


if __name__ == "__main__":
    test_data = Iterator(  # just example, use your data to replace;
        pd.DataFrame(
            columns=[
                "cont_fea1",
                "cont_fea2",
                "cont_fea3",
                "cont_fea4",
                "cont_fea5",
                "cat_fea1",
                "cat_fea2",
                "label",
            ]
        )
    )

    def __getitem__(self, idx: int):
        index = self._indexer[idx]
        encoder_idx = index["encoder_idx"]
        decoder_idx = index["decoder_idx"]
        # to filter data dont belong to current group
        encoder_idx_range = index["encoder_idx_range"]
        decoder_idx_range = index["decoder_idx_range"]
        encoder_period: pd.DataFrame = self._data[self._data[TIME_IDX].isin(encoder_idx_range)].loc[
            encoder_idx
        ]
        decoder_period: pd.DataFrame = self._data[self._data[TIME_IDX].isin(decoder_idx_range)].loc[
            decoder_idx
        ]

        # TODO 缺失值是个值得研究的主题
        encoder_cont = torch.tensor(encoder_period[self.encoder_cont].to_numpy(np.float64), dtype=torch.float)
        encoder_cat = torch.tensor(encoder_period[self.encoder_cat].to_numpy(np.int64), dtype=torch.int)
        encoder_time_idx = encoder_period[TIME_IDX]
        time_idx_start = encoder_time_idx.min()
        encoder_time_idx = torch.tensor(
            (encoder_time_idx - time_idx_start).to_numpy(np.int64), dtype=torch.long
        )
        encoder_target = torch.tensor(encoder_period[self.targets].to_numpy(np.float64), dtype=torch.float)
        encoder_time_features = torch.tensor(
            encoder_period[self.time_features].to_numpy(np.float64), dtype=torch.float
        )
        encoder_lag_features = torch.tensor(
            encoder_period[self.encoder_lag_features].to_numpy(np.float64),
            dtype=torch.float,
        )

        decoder_cont = torch.tensor(decoder_period[self.decoder_cont].to_numpy(np.float64), dtype=torch.float)
        decoder_cat = torch.tensor(decoder_period[self.decoder_cat].to_numpy(np.float64), dtype=torch.int)
        decoder_time_idx = decoder_period[TIME_IDX]
        decoder_time_idx = torch.tensor(
            (decoder_time_idx - time_idx_start).to_numpy(np.int64), dtype=torch.long
        )

        decoder_target = torch.tensor(decoder_period[self.targets].to_numpy(np.float64), dtype=torch.float)
        decoder_time_features = torch.tensor(
            decoder_period[self.time_features].to_numpy(np.float64), dtype=torch.float
        )
        decoder_lag_features = torch.tensor(
            decoder_period[self.decoder_lag_features].to_numpy(np.float64),
            dtype=torch.float,
        )
        targets = torch.tensor(decoder_period[self.targets].to_numpy(np.float64), dtype=torch.float)

        target_scales = torch.stack(
            [
                torch.tensor(self._target_normalizers[i].get_norm(decoder_period), dtype=torch.float)
                for i, _ in enumerate(self.targets)
            ],
            dim=-1,
        )
        target_scales_back = torch.stack(
            [
                torch.tensor(self._target_normalizers[i].get_norm(encoder_period), dtype=torch.float)
                for i, _ in enumerate(self.targets)
            ],
            dim=-1,
        )

        return (
            dict(  # batch[0]
                encoder_cont=encoder_cont,
                encoder_cat=encoder_cat,
                encoder_time_idx=encoder_time_idx,
                encoder_idx=torch.tensor(encoder_idx, dtype=torch.long),
                encoder_idx_range=torch.tensor(encoder_idx_range, dtype=torch.long),
                encoder_target=encoder_target,
                encoder_time_features=encoder_time_features,
                encoder_lag_features=encoder_lag_features,
                decoder_cont=decoder_cont,
                decoder_cat=decoder_cat,
                decoder_time_idx=decoder_time_idx,
                decoder_idx=torch.tensor(decoder_idx, dtype=torch.long),
                decoder_idx_range=torch.tensor(decoder_idx_range, dtype=torch.long),
                decoder_target=decoder_target,
                decoder_time_features=decoder_time_features,
                decoder_lag_features=decoder_lag_features,
                target_scales=target_scales,
                target_scales_back=target_scales_back,
            ),
            targets,  # batch[1]
        )

    def _collate_fn(self, batches):
        encoder_cont = torch.stack([batch[0]["encoder_cont"] for batch in batches])
        encoder_cat = torch.stack([batch[0]["encoder_cat"] for batch in batches])
        encoder_time_idx = torch.stack([batch[0]["encoder_time_idx"] for batch in batches])
        encoder_idx = torch.stack([batch[0]["encoder_idx"] for batch in batches])
        encoder_idx_range = torch.stack([batch[0]["encoder_idx_range"] for batch in batches])
        encoder_target = torch.stack([batch[0]["encoder_target"] for batch in batches])
        encoder_length = torch.tensor([len(batch[0]["encoder_target"]) for batch in batches])

        decoder_cont = torch.stack([batch[0]["decoder_cont"] for batch in batches])
        decoder_cat = torch.stack([batch[0]["decoder_cat"] for batch in batches])
        decoder_time_idx = torch.stack([batch[0]["decoder_time_idx"] for batch in batches])
        decoder_idx = torch.stack([batch[0]["decoder_idx"] for batch in batches])
        decoder_idx_range = torch.stack([batch[0]["decoder_idx_range"] for batch in batches])
        decoder_target = torch.stack([batch[0]["decoder_target"] for batch in batches])
        decoder_length = torch.tensor([len(batch[0]["decoder_target"]) for batch in batches])

        target_scales = torch.stack([batch[0]["target_scales"] for batch in batches])
        target_scales_back = torch.stack([batch[0]["target_scales_back"] for batch in batches])
        targets = torch.stack([batch[1] for batch in batches])
        encoder_time_features = torch.stack([batch[0]["encoder_time_features"] for batch in batches])
        decoder_time_features = torch.stack([batch[0]["decoder_time_features"] for batch in batches])
        encoder_lag_features = torch.stack([batch[0]["encoder_lag_features"] for batch in batches])
        decoder_lag_features = torch.stack([batch[0]["decoder_lag_features"] for batch in batches])
        return (
            dict(
                encoder_cont=encoder_cont,
                encoder_cat=encoder_cat,
                encoder_time_idx=encoder_time_idx,
                encoder_idx=encoder_idx,
                encoder_idx_range=encoder_idx_range,
                encoder_target=encoder_target,
                encoder_length=encoder_length,
                decoder_cont=decoder_cont,
                decoder_cat=decoder_cat,
                decoder_time_idx=decoder_time_idx,
                decoder_idx=decoder_idx,
                decoder_idx_range=decoder_idx_range,
                decoder_target=decoder_target,
                decoder_length=decoder_length,
                target_scales=target_scales,
                target_scales_back=target_scales_back,
                encoder_time_features=encoder_time_features,
                decoder_time_features=decoder_time_features,
                encoder_lag_features=encoder_lag_features,
                decoder_lag_features=decoder_lag_features,
            ),
            dict(
                encoder_cont=encoder_cont + encoder_time_features + encoder_lag_features,
                decoder_cont=decoder_cont + decoder_time_features + decoder_lag_features,
                embedding_sizes=embedding_sizes,
                # only for cont, cat will be added in __init__
                covariate_number=len(encoder_cont + encoder_time_features + encoder_lag_features) - 1,
                x_categoricals=categoricals,
                context_length=72,
                prediction_length=24,
            ),
            targets,
        )

    # return cls(
    #         dataset.targets,
    #         encoder_cont=dataset.encoder_cont + dataset.time_features + dataset.encoder_lag_features,
    #         decoder_cont=dataset.decoder_cont + dataset.time_features + dataset.decoder_lag_features,
    #         embedding_sizes=embedding_sizes,
    #         # only for cont, cat will be added in __init__
    #         covariate_number=len(dataset.encoder_cont + dataset.time_features + dataset.encoder_lag_features)
    #         - dataset.n_targets,
    #         x_categoricals=dataset.categoricals,
    #         context_length=dataset.get_parameters().get("indexer").get("params").get("look_back"),
    #         prediction_length=dataset.get_parameters().get("indexer").get("params").get("look_forward"),
    #         **kwargs,
    #     )

    def cutomized_collet_fn(scale_boudary={}):
        # cutomize your collet_fn to adjust SimpleClassify Model
        # involve pre-process 1. encoder str-like features to numeric;
        #                     2. scale numeric features, the oveall min/max/avg/std can be accessed by `fs.stats` and transported or written in this func;
        #                     3. others if need
        # involve collect method to convert original data format to that used in SimpleClassify.forward and loss calculation
        pass

    test_data_loader = DataLoader(  # `batch_siz`e and `drop_last`` do not matter now, `sampler`` set it to be None cause `test_data`` is a Iterator
        test_data, collate_fn=cutomized_collet_fn, batch_size=4, drop_last=False, sampler=None
    )
    # cont_nbr/cat_nbr means the number of continuous/categorical features delivered to model;
    # max_types means the max different types in all categorical features
    # emd_dim is a parameter do not matter now
    # model = SimpleClassify(cont_nbr=5, cat_nbr=2, emd_dim=2, max_types=6)
    model = NbeatsNetwork()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # no need to change
    loss_fn = nn.BCELoss()  # loss function to train a classification model

    for epoch in range(10):  # assume 10 epoch
        print(f"epoch: {epoch} begin")
        pred_label = model(test_data_loader)
        true_label = test_data["labels"]
        loss = loss_fn(pred_label, true_label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f"epoch: {epoch} done, loss: {loss}")
