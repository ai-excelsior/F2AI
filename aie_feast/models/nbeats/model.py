import torch
import pandas as pd
from torch import nn
from torchmetrics import MeanAbsoluteError
from typing import List, Dict, Tuple
from torch.utils.data import DataLoader
from aie_feast.featurestore import FeatureStore
from aie_feast.common.sampler import GroupFixednbrSampler
from aie_feast.common.collecy_fn import nbeats_collet_fn

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


if __name__ == "__main__":

    fs = FeatureStore("file:///Users/xuyizhou/Desktop/xyz_warehouse/gitlab/guizhou_traffic")
    
    dataset = fs.get_dataset(
        service_name="traval_time_prediction_embedding_v1",
        sampler=GroupFixednbrSampler( 
            time_bucket="10 days",
            stride=1,
            group_ids=None,
            group_names=None,
            start="2020-08-01",
            end="2021-09-30",  #set paras
        ),
    )

    features_cat = [
        fea
        for fea in fs._get_available_features(fs.services["traval_time_prediction_embedding_v1"])
        if fea not in fs._get_available_features(fs.services["traval_time_prediction_embedding_v1"], True)
    ]
    cat_unique = fs.stats(
        fs.services["traval_time_prediction_embedding_v1"],
        fn="unique",
        group_key=[],
        start="2020-08-01",
        end="2021-09-30",
        features=features_cat,
    ).to_dict()
    cat_count = {key: len(cat_unique[key]) for key in cat_unique.keys()}
    cont_scalar_max = fs.stats(
        fs.services["traval_time_prediction_embedding_v1"], fn="max", group_key=[], start="2020-08-01", end="2021-09-30"
    ).to_dict()
    cont_scalar_min = fs.stats(
        fs.services["traval_time_prediction_embedding_v1"], fn="min", group_key=[], start="2020-08-01", end="2021-09-30"
    ).to_dict()
    cont_scalar = {key: [cont_scalar_min[key], cont_scalar_max[key]] for key in cont_scalar_min.keys()}
    label = fs._get_available_labels(fs.services["traval_time_prediction_embedding_v1"]),

    i_ds = dataset.to_pytorch()
    test_data_loader = DataLoader(  
        i_ds,
        collate_fn=lambda x: nbeats_collet_fn(
            x,
            cont_scalar=cont_scalar,
            categoricals=cat_unique,           
            label=label,
        ),
        batch_size=8,
        drop_last=False,
        sampler=None,
    )

    model = NbeatsNetwork(
        targets=label,
        prediction_length= 0,
        context_length= 0, # TODO set paras 周期的获取
        covariate_number = len(cont_scalar),
        encoder_cont = list(cont_scalar.keys()) + label,
        decoder_cont = list(cont_scalar.keys()),
        x_categoricals = features_cat,
        output_size=1,
    ) 

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  
    loss_fn = MeanAbsoluteError() 

    for epoch in range(10):  # assume 10 epoch
        print(f"epoch: {epoch} begin")
        for x, y in test_data_loader:
            pred_label = model(x)
            true_label = y
            loss = loss_fn(pred_label, true_label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"epoch: {epoch} done, loss: {loss}")

    