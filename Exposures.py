import pandas as pd
import polars as pl
import numpy as np

import plotly.graph_objects as go
from plotly.subplots import make_subplots

class Exposures():
    def __init__(self,
                 base_df:pd.DataFrame,
                 trade_date_col:str,
                 symbol_col:str,
                 price_col:str,
                 factor_name:str,
                 exposure_cols:list,
                 rebalance_period:int = 1,
                 bins:int = 5,
                 position:str = 'ls',
                 overnight: str = "on",
                 fac_shift: int | None = None):
        
        self.price_col = price_col 
        self.trade_date_col = trade_date_col
        self.symbol_col = symbol_col
        self.factor_name = factor_name
        self.overnight = overnight
        self.fac_shift = fac_shift
        self.exposure_cols = exposure_cols
        self.rebalance_period = rebalance_period
        self.bins = bins
        self.position = position
        
        if isinstance(self.base_df,pd.DataFrame):
            self.base_df= pl.from_pandas(base_df).drop_nulls()
        else:
            self.base_df = base_df.drop_nulls()
        self.base_df:pl.DataFrame = self.base_df.with_columns(pl.col(self.trade_date_col).cast(pl.Datetime)).sort([self.symbol_col,self.trade_date_col])
        self.td = self.base_df[self.trade_date_col][1] - self.base_df[self.trade_date_col][0]
        
        if self.fac_shift:
            self.base_df = self.base_df.with_columns(
                pl.col([self.factor_name] + exposure_cols).shift(self.fac_shift).over(self.symbol_col)
            )
        
        if self.bins < 2:
            raise ValueError(f"bins must be >= 2, got {self.bins}")
        
    def calc_stats(self):
        def _cross_section_ols(sub_df, factor_cols):
            y = sub_df["fut_ret"].to_numpy()
            X = sub_df.select(factor_cols).to_numpy()

            X = np.column_stack([np.ones(len(X)), X])

            XtX = X.T @ X
            XtX_inv = np.linalg.pinv(XtX)

            beta = XtX_inv @ (X.T @ y)

            return beta
        df = self.base_df.clone()
        
        df = df.with_columns([
            pl.when(pl.col(self.price_col) == 0).then(None).otherwise(pl.col(self.price_col)).alias(self.price_col),
            pl.when(pl.col(self.price_col).shift(-1).over(self.symbol_col) == 0)
            .then(None)
            .otherwise(pl.col(self.price_col).shift(-1).over(self.symbol_col))
            .alias("price_fut")
        ])
        df = df.with_columns(
            ((pl.col("price_fut") / pl.col(self.price_col)) - 1).alias("fut_ret")
        )
        df = df.with_columns(
            pl.when(pl.col("fut_ret") == -1).then(None).otherwise(pl.col("fut_ret")).alias("fut_ret")
        )
        
        all_dates = df[self.trade_date_col].unique().sort()
        rebalance_dates = all_dates[::self.rebalance_period]
        rebalance_mask = pl.col(self.trade_date_col).is_in(
            rebalance_dates.implode()
        )
        
        df = df.with_columns(
            pl.col(self.trade_date_col).max()
            .over([self.symbol_col, pl.col(self.trade_date_col).dt.date()])
            .alias("day_last_dt"),

            (pl.col(self.trade_date_col) + self.rebalance_period * self.td).alias("target_dt")
        )

        df = df.with_columns(
            (pl.col("target_dt") > pl.col("day_last_dt"))
            .alias("is_overnight")
        )

        if self.overnight == "on":
            mask = rebalance_mask

        elif self.overnight == "off":
            mask = (~pl.col("is_overnight")) | rebalance_mask

        elif self.overnight == "only":
            mask = (pl.col("is_overnight")) | rebalance_mask

        df = df.filter(mask).select(
            [self.trade_date_col, self.symbol_col, self.factor_name, "fut_ret"] + self.exposure_cols
        )           
        
        df = df.with_columns([
            (
                (pl.col(f) - pl.col(f).mean().over(self.trade_date_col))
                / pl.col(f).std().over(self.trade_date_col)
            ).alias(f)
            for f in self.exposure_cols + [self.factor_name]
        ])
        
        df = df.with_columns([
            pl.col(self.factor_name).rank("average", descending=True).over(self.trade_date_col).alias("rank"),
            pl.len().over(self.trade_date_col).alias("n_stocks"),
        ])

        df = df.sort([self.trade_date_col, "rank"])  
        df = df.with_columns([
            pl.arange(0, pl.len(), eager=False).over(self.trade_date_col).alias("pos_index"),
        ])

        df = df.with_columns([
            (
                ((pl.col("pos_index") * self.bins) / pl.col("n_stocks")).floor() + 1
            ).cast(pl.Int32).alias("quantile_temp")
        ])

        df = df.with_columns(
            pl.when(pl.col("quantile_temp") > self.bins)
            .then(self.bins)
            .otherwise(pl.col("quantile_temp"))
            .alias("quantile")
        ).drop("quantile_temp").drop("pos_index").filter(pl.col(self.factor_name).is_not_null())
        df = df.drop_nulls(["fut_ret"])
        
        if self.position == 'ls':
            mean_df = df.filter((pl.col("quantile") == 1) | (pl.col("quantile") == self.bins))
            mean_df = mean_df.with_columns(
                    [
                        pl.when(pl.col("quantile") == self.bins)
                        .then(-pl.col(f))
                        .otherwise(pl.col(f))
                        .alias(f)
                        for f in self.exposure_cols
                    ]
                )
            mean_df = (
                mean_df
                .group_by(self.trade_date_col)
                .agg([
                    pl.col(f).mean().alias(f"{f}_expo")
                    for f in self.exposure_cols
                ])
                .sort(self.trade_date_col)
            )
        elif self.position == 'l':
            mean_df = df.filter((pl.col("quantile") == 1))
            mean_df = (
                mean_df
                .group_by(self.trade_date_col)
                .agg([
                    pl.col(f).mean().alias(f"{f}_expo")
                    for f in self.exposure_cols
                ])
                .sort(self.trade_date_col)
            )
        elif self.position == 's':
            mean_df = df.filter((pl.col("quantile") == self.bins))
            mean_df = mean_df.with_columns(
                    [
                        pl.when(pl.col("quantile") == self.bins)
                        .then(-pl.col(f))
                        for f in self.exposure_cols
                    ]
                )
            mean_df = (
                mean_df
                .group_by(self.trade_date_col)
                .agg([
                    pl.col(f).mean().alias(f"{f}_expo")
                    for f in self.exposure_cols
                ])
                .sort(self.trade_date_col)
            )
        else:
            raise 
            
        betas = []
        for dt, sub_df in df.group_by(self.trade_date_col):
            beta = _cross_section_ols(sub_df, self.exposure_cols)
            dt = sub_df[self.trade_date_col][0]
            betas.append(
                [dt] + beta.tolist()
            )

        beta_df = pl.DataFrame(
            betas,
            schema=[self.trade_date_col, "intercept"] + self.exposure_cols,
            orient="row"
        )
        
        result_df = mean_df.join(
            beta_df,
            on=self.trade_date_col,
            how="inner"
        )
        result_df = result_df.fill_null(0)
        result_df = result_df.sort(self.trade_date_col)
        result_df = (
            result_df
            .sort(self.trade_date_col)
            .with_columns(
                [
                    (pl.col(f"{f}_expo") * pl.col(f)).alias(f"{f}_attr")
                    for f in self.exposure_cols
                ]
            )
            .with_columns(
                [
                    (
                        (1 + pl.col(f"{f}_attr"))
                        .cum_prod()
                        - 1
                    ).alias(f"{f}_cum_ret")
                    for f in self.exposure_cols
                ]
            )
            .with_columns(
                [
                    (
                        (1 + pl.col("intercept"))
                        .cum_prod()
                        - 1
                    ).alias("Alpha")
                ]
            )
        )
        return result_df

    def run(self):
        self.result_df = self.calc_stats()
        
    def plot_exposures(self, staticPlot: bool = False, return_fig: bool = False):
        expo_cols = [col for col in self.result_df.columns if col.endswith("_expo")]
        fig = go.Figure()

        for col in expo_cols:
            fig.add_trace(
                go.Scatter(
                    x=self.result_df[self.trade_date_col],
                    y=self.result_df[col],
                    mode="lines",
                    name=col,
                    marker=dict(size=6),
                    line=dict(width=2)
                )
            )
        
        fig.update_layout(
            title=dict(
                text=f"Exposures Over Time ({self.factor_name})",
                font=dict(
                    size=24,          
                    family="Arial",  
                    color="black"    
                ),),
            xaxis_title="Date",
            yaxis_title="Exposure",
            template="plotly_white",
            legend=dict(
                font=dict(size=16)
            ),height=550
            
        )
        fig.update_xaxes(
                type="category",
                categoryorder="array",
                categoryarray=self.result_df[self.trade_date_col],
                title_text="Date",
                title_font=dict(size=18),
                tickfont=dict(size=8)
            )
        fig.update_yaxes(title_text="Exposure", title_font=dict(size=18), tickfont=dict(size=16))
        if staticPlot:
            fig.show(config={"staticPlot": True, "responsive": True})
        else:
            fig.show(config={"responsive": True})

        if return_fig:
            return fig
    
    def plot_returns(self, staticPlot: bool = False, return_fig: bool = False):
        expo_cols = [col for col in self.result_df.columns if col.endswith("_cum_ret") or col.endswith("Alpha")] 
        fig = go.Figure()

        for col in expo_cols:
            fig.add_trace(
                go.Scatter(
                    x=self.result_df[self.trade_date_col],
                    y=self.result_df[col],
                    line=dict(width=2),
                    mode="lines",
                    name=col
                )
            )

        fig.update_layout(
            title=dict(
                text=f"Cummulative Returns of Exposures over time ({self.factor_name})",
                font=dict(
                    size=24,          
                    family="Arial",  
                    color="black"    
                ),),
            xaxis_title="Date",
            yaxis_title="Exposure",
            template="plotly_white",
            legend=dict(
                font=dict(size=16)
            ),height=550
        )
        
        fig.update_xaxes(
                type="category",
                categoryorder="array",
                categoryarray=self.result_df[self.trade_date_col],
                title_text="Date",
                title_font=dict(size=18),
                tickfont=dict(size=8)
            )
        fig.update_yaxes(title_text="Exposure", title_font=dict(size=18), tickfont=dict(size=16))
        
        
        if staticPlot:
            fig.show(config={"staticPlot": True, "responsive": True})
        else:
            fig.show(config={"responsive": True})

        if return_fig:
            return fig
    
    def plot_exposures_and_returns(self, staticPlot: bool = False, return_fig: bool = False):

        n_rows = len(self.exposure_cols)

        fig = make_subplots(
            rows=n_rows,
            cols=1,
            shared_xaxes=False,
            vertical_spacing=0.15,
            specs=[[{"secondary_y": True}] for _ in range(n_rows)],
            subplot_titles=self.exposure_cols
        )

        date = self.result_df[self.trade_date_col]

        for row, factor in enumerate(self.exposure_cols, start=1):

            expo_col = f"{factor}_expo"
            attr_col = f"{factor}_cum_ret"

            fig.add_trace(
                go.Scatter(
                    x=date,
                    y=self.result_df[expo_col],
                    fill="tozeroy",
                    name="Exposure" if row == 1 else None,
                    marker=dict(size=6),
                    line=dict(width=2,color='lightblue'),
                    showlegend=False
                ),
                row=row,
                col=1,
                secondary_y=False
            )

            fig.add_trace(
                go.Scatter(
                    x=date,
                    y=self.result_df[attr_col],
                    marker=dict(size=6),
                    name="Cumulative Return" if row == 1 else None,
                    line=dict(width=2,color='red'),
                    showlegend=False
                ),
                row=row,
                col=1,
                secondary_y=True
            )

            fig.update_yaxes(
                title_text=f"{factor} Exposure",
                row=row,
                col=1,
                secondary_y=False,
                title_font=dict(size=18), 
                tickfont=dict(size=16)
            )

            fig.update_yaxes(
                title_text="Cumulative Return",
                tickformat=".2f",
                row=row,
                col=1,
                secondary_y=True, 
                title_font=dict(size=18), 
                tickfont=dict(size=16)
            )

            fig.update_xaxes(
                type="category",
                categoryorder="array",
                categoryarray=date,
                title_text="Date",
                row=row, col=1,
                title_font=dict(size=18),
                tickfont=dict(size=8)
            )
        fig.update_annotations(
            font=dict(size=22, family="Arial", color="black")
        )
        height_per_row = 550

        fig.update_layout(
            template="plotly_white",
            height=n_rows * height_per_row,
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            margin=dict(t=100, b=60, l=60, r=60),
            title=dict(
                text=f"Returns & Exposures ({self.factor_name})",
                font=dict(size=24, family="Arial", color="black")
            )
        )

        if staticPlot:
            fig.show(config={"staticPlot": True, "responsive": True})
        else:
            fig.show(config={"responsive": True})

        if return_fig:
            return fig

