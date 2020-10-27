import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor

from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
import pandas as pd
import datetime as dt
import numpy as np

# load data
fig_num = 1
file = 'Figura_de_control_desde_feb_fig' + str(fig_num) + '.xlsx'

data = pd.read_excel(file, usecols=[0,1], names=['times', 'defs'])

try:
    times = np.array([dt.datetime.timestamp(x) for x in data['times']])
except:
    times = np.array(data['times'])

# Convert times from seconds to days
times = (times/(3600*24) -
         (times/(3600*24))[0])

defs = np.array(data['defs'])

# define dataset
max_encode_length = 36
max_prediction_length = 6

training = TimeSeriesDataSet(
    data,
    time_idx='times',
    target='defs',
    # weight="weight",
    group_ids=[],
    max_encode_length=max_encode_length,
    max_prediction_length=max_prediction_length,
    static_categoricals=[],
    static_reals=[],
    time_varying_known_categoricals=[],
    time_varying_known_reals=['times'],
    time_varying_unknown_categoricals=[],
    time_varying_unknown_reals=['defs'],
)

# create validation and training dataset
validation = TimeSeriesDataSet.from_dataset(training, data, min_prediction_idx=training.index.time.max() + 1, stop_randomization=True)
batch_size = 128
train_dataloader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=2)
val_dataloader = validation.to_dataloader(train=False, batch_size=batch_size, num_workers=2)

# define trainer with early stopping
early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=1, verbose=False, mode="min")
lr_logger = LearningRateMonitor()
trainer = pl.Trainer(
    max_epochs=100,
    gpus=1,
    gradient_clip_val=0.1,
    limit_train_batches=30,
    callbacks=[lr_logger, early_stop_callback],
)

# create the model
tft = TemporalFusionTransformer.from_dataset(
    training,
    learning_rate=0.03,
    hidden_size=32,
    attention_head_size=1,
    dropout=0.1,
    hidden_continuous_size=16,
    output_size=7,
    loss=QuantileLoss(),
    log_interval=2,
    reduce_on_plateau_patience=4
)
print(f"Number of parameters in network: {tft.size()/1e3:.1f}k")

# find optimal learning rate (set limit_train_batches to 1.0 and log_interval = -1)
res = trainer.tuner.lr_find(
    tft, train_dataloader=train_dataloader, val_dataloaders=val_dataloader, early_stop_threshold=1000.0, max_lr=0.3,
)

print(f"suggested learning rate: {res.suggestion()}")
fig = res.plot(show=True, suggest=True)
fig.show()

# fit the model
trainer.fit(
    tft, train_dataloader=train_dataloader, val_dataloaders=val_dataloader,
)
