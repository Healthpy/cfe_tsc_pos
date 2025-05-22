# LSTMFCN Architecture

```mermaid
graph TD
    input[Input<br/>shape: batch × features × seq_length] --> split

    subgraph LSTM_Branch
        split --> transpose[Transpose<br/>shape: batch × seq_length × features]
        transpose --> lstm1[LSTM Layer 1<br/>units: 128]
        lstm1 --> lstm2[LSTM Layer 2<br/>units: 128]
        lstm2 --> last_step[Last Time Step]
        last_step --> lstm_drop[Dropout 0.8]
    end

    subgraph FCN_Branch
        split --> conv1[Conv1D<br/>128 filters, k=8]
        conv1 --> bn1[BatchNorm]
        bn1 --> relu1[ReLU]
        relu1 --> se1[SE Layer]
        se1 --> drop1[Dropout 0.3]

        drop1 --> conv2[Conv1D<br/>256 filters, k=5]
        conv2 --> bn2[BatchNorm]
        bn2 --> relu2[ReLU]
        relu2 --> se2[SE Layer]
        se2 --> drop2[Dropout 0.3]

        drop2 --> conv3[Conv1D<br/>128 filters, k=3]
        conv3 --> bn3[BatchNorm]
        bn3 --> relu3[ReLU]
        relu3 --> drop3[Dropout 0.3]
        drop3 --> gap[Global Avg Pool]
    end

    lstm_drop --> concat{Concatenate}
    gap --> concat
    concat --> fc[Fully Connected]
    fc --> softmax[Log Softmax]
    softmax --> output[Output<br/>shape: batch × classes]

    style input fill:#f9f,stroke:#333,stroke-width:2px
    style output fill:#f9f,stroke:#333,stroke-width:2px
    style LSTM_Branch fill:#e6f3ff,stroke:#333,stroke-width:2px
    style FCN_Branch fill:#fff0e6,stroke:#333,stroke-width:2px
```

## Architecture Details

The LSTMFCN model combines two parallel processing branches:

1. **LSTM Branch**:
   - Two-layer LSTM with 128 units
   - Takes the last time step output
   - Applies dropout (0.8)

2. **FCN Branch**:
   - Three Conv1D layers with increasing then decreasing filters
   - Each conv layer followed by BatchNorm and ReLU
   - Two Squeeze-and-Excitation (SE) layers after first two convolutions
   - Dropout (0.3) after each block
   - Global Average Pooling at the end

The outputs from both branches are concatenated and passed through a final fully connected layer with log softmax activation.
