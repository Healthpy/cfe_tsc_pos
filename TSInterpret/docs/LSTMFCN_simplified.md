# Simplified LSTMFCN Architecture

```mermaid
graph LR
    input[Input<br/>Time Series] --> split{{"Split"}}
    
    subgraph LSTM_Path
        split --> lstm[2-Layer LSTM<br/>128 units] 
        lstm --> last[Last Time Step]
        last --> drop1[Dropout 0.8]
    end
    
    subgraph FCN_Path
        split --> conv_block1[Conv Block 1<br/>128 filters] 
        conv_block1 --> se1[SE Layer 1<br/>128 channels]
        se1 --> conv_block2[Conv Block 2<br/>256 filters]
        conv_block2 --> se2[SE Layer 2<br/>256 channels]
        se2 --> conv_block3[Conv Block 3<br/>128 filters]
        conv_block3 --> gap[Global Avg Pool]
    end
    
    drop1 --> concat((Concatenate))
    gap --> concat
    concat --> fc[FC Layer]
    fc --> output[Output<br/>Classification]

    classDef blocks fill:#f9f,stroke:#333,stroke-width:2px
    classDef se fill:#e6fff2,stroke:#333,stroke-width:2px
    classDef path fill:#fff0e6,stroke:#333,stroke-width:1px
    
    class input,output blocks
    class se1,se2 se
    class LSTM_Path,FCN_Path path

    %% Note: Each Conv Block includes Conv1D + BatchNorm + ReLU + Dropout
    %% Each SE Layer includes Global AvgPool + FC reduction + ReLU + FC expansion + Sigmoid
```

## Key Components
1. **Dual-Path Architecture**
   - LSTM path: Sequence modeling
   - FCN path: Feature extraction with attention

2. **Conv Blocks (FCN Path)**
   - Conv1D → BatchNorm → ReLU → Dropout(0.3)
   - Filter sizes: 128 → 256 → 128

3. **SE Layers**
   - Channel attention mechanism
   - Reduction ratio = 16
   - Applied after first two conv blocks
