                # ERA5 Temperature Modeling Project

---

## Project Overview - Predicting Surface Temperature with AI

### **Project Goal**
**Use atmospheric data to predict 2-meter air temperature (t2m) patterns with machine learning**

### **What is ERA5?**
- **ERA5** = European Centre for Medium-Range Weather Forecasts Reanalysis v5
- Global atmospheric reanalysis dataset covering 1940-present
- Combines observations with weather models to create complete atmospheric state
- **Resolution**: ~31km globally, hourly temporal resolution
- **Gold standard** for atmospheric research and ML applications

### **Our Dataset**
- **Region**: US Midwest (agricultural region prone to extreme weather)
- **Time Period**: 2020-2023 (4 years of data)
- **Target**: Predict 2-meter air temperature (`t2m`)
- **Features**: atmospheric variables

### **Why This Matters**
- **Agriculture**: Growing season planning and frost protection
- **Energy Management**: Heating/cooling demand forecasting
- **Climate Research**: Understanding temperature patterns and extremes in continental regions

---

## Input Features - The Atmospheric "Recipe" for Temperature

### **Primary Input Features** (Essential for US Midwest Temperature)

#### **Pressure Pattern Indicators**
- **Geopotential Heights**: `z_1000`, `z_600`, `z_200`
  - **Physical Role**: Define high/low pressure systems, jet stream position
  - **Why Critical**: Control air mass movement and temperature advection across the Great Plains

#### **Wind Transport**
- **Surface Winds**: `u10`, `v10` - 10-meter wind components
  - **Physical Role**: Local temperature advection and mixing
- **Upper-Level Winds**: `u_800`, `u_600` - Mid-level wind components
  - **Physical Role**: Storm system movement and warm/cold air mass transport

#### **Moisture Content**
- **Humidity Profile**: `q_1000`, `q_800`, `q_600`
  - **Physical Role**: Affects radiative processes and heat capacity
  - **Why Important**: Humid air has different thermal properties than dry air

#### **Geographic Context**
- **Land-Sea Mask**: `lsm`
  - **Physical Role**: Distinguishes Great Lakes from land areas
  - **Why Important**: Great Lakes moderate local temperatures

### **Secondary Features** (Radiative Effects)
- **Cloud Water Content**: `clwc_800`, `clwc_600` (liquid water clouds)
- **Ice Cloud Content**: `ciwc_800`, `ciwc_600` (ice clouds)
- **Physical Role**: Cloud cover affects solar radiation and nighttime cooling

### **Features EXCLUDED to Avoid Data Leakage**
**Temperature at pressure levels** (`t_800`, `t_600`, `t_400`) - Too similar to target  
**Skin temperature** (`skt`) - Directly correlated with t2m  
**Sea surface temperature** (`sst`) - Minimal oceanic influence in continental Midwest

---

## Physical Understanding & Technical Implementation

### **Continental Climate Physics**
The US Midwest experiences **continental climate** with large temperature swings due to:

| Process | Input Variable | How It Affects t2m |
|---------|----------------|-------------------|
| **Air Mass Movement** | `z_1000`, `z_600`, `z_200` | High/low pressure systems bring warm/cold air |
| **Advection** | `u10`, `v10`, `u_800`, `u_600` | Winds transport temperature across the plains |
| **Radiative Balance** | `clwc_*`, `ciwc_*` | Clouds affect heating/cooling rates |
| **Moisture Effects** | `q_1000`, `q_800`, `q_600` | Humidity influences heat capacity |
| **Lake Effects** | `lsm` | Great Lakes moderate extreme temperatures |

### **Data Pipeline**
```python
# Our t2m prediction setup:
Input Shape:  [batch_size, 12_channels, lat, lon]  # Selected atmospheric features
Output Shape: [batch_size, 1_channel, lat, lon]    # 2-meter temperature map
```

###  **Technical Features**
- **Unified Dataset**: Surface + pressure level data in single files
- **Efficient Loading**: Zarr format with Dask for large datasets
- **PyTorch Integration**: Ready for deep learning workflows
- **Flexible Channels**: Select specific atmospheric variables
- **Visualization Tools**: Compare inputs/outputs, create animations

### **Data Characteristics**
- **Spatial**: 53×97 grid covering US Midwest
- **Temporal**: 6-hour intervals (4 samples per day)
- **Training**: 2020-2022
- **Validation**: 2023

### **Learning Objectives for Students**
1. **Understand** how atmospheric variables relate to temperature
2. **Experience** working with real-world meteorological datasets
3. **Apply** deep learning to scientific prediction problems
4. **Visualize** and interpret high-dimensional atmospheric data


### **Model Considerations**
- **Model Architecture**:  U-Net or Transformer-based models
- **Loss Function**: MSE with potential weighting for extreme temperatures
- **Evaluation**: RMSE, temperature gradient accuracy, seasonal skill
- **Challenge**: Capturing both large-scale patterns and local effects