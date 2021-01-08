# Basis results of Binary GCN


| |Cora|CiteSeer|PubMed|
|---|---|---|---|
|Bi-GCN(16)|77.04(1.54)|64.90(2.02)|76.48(1.31)|
|Bi-GCN(32)|79.97(1.11)|67.42(1.02)|77.49(1.09)|
|Bi-GCN(64)|81.21(0.76)|68.79(0.85)|78.28(0.92)|
|Bi-GCN(96)|81.66(0.74)|69.14(0.90)|78.52(0.84)|
|Bi-GCN(128)|81.77(0.73)|69.07(0.84)|78.61(0.92)|
|Bi-GCN(256)|82.08(0.61)|68.76(0.88)|78.55(0.77)|
|Bi-GCN(512)|81.47(0.79)|66.07(1.92)|78.32(0.76)|

每个配置运行100次，取均值和样本方差。（不同的100次之间还是有差距，基本都取跑3次的均值的中位数）

# Table 2, 3 (full version)

|Cora    |Model Size| Data Size | Cycle Operations | Accuarcy |
|---     |---       |---        | ---              | ---      |
|FASTGCN | 360K     | 14.8M     | 249,954,739      |79.8 ± 0.3|
|SGC     | 39.18K   | 14.8M     | 27,239,926       |81.0 ± 0.0|
|GAT     | 360.55K  | 14.8M     | 250,774,518      |83.0 ± 0.7|
|GCN     | 360K     | 14.8M     | 249,954,739      |81.4 ± 0.4|
|Bi-GCN  | 11.53K   | 0.47M     | 4,669,515        |81.2 ± 0.8|

|CiteSeer|Model Size| Data Size | Cycle Operations | Accuarcy |
|---     |---       |---        | ---              | ---      |
|FASTGCN | 927.25K  | 40.0M     | 790,081,192      |68.8 ± 0.6|
|SGC     | 86.79K   | 40.0M     | 73,248,070       |71.9 ± 0.1|
|GAT     | 927.8K   | 40.0M     | 790,786,260      |72.5 ± 0.7|
|GCN     | 927.25K  | 40.0M     | 790,081,192      |70.9 ± 0.5|
|Bi-GCN  | 29.25K   | 1.48M     | 13,136,863       |68.8 ± 0.9|

|PubMed  |Model Size| Data Size | Cycle Operations | Accuarcy |
|---     |---       |---        | ---              | ---      |
|FASTGCN | 125.75K  | 37.6M     | 637,507,158      |77.4 ± 0.3|
|SGC     | 5.85K    | 37.6M     | 29,832,528       |78.9 ± 0.0|
|GAT     | 126.27K  | 37.6M     | 643,847,492      |79.0 ± 0.3|
|GCN     | 125.75K  | 37.6M     | 637,507,158      |79.0 ± 0.3|
|Bi-GCN  | 3.92K    | 1.25M     | 15,526,553       |78.2 ± 1.0|

|Reddit       |Model Size| Data Size | Cycle Operations | Accuarcy |
|---          |---       |---        | ---              | ---      |
|indGCN       | 643.00K  | 534.99M   | 41,795,157,663   |93.8 ± 0.1|
|Bi-indGCN    | 21.25K   | 17.61M    | 4,184,822,133      |93.1 ± 0.2|
|GraphSAGE    | 1286.00K | 534.99M   | 80,143,060,383   |95.2 ± 0.1|
|Bi-GraphSAGE | 42.51K   | 17.61M    | 4,922,389,323      |95.3 ± 0.1|
|GraphSAINT   | 1798.00K | 534.99M   | 113,173,736,448   |95.2 ± 0.2|
|Bi-GraphSAINT| 139.62K  | 17.61M    | 10,413,840,303    |95.2 ± 0.1|


|Flickr       |Model Size| Data Size | Cycle Operations | Accuarcy |
|---          |---       |---        | ---              | ---      |
|indGCN       | 507.00K  | 170.23M   | 11,820,571,828   |50.9 ± 0.3|
|Bi-indGCN    | 16.87K   | 5.66M     | 464,580,328      |50.2 ± 0.4|
|GraphSAGE    | 1014.00K | 170.23M   | 23,404,507,828   |50.9 ± 1.0|
|Bi-GraphSAGE | 33.74K   | 5.66M     | 692,524,828      |50.2 ± 0.4|
|GraphSAINT   | 1526.00K | 170.23M   | 35,326,723,072   |51.7 ± 0.1|
|Bi-GraphSAINT| 65.25K   | 5.66M     | 1,279,075,828    |50.7 ± 0.1|


各数据集上的压缩加速比

|Ratio        |Model Size| Data Size | Cycle Operations|
|---          |---       |---        | ---              
|Cora         | 31.23    | 31.30     | 53.53
|CiteSeer     | 31.70    | 31.73     | 60.14
|PubMed       | 30.00    | 30.08     | 41.06
|avg          | 30.98    | 31.04     | 51.58

|Ratio        |Model Size| Data Size | Cycle Operations|
|---          |---       |---        | ---              
|Reddit-gcn   | 31.23    | 30.38     | 53.53
|Reddit-sage  | 31.23    | 30.38     | 53.53
|Reddit-saint | 31.23    | 30.38     | 53.53
|Flickr       | 31.70    | 31.73     | 60.14
|avg          | 31.23    | 