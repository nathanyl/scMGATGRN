# scMGATGRN
a multi-view graph attention network based method for inferring gene regulatory networks from single-cell transcriptomic data


### Requirement
- Python == 3.7.3
- Pytorch == 1.9.1
- scikit-learn==1.0.2
- numpy==1.19.3
- pandas==1.2.4
- scipy==1.7.3

### Data Preparation

```
project_base_path
└───  Dataset
      └─── Benchmark Dataset
           └───Lofgof Dataset
               └───mESC
                   └───TFs+500
                        | BL--ExpressionData.csv
                        | BL--network.csv
                        | Label.csv
                        | Target.csv
                        | TF.csv
                        
                   └───TFs+1000
                        | BL--ExpressionData.csv
                        | BL--network.csv
                        | Label.csv
                        | Target.csv
                        | TF.csv
           └───Non-Specific Dataset
               └───hESC
                   └───TFs+500
                        | BL--ExpressionData.csv
                        | BL--network.csv
                        | Label.csv
                        | Target.csv
                        | TF.csv
                        
                   └───TFs+1000
                        | BL--ExpressionData.csv
                        | BL--network.csv
                        | Label.csv
                        | Target.csv
                        | TF.csv
               └───hHEP
                   └───TFs+500
                        | BL--ExpressionData.csv
                        | BL--network.csv
                        | Label.csv
                        | Target.csv
                        | TF.csv
                        
                   └───TFs+1000
                        | BL--ExpressionData.csv
                        | BL--network.csv
                        | Label.csv
                        | Target.csv
                        | TF.csv
               └───mDC
                   └───TFs+500
                        | BL--ExpressionData.csv
                        | BL--network.csv
                        | Label.csv
                        | Target.csv
                        | TF.csv
                        
                   └───TFs+1000
                        | BL--ExpressionData.csv
                        | BL--network.csv
                        | Label.csv
                        | Target.csv
                        | TF.csv
               └───mESC
                   └───TFs+500
                        | BL--ExpressionData.csv
                        | BL--network.csv
                        | Label.csv
                        | Target.csv
                        | TF.csv
                        
                   └───TFs+1000
                        | BL--ExpressionData.csv
                        | BL--network.csv
                        | Label.csv
                        | Target.csv
                        | TF.csv
               └───mHSC-E
                   └───TFs+500
                        | BL--ExpressionData.csv
                        | BL--network.csv
                        | Label.csv
                        | Target.csv
                        | TF.csv
                        
                   └───TFs+1000
                        | BL--ExpressionData.csv
                        | BL--network.csv
                        | Label.csv
                        | Target.csv
                        | TF.csv
               └───mHSC-GM
                   └───TFs+500
                        | BL--ExpressionData.csv
                        | BL--network.csv
                        | Label.csv
                        | Target.csv
                        | TF.csv
                        
                   └───TFs+1000
                        | BL--ExpressionData.csv
                        | BL--network.csv
                        | Label.csv
                        | Target.csv
                        | TF.csv
               └───mHSC-L
                   └───TFs+500
                        | BL--ExpressionData.csv
                        | BL--network.csv
                        | Label.csv
                        | Target.csv
                        | TF.csv
                        
                   └───TFs+1000
                        | BL--ExpressionData.csv
                        | BL--network.csv
                        | Label.csv
                        | Target.csv
                        | TF.csv
           └───Specific Dataset
               └───hESC
                   └───TFs+500
                        | BL--ExpressionData.csv
                        | BL--network.csv
                        | Label.csv
                        | Target.csv
                        | TF.csv
                        
                   └───TFs+1000
                        | BL--ExpressionData.csv
                        | BL--network.csv
                        | Label.csv
                        | Target.csv
                        | TF.csv
               └───hHEP
                   └───TFs+500
                        | BL--ExpressionData.csv
                        | BL--network.csv
                        | Label.csv
                        | Target.csv
                        | TF.csv
                        
                   └───TFs+1000
                        | BL--ExpressionData.csv
                        | BL--network.csv
                        | Label.csv
                        | Target.csv
                        | TF.csv
               └───mDC
                   └───TFs+500
                        | BL--ExpressionData.csv
                        | BL--network.csv
                        | Label.csv
                        | Target.csv
                        | TF.csv
                        
                   └───TFs+1000
                        | BL--ExpressionData.csv
                        | BL--network.csv
                        | Label.csv
                        | Target.csv
                        | TF.csv
               └───mESC
                   └───TFs+500
                        | BL--ExpressionData.csv
                        | BL--network.csv
                        | Label.csv
                        | Target.csv
                        | TF.csv
                        
                   └───TFs+1000
                        | BL--ExpressionData.csv
                        | BL--network.csv
                        | Label.csv
                        | Target.csv
                        | TF.csv
               └───mHSC-E
                   └───TFs+500
                        | BL--ExpressionData.csv
                        | BL--network.csv
                        | Label.csv
                        | Target.csv
                        | TF.csv
                        
                   └───TFs+1000
                        | BL--ExpressionData.csv
                        | BL--network.csv
                        | Label.csv
                        | Target.csv
                        | TF.csv
               └───mHSC-GM
                   └───TFs+500
                        | BL--ExpressionData.csv
                        | BL--network.csv
                        | Label.csv
                        | Target.csv
                        | TF.csv
                        
                   └───TFs+1000
                        | BL--ExpressionData.csv
                        | BL--network.csv
                        | Label.csv
                        | Target.csv
                        | TF.csv
               └───mHSC-L
                   └───TFs+500
                        | BL--ExpressionData.csv
                        | BL--network.csv
                        | Label.csv
                        | Target.csv
                        | TF.csv
                        
                   └───TFs+1000
                        | BL--ExpressionData.csv
                        | BL--network.csv
                        | Label.csv
                        | Target.csv
                        | TF.csv
           └───STRING Dataset
               └───hESC
                   └───TFs+500
                        | BL--ExpressionData.csv
                        | BL--network.csv
                        | Label.csv
                        | Target.csv
                        | TF.csv
                        
                   └───TFs+1000
                        | BL--ExpressionData.csv
                        | BL--network.csv
                        | Label.csv
                        | Target.csv
                        | TF.csv
               └───hHEP
                   └───TFs+500
                        | BL--ExpressionData.csv
                        | BL--network.csv
                        | Label.csv
                        | Target.csv
                        | TF.csv
                        
                   └───TFs+1000
                        | BL--ExpressionData.csv
                        | BL--network.csv
                        | Label.csv
                        | Target.csv
                        | TF.csv
               └───mDC
                   └───TFs+500
                        | BL--ExpressionData.csv
                        | BL--network.csv
                        | Label.csv
                        | Target.csv
                        | TF.csv
                        
                   └───TFs+1000
                        | BL--ExpressionData.csv
                        | BL--network.csv
                        | Label.csv
                        | Target.csv
                        | TF.csv
               └───mESC
                   └───TFs+500
                        | BL--ExpressionData.csv
                        | BL--network.csv
                        | Label.csv
                        | Target.csv
                        | TF.csv
                        
                   └───TFs+1000
                        | BL--ExpressionData.csv
                        | BL--network.csv
                        | Label.csv
                        | Target.csv
                        | TF.csv
               └───mHSC-E
                   └───TFs+500
                        | BL--ExpressionData.csv
                        | BL--network.csv
                        | Label.csv
                        | Target.csv
                        | TF.csv
                        
                   └───TFs+1000
                        | BL--ExpressionData.csv
                        | BL--network.csv
                        | Label.csv
                        | Target.csv
                        | TF.csv
               └───mHSC-GM
                   └───TFs+500
                        | BL--ExpressionData.csv
                        | BL--network.csv
                        | Label.csv
                        | Target.csv
                        | TF.csv
                        
                   └───TFs+1000
                        | BL--ExpressionData.csv
                        | BL--network.csv
                        | Label.csv
                        | Target.csv
                        | TF.csv
               └───mHSC-L
                   └───TFs+500
                        | BL--ExpressionData.csv
                        | BL--network.csv
                        | Label.csv
                        | Target.csv
                        | TF.csv
                        
                   └───TFs+1000
                        | BL--ExpressionData.csv
                        | BL--network.csv
                        | Label.csv
                        | Target.csv
                        | TF.csv

└───  Lofgof
      └───mESC 500
          | Test_set.csv
          | Train_set.csv
          | Validation_set.csv
      └───mESC 1000
          | Test_set.csv
          | Train_set.csv
          | Validation_set.csv
      
└───  Non-Specific
      └───hESC 500
          | Test_set.csv
          | Train_set.csv
          | Validation_set.csv
      └───hESC 1000
          | Test_set.csv
          | Train_set.csv
          | Validation_set.csv
      └───hHEP 500
          | Test_set.csv
          | Train_set.csv
          | Validation_set.csv
      └───hHEP 500
          | Test_set.csv
          | Train_set.csv
          | Validation_set.csv
      └───mDC 500
          | Test_set.csv
          | Train_set.csv
          | Validation_set.csv
      └───mDC 1000
          | Test_set.csv
          | Train_set.csv
          | Validation_set.csv
      └───mESC 500
          | Test_set.csv
          | Train_set.csv
          | Validation_set.csv
      └───mESC 1000
          | Test_set.csv
          | Train_set.csv
          | Validation_set.csv
      └───mHSC-E 500
          | Test_set.csv
          | Train_set.csv
          | Validation_set.csv
      └───mHSC-E 1000
          | Test_set.csv
          | Train_set.csv
          | Validation_set.csv
      └───mHSC-GM 500
          | Test_set.csv
          | Train_set.csv
          | Validation_set.csv
      └───mHSC-GM 1000
          | Test_set.csv
          | Train_set.csv
          | Validation_set.csv
      └───mHSC-L 500
          | Test_set.csv
          | Train_set.csv
          | Validation_set.csv
      └───mHSC-L 1000
          | Test_set.csv
          | Train_set.csv
          | Validation_set.csv
└───  Specific
      └───hESC 500
          | Test_set.csv
          | Train_set.csv
          | Validation_set.csv
      └───hESC 1000
          | Test_set.csv
          | Train_set.csv
          | Validation_set.csv
      └───hHEP 500
          | Test_set.csv
          | Train_set.csv
          | Validation_set.csv
      └───hHEP 500
          | Test_set.csv
          | Train_set.csv
          | Validation_set.csv
      └───mDC 500
          | Test_set.csv
          | Train_set.csv
          | Validation_set.csv
      └───mDC 1000
          | Test_set.csv
          | Train_set.csv
          | Validation_set.csv
      └───mESC 500
          | Test_set.csv
          | Train_set.csv
          | Validation_set.csv
      └───mESC 1000
          | Test_set.csv
          | Train_set.csv
          | Validation_set.csv
      └───mHSC-E 500
          | Test_set.csv
          | Train_set.csv
          | Validation_set.csv
      └───mHSC-E 1000
          | Test_set.csv
          | Train_set.csv
          | Validation_set.csv
      └───mHSC-GM 500
          | Test_set.csv
          | Train_set.csv
          | Validation_set.csv
      └───mHSC-GM 1000
          | Test_set.csv
          | Train_set.csv
          | Validation_set.csv
      └───mHSC-L 500
          | Test_set.csv
          | Train_set.csv
          | Validation_set.csv
      └───mHSC-L 1000
          | Test_set.csv
          | Train_set.csv
          | Validation_set.csv
└───  STRING
      └───hESC 500
          | Test_set.csv
          | Train_set.csv
          | Validation_set.csv
      └───hESC 1000
          | Test_set.csv
          | Train_set.csv
          | Validation_set.csv
      └───hHEP 500
          | Test_set.csv
          | Train_set.csv
          | Validation_set.csv
      └───hHEP 500
          | Test_set.csv
          | Train_set.csv
          | Validation_set.csv
      └───mDC 500
          | Test_set.csv
          | Train_set.csv
          | Validation_set.csv
      └───mDC 1000
          | Test_set.csv
          | Train_set.csv
          | Validation_set.csv
      └───mESC 500
          | Test_set.csv
          | Train_set.csv
          | Validation_set.csv
      └───mESC 1000
          | Test_set.csv
          | Train_set.csv
          | Validation_set.csv
      └───mHSC-E 500
          | Test_set.csv
          | Train_set.csv
          | Validation_set.csv
      └───mHSC-E 1000
          | Test_set.csv
          | Train_set.csv
          | Validation_set.csv
      └───mHSC-GM 500
          | Test_set.csv
          | Train_set.csv
          | Validation_set.csv
      └───mHSC-GM 1000
          | Test_set.csv
          | Train_set.csv
          | Validation_set.csv
      └───mHSC-L 500
          | Test_set.csv
          | Train_set.csv
          | Validation_set.csv
      └───mHSC-L 1000
          | Test_set.csv
          | Train_set.csv
          | Validation_set.csv
```

 ### Usage
 Command to run scMGATGRN 
 ```
 python main.py --epochs 20 --batch_size 256 --net Specific --num 500 --data hHEP
```
           

             
