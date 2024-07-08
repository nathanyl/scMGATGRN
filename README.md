# scMGATGRN
The seven scRNA-seq datasets can be downloaded from Gene Expression Omnibus (https://www.ncbi.nlm.nih.gov/geo/) database with the accession numbers GSE75748 (hESC, https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE75748), GSE81252 (hHEP, https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE81252), GSE48968 (mDC, https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE48968), GSE98664 (mESC, https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE98664) and GSE81682 (mHSC, https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE81682). All above single-cell datasets with four different kinds of ground-truth networks are available at https://doi.org/10.5281/zenodo.3378975.

### Requirement
- python == 3.7.3
- torch == 1.9.1
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
           

             
