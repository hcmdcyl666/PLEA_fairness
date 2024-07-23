<<<<<<< HEAD
# Achieving Group Fairness under Erroneous Pseudo-Labels of Sensitive Attributes

Pytorch implementation of <strong>Achieving Group Fairness under Erroneous
Pseudo-Labels of Sensitive Attributes</strong> | 
[Yulin Cai]<sup>1</sup> [Xiaoying Zhang]<sup>2</sup> [Hong Xie]<sup>3</sup> etc.

<sup>1</sup><sub>Chongqing Institute of Green and Intelligent Technology, CAS<br>
<sup>2</sup><sub>Bytedance</sub><br>
<sup>3</sup><sub>University of Science and Technology of China

This paper studies the problem of achieving group fairness 
in the practical setting where the sensitive attribute of 
a tiny portion of the training data is accessible.   
Existing fairness-aware algorithms typically 
train a sensitive attribute classifier from
the subset of training data with sensitive attribute 
to impute the training data with missing sensitive attribute.  
Unfortunately, this direct approach has been proven ineffective because the aforementioned classifier will produce erroneous pseudo-labels for sensitive attributes, which will mislead the model's fair training.  
In this paper, we propose a Pseudo-Label Error Aware (PLEA) framework that can assist existing fairness-aware learning algorithms in achieving group fairness. Specifically, PLEA also uses an additional classifier to predict pseudo-labels for sensitive attributes. However, this framework mitigates the negative impact of erroneous pseudo-labels on the model's fair training, i.e., its fairness performance, through two approaches: 1) resampling based on sample weights and 2) computing weighted fairness loss in bins, where the sample weights are inversely proportional to the error of the sensitive attribute pseudo-labels.
Our proposed framework enhances existing fairness-aware learning algorithms by leveraging samples that lack sensitive labels, aiming to improve their fairness performances in such real-world application scenarios. Experiments on two real-world datasets demonstrate that PLEA can significantly enhance the fairness performance, with almost no loss in accuracy.

## Updates

- 23 July, 2024: Initial upload.
    
```
To ensure the code runs correctly, it is best to configure the runtime environment according to the fair.yaml file.
```    
## How to train 
   The training of the sensitive attribute classifier g and the class classifier f is given according to the command line examples below, and relevant parameters can be modified.

### 1. Train a sensitive attribute classifier
```
$ python main_groupclf.py --model mlp \
--method scratch \
--dataset compas \
--version groupclf_val \
--sv 0.05 \
--seed 0  
```

### 2. Train a fair classifier

```
- FairHSIC
```
```
w_method=(exp poly)
gammas=(0.5 1.0 2.0)
dis_methods=(msp energy) # 就是dis-metric
methods=(mfd mfd_bin)
ratios=(0.1 0.05)

$ python main.py \
--model mlp \
--method fairhsic \
--dataset compas \
--labelwise \
--version weight \
--seed 1 \
--sv 0.05 \
--lamb 10000 \
--batch-size 128 \
--epochs 60 \
--t 1 \
--dis-metric msp \
--dis-method exp \
--w-method resample \
--gamma 0.5 \
--device 0 \
--teacher-type mlp \
--teacher-path "..."
```
After executing the above commands, a "distances" folder will be generated to store metrics quantifying errors, a "weights" folder will be generated to store the sample weights obtained using the corresponding seed and response calculation method, a "result" folder will be created, and the trained models will be stored in the "trained_models" folder.

## How to cite


## License


=======
# fairness
just test it!
>>>>>>> 701a626b9c004b53d6cbdb30b58db8295f247746
