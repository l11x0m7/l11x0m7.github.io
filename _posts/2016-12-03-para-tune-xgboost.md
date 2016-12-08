--- 
layout: post 
title: xgboost完全调参指南
date: 2016-12-03 
categories: blog 
tags: [机器学习] 
description: 翻译xgboost调参指南
--- 

# xgboost调参指南

## 介绍

如果在使用你的预测模型的时候事情没有进展的那么顺利，可以考虑使用XGboost。XGBoost算法是很多数据科学家强有力的武器。它是一个高度复杂的算法，强大到足以解决任何非规则数据。  
使用XGBoost建模很简单，但是使用XGBoost来改进模型是很困难的。这个算法使用了很多参数。为了改进模型，调参非常必要。需要调节哪些参数，以及如何获取最理想的参数值，这些问题都很难回答。  
这篇文章最适合那些XGBoost使用新手。在这篇文章里，我们可以学习到调参的艺术，以及一些关于XGBoost方面的有用的信息。当然，我们会使用数据集来在Python中实践。

## 内容表

接下来按照以下三个大块来讲：

1. The XGBoost Advantage
2. Understanding XGBoost Parameters
3. Tuning Parameters (with Example)

### 1.The XGBoost Advantage

我总是向往这个算法在预测模型里的提升能力。当我在探索它的表现和高准确率下的科学原理的时候，我发现了它的很多优点：  

##### 正则化

* 标准的梯度提升机器（GBM）的实现没有像XGBoost那样的正则项，因此它能够在过拟合方面有所帮助。
* 实际上，XGBoost也被认为是一种“正则提升”技术。  

##### 并行处理

* XGBoost实现了并行处理，和GBM比起来非常快。
* 但是我们也知道提升是一个序列处理过程，因此如何才能做到并行化？我们知道每一棵树能够只根据之前的那一颗来建立，那么是什么阻碍了我们并行化建树？可以查看这个链接来深入探索：[Parallel Gradient Boosting Decision Trees](http://zhanpengfang.github.io/418home.html)  
* XGBoost可以在Hadoop上实现。

##### 高自由度

* XGBoost允许用户自定义优化目标和评估标准。
* 这增加了模型的一个全新的维度，并且并不会限制我们所能做的东西。

##### 处理缺失值

* XGBoost有内建的方法来处理缺失值。
* 用户只需要提供一个不同值，而不是观察并将其作为一个参数。XGBoost在遇到缺失值的时候总是尝试着寻找不同的方式并学习如何去填充缺失值。  

##### 树剪枝

* 当在分割的过程中遇到负损失时，GBM会停止从一个节点产生分支。因此这更像是一种贪婪算法。
* 而XGBoost先产生分支直到最大深度，之后再开始回溯剪枝，并移除哪些不能够获得正收益的分割。
* 另一个这么做的优点是当我们遇到一个负损失分割的时候，比如-2，那么如果接下来的划分为+10。如果是GBM，则是会在遇到-2的时候停止产生分支。但是XGBoost则会继续产生分支，这会使得最终的总分支得分为+8，从而保留这个分支。  

##### 内建交叉验证

* XGBoost允许用户在每次提升的迭代过程中跑一次交叉验证，因此这很容易在跑一次的过程中得到最优的提升迭代次数。
* 这不像GBM一样跑一个grid search并且只有固定的值能够被测试到。

##### 可以衔接到已存在的模型上

* 用户从它之前一次运行的最后一步迭代中开始训练XGBoost模型。这在某类具体的应用中有非常显著的优势。
* sklearn里的GBM的实现同样有这个特性，因此在这一点上GBM和XGBoost一致。

深入理解：

* [XGBoost Guide – Introduction to Boosted Trees](http://xgboost.readthedocs.io/en/latest/model.html)

### 2.XGBoost Parameters

所有的参数可以分成三部分：  

1. General Parameters：主导全局性能  
2. Booster Parameters：主导每一步的独立提升（树/回归）  
3. Learning Task Parameters：主导优化目标

我将通过GBM来类比。  

#### General Parameters

用来调控XGBoost的全局的性能。  

1. booster [default=gbtree]
	* Select the type of model to run at each iteration. It has 2 options:
		* gbtree: tree-based models
		* gblinear: linear models
2. silent [default=0]:
	* Silent mode is activated is set to 1, i.e. no running messages will be printed.
	* It’s generally good to keep it 0 as the messages might help in understanding the model.
3. nthread [default to maximum number of threads available if not set]
	* This is used for parallel processing and number of cores in the system should be entered
	* If you wish to run on all cores, value should not be entered and algorithm will detect automatically

其实还有两个参数，XGBoost设了默认值，不用理会。

#### Booster Parameters

虽然有两种类型的booster，我们只考虑tree booster因为它比linear booster性能要好。

1. eta [default=0.3]
	* Analogous to learning rate in GBM
	* Makes the model more robust by shrinking the weights on each step
	* Typical final values to be used: 0.01-0.2
2. min_child_weight [default=1]
	* Defines the minimum sum of weights of all observations required in a child.
	* This is similar to min_child_leaf in GBM but not exactly. This refers to min “sum of weights” of observations while GBM has min “number of observations”.
	* Used to control over-fitting. Higher values prevent a model from learning relations which might be highly specific to the particular sample selected for a tree.
	* Too high values can lead to under-fitting hence, it should be tuned using CV.
3. max_depth [default=6]
	* The maximum depth of a tree, same as GBM.
	* Used to control over-fitting as higher depth will allow model to learn relations very specific to a particular sample.
	* Should be tuned using CV.
	* Typical values: 3-10
4. max_leaf_nodes
	* The maximum number of terminal nodes or leaves in a tree.
	* Can be defined in place of max_depth. Since binary trees are created, a depth of ‘n’ would produce a maximum of 2^n leaves.
	* If this is defined, GBM will ignore max_depth.
5. gamma [default=0]
	* A node is split only when the resulting split gives a positive reduction in the loss function. Gamma specifies the minimum loss reduction required to make a split.
	* Makes the algorithm conservative. The values can vary depending on the loss function and should be tuned.
6. max_delta_step [default=0]
	* In maximum delta step we allow each tree’s weight estimation to be. If the value is set to 0, it means there is no constraint. If it is set to a positive value, it can help making the update step more conservative.
	* Usually this parameter is not needed, but it might help in logistic regression when class is extremely imbalanced.
	* This is generally not used but you can explore further if you wish.
7. subsample [default=1]
	* Same as the subsample of GBM. Denotes the fraction of observations to be randomly samples for each tree.
	* Lower values make the algorithm more conservative and prevents overfitting but too small values might lead to under-fitting.
	* Typical values: 0.5-1
	* 用于训练模型的子样本占整个样本集合的比例。如果设置为0.5则意味着XGBoost将随机的从整个样本集合中随机的抽取出50%的子样本建立树模型，这能够防止过拟合
8. colsample_bytree [default=1]
	* Similar to max_features in GBM. Denotes the fraction of columns to be randomly samples for each tree.
	* Typical values: 0.5-1
9. colsample_bylevel [default=1]
	* Denotes the subsample ratio of columns for each split, in each level.
	* I don’t use this often because subsample and colsample_bytree will do the job for you. but you can explore further if you feel so.
10. lambda [default=1]
	* L2 regularization term on weights (analogous to Ridge regression)
	* This used to handle the regularization part of XGBoost. Though many data scientists don’t use it often, it should be explored to reduce overfitting.
11. alpha [default=0]
	* L1 regularization term on weight (analogous to Lasso regression)
	* Can be used in case of very high dimensionality so that the algorithm runs faster when implemented
12. scale_pos_weight [default=1]
	* A value greater than 0 should be used in case of high class imbalance as it helps in faster convergence.
	
	
#### Learning Task Parameters

这些参数被用来定义每一步迭代的优化目标的标准。


1. objective [default=reg:linear]
	* This defines the loss function to be minimized. Mostly used values are:
		* binary:logistic –logistic regression for binary classification, returns predicted probability (not class)
		* multi:softmax –multiclass classification using the softmax objective, returns predicted class (not probabilities)
			* you also need to set an additional num_class (number of classes) parameter defining the number of unique classes
		* multi:softprob –same as softmax, but returns predicted probability of each data point belonging to each class.
2. eval_metric [ default according to objective ]
	* The metric to be used for validation data.
	* The default values are rmse for regression and error for classification.
	* Typical values are:
		* rmse – root mean square error
		* mae – mean absolute error
		* logloss – negative log-likelihood
		* error – Binary classification error rate (0.5 threshold)
		* merror – Multiclass classification error rate
		* mlogloss – Multiclass logloss
		* auc: Area under the curve
3. seed [default=0]
	* The random number seed.
	* Can be used for generating reproducible results and also for parameter tuning.
	
当然xgboost可以使用sklearn wrapper里的XGBClassifier来调用。这样就可以直接使用sklearn里的常用的参数名称了。参数名称的变化为（左边为xgboost包里的名称，右边为sklearn包里的名称）：

eta –> learning_rate  
lambda –> reg_lambda  
alpha –> reg_alpha  

记得使用标准xgboost接口来fit function的时候传入num_boosting_rounds参数。

#### Console Parameters

The following parameters are only used in the console version of xgboost
以下参数只能够在控制台使用，即只能在调用的时候传入。比如在train函数里，其参数列表为：

def train(params, dtrain, num_boost_round=10, evals=(), obj=None, feval=None,
          maximize=False, early_stopping_rounds=None, evals_result=None,
          verbose_eval=True, xgb_model=None, callbacks=None, learning_rates=None):

params和其他参数需要分别传入。          

1. use_buffer [ default=1 ] 
	* 是否为输入创建二进制的缓存文件，缓存文件可以加速计算。缺省值为1 
2. num_round 
	* boosting迭代计算次数
3. data 
	* 输入数据的路径 
4. test:data 
	* 测试数据的路径 
5. save_period [default=0] 
	* 表示保存第i*save_period次迭代的模型。例如save_period=10表示每隔10迭代计算XGBoost将会保存中间结果，设置为0表示每次计算的模型都要保持。 
6. task [default=train] options: train, pred, eval, dump 
	* train：训练明显 
	* pred：对测试数据进行预测 
	* eval：通过eval[name]=filenam定义评价指标 
	* dump：将学习模型保存成文本格式 
7. model_in [default=NULL] 
	* 指向模型的路径在test, eval, dump都会用到，如果在training中定义XGBoost将会接着输入模型继续训练 
8. model_out [default=NULL] 
	* 训练完成后模型的保持路径，如果没有定义则会输出类似0003.model这样的结果，0003是第三次训练的模型结果
9. model_dir [default=models] 
	* 输出模型所保存的路径。 
10. fmap 
	* feature map, used for dump model 
11. name_dump [default=dump.txt] 
	* name of model dump file 
12. name_pred [default=pred.txt] 
	* 预测结果文件 
13. pred_margin [default=0] 
	* 输出预测的边界，而不是转换后的概率

## 参考

* [Complete Guide to Parameter Tuning in XGBoost (with codes in Python)](https://www.analyticsvidhya.com/blog/2016/03/complete-guide-parameter-tuning-xgboost-with-codes-python/)

