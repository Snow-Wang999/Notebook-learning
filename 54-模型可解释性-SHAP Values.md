# 54-SHAP Values

了解个体预测

## 介绍

您已经看到（并使用）从机器学习模型中提取一般见解的技术。但是，如果您想分解模型如何为单个预测工作呢？ 

SHAP值（SHapley Additive exPlanations的缩写）分解预测，以显示每个功能的影响。你可以在哪里使用这个？ 

- 一个模型说银行不应该借钱给某人，法律要求银行解释每次拒绝贷款的依据 

- 医疗保健提供者希望确定哪些因素驱动每个患者患某种疾病的风险，以便他们可以通过针对性的健康干预措施直接解决这些风险因素 

在本课中，您将使用SHAP值来解释个体预测。在下一课中，您将了解如何将这些信息聚合为强大的模型级见解。

## How They Work

SHAP值解释了给定特征具有特定值的影响，与我们在该特征具有某个基线值时所做的预测相比。 

一个例子很有用，我们将从排列重要性（[permutation importance](https://www.kaggle.com/dansbecker/permutation-importance)）和部分相关性图（[partial dependence plots](https://www.kaggle.com/dansbecker/partial-plots)）课程中继续足球/足球的例子。 

在这些教程中，我们预测了一支球队是否会有一名球员获得比赛最佳球员奖。

我们可以问： 

- 球队进了3个球，这一预测在多大程度上受到了影响？ 

但如果我们将其重述为：

-  这个预测有多大程度上是由球队打进3球**而不是一些底线进球数所驱动的**。

当然，每个团队都有很多特点。因此，如果我们针对目标数量回答这个问题，我们可以针对所有其他功能重复这个过程。

SHAP值以保证良好属性的方式实现这一点。具体来说，可以使用以下公式分解预测：

```python
sum(SHAP values for all features) = pred_for_team - pred_for_baseline_values
```

也就是说，所有特征的SHAP值相加可以解释为什么我的预测与基线不同。这允许我们将预测分解为如下图：

![image-20221122135742352](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221122135742352.png)

你如何解释这一点？ 

我们预测的是0.7，而base_value是0.4979。导致预测增加的特征值是粉红色的，它们的视觉大小显示了特征效果的大小。降低预测的特征值以蓝色表示。最大的影响来自进球得分为2。尽管控球值对预测有一定的影响。 

如果从粉色条的长度减去蓝色条的长度，则等于从基值到输出的距离。 

这项技术有一定的复杂性，以确保基线加上个体效应的总和加上预测（这并不像听起来那么简单）。我们在这里不再赘述，因为这对使用该技术并不重要。这篇博文有更长的理论解释。

## 计算SHAP值的代码

我们使用精彩的SHAP库计算SHAP值。 

在本例中，我们将重用您已经在足球数据中看到的模型。

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

data = pd.read_csv('../input/fifa-2018-match-statistics/FIFA 2018 Statistics.csv')
y = (data['Man of the Match'] == "Yes")  # Convert from string "Yes"/"No" to binary
feature_names = [i for i in data.columns if data[i].dtype in [np.int64, np.int64]]
X = data[feature_names]
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)
my_model = RandomForestClassifier(random_state=0).fit(train_X, train_y)
```

我们将查看数据集中单行的SHAP值（我们任意选择了第5行）。对于上下文，我们将在查看SHAP值之前查看原始预测。

```python
row_to_show = 5
data_for_prediction = val_X.iloc[row_to_show]  # use 1 row of data here. Could use multiple rows if desired
data_for_prediction_array = data_for_prediction.values.reshape(1, -1)

my_model.predict_proba(data_for_prediction_array)
```

```
array([[0.29, 0.71]])
```

该队有70%的可能性让一名球员获奖。 现在，我们将转到获取单个预测的SHAP值的代码。

```python
import shap  # package used to calculate Shap values

# Create object that can calculate shap values
explainer = shap.TreeExplainer(my_model)

# Calculate Shap values
shap_values = explainer.shap_values(data_for_prediction)
```

上面的shap_values对象是一个包含两个数组的列表。第一个数组是负面结果（不获奖）的SHAP值，第二个数组是正面结果（获奖）的HAP值列表。我们通常根据对积极结果的预测来考虑预测，因此我们将为积极结果提取SHAP值（提取SHAP_values[1]）。 查看原始数组很麻烦，但是shap包有一个很好的方法来可视化结果。

```python
shap.initjs()
shap.force_plot(explainer.expected_value[1], shap_values[1], data_for_prediction)
```

![image-20221122141206258](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221122141206258.png)

如果仔细查看我们创建SHAP值的代码，您会发现我们在`shap.TreeExplainer(my_model)`中引用了Trees。但是，SHAP包对每种类型的模型都有解释。 

- `shap.DeepExplainer`使用深度学习模型。 

- `shap.KernelExplainer`适用于所有模型，尽管它比其他解释程序慢，而且它提供的是近似值而不是精确的Shap值。 

下面是一个使用`KernelExplainer`获得类似结果的示例。结果不完全相同，因为`KernelExplainer`给出了一个近似的结果。但结果也说明了同样的道理。

```python
# use Kernel SHAP to explain test set predictions
k_explainer = shap.KernelExplainer(my_model.predict_proba, train_X)
k_shap_values = k_explainer.shap_values(data_for_prediction)
shap.force_plot(k_explainer.expected_value[1], k_shap_values[1], data_for_prediction)
```

![image-20221122141508125](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221122141508125.png)

## 轮到你了 

SHAP值太棒了。将它们与您所学的其他工具一起应用，以解决完整的数据科学场景。

## Exercise: SHAP Values

### Set Up

在这一点上，您有足够的工具来为现实世界的问题提供令人信服的解决方案。您需要为以下数据科学场景的每个部分选择正确的技术。在此过程中，您将使用SHAP值和其他洞察工具。 

下面的问题通过使用一些检查代码为您的工作提供反馈。运行以下单元格以设置反馈系统。

```python
from learntools.ml_explainability.ex4 import *
print("Setup Complete")
```

### 方案(The Scenario)

一家医院一直在为“重新入院”而挣扎，即在患者康复之前释放患者，患者会出现健康并发症。 

医院希望您帮助确定再次入院风险最高的患者。医生（而不是你的模特）将决定何时释放每个病人；但他们希望你的模型能突出医生在释放病人时应该考虑的问题。 

医院已经向您提供了相关的患者医疗信息。以下是数据中的列列表：

```python
import pandas as pd
data = pd.read_csv('../input/hospital-readmissions/train.csv')
data.columns
```

```
Index(['time_in_hospital', 'num_lab_procedures', 'num_procedures',
       'num_medications', 'number_outpatient', 'number_emergency',
       'number_inpatient', 'number_diagnoses', 'race_Caucasian',
       'race_AfricanAmerican', 'gender_Female', 'age_[70-80)', 'age_[60-70)',
       'age_[50-60)', 'age_[80-90)', 'age_[40-50)', 'payer_code_?',
       'payer_code_MC', 'payer_code_HM', 'payer_code_SP', 'payer_code_BC',
       'medical_specialty_?', 'medical_specialty_InternalMedicine',
       'medical_specialty_Emergency/Trauma',
       'medical_specialty_Family/GeneralPractice',
       'medical_specialty_Cardiology', 'diag_1_428', 'diag_1_414',
       'diag_1_786', 'diag_2_276', 'diag_2_428', 'diag_2_250', 'diag_2_427',
       'diag_3_250', 'diag_3_401', 'diag_3_276', 'diag_3_428',
       'max_glu_serum_None', 'A1Cresult_None', 'metformin_No',
       'repaglinide_No', 'nateglinide_No', 'chlorpropamide_No',
       'glimepiride_No', 'acetohexamide_No', 'glipizide_No', 'glyburide_No',
       'tolbutamide_No', 'pioglitazone_No', 'rosiglitazone_No', 'acarbose_No',
       'miglitol_No', 'troglitazone_No', 'tolazamide_No', 'examide_No',
       'citoglipton_No', 'insulin_No', 'glyburide-metformin_No',
       'glipizide-metformin_No', 'glimepiride-pioglitazone_No',
       'metformin-rosiglitazone_No', 'metformin-pioglitazone_No', 'change_No',
       'diabetesMed_Yes', 'readmitted'],
      dtype='object')
```

```python
data.info()
```

```
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 25000 entries, 0 to 24999
Data columns (total 65 columns):
 #   Column                                    Non-Null Count  Dtype
---  ------                                    --------------  -----
 0   time_in_hospital                          25000 non-null  int64
 1   num_lab_procedures                        25000 non-null  int64
 2   num_procedures                            25000 non-null  int64
 3   num_medications                           25000 non-null  int64
 4   number_outpatient                         25000 non-null  int64
 5   number_emergency                          25000 non-null  int64
 6   number_inpatient                          25000 non-null  int64
 7   number_diagnoses                          25000 non-null  int64
 8   race_Caucasian                            25000 non-null  bool 
 9   race_AfricanAmerican                      25000 non-null  bool 
 10  gender_Female                             25000 non-null  bool 
 11  age_[70-80)                               25000 non-null  bool 
 12  age_[60-70)                               25000 non-null  bool 
 13  age_[50-60)                               25000 non-null  bool 
 14  age_[80-90)                               25000 non-null  bool 
 15  age_[40-50)                               25000 non-null  bool 
 16  payer_code_?                              25000 non-null  bool 
 17  payer_code_MC                             25000 non-null  bool 
 18  payer_code_HM                             25000 non-null  bool 
 19  payer_code_SP                             25000 non-null  bool 
 20  payer_code_BC                             25000 non-null  bool 
 21  medical_specialty_?                       25000 non-null  bool 
 22  medical_specialty_InternalMedicine        25000 non-null  bool 
 23  medical_specialty_Emergency/Trauma        25000 non-null  bool 
 24  medical_specialty_Family/GeneralPractice  25000 non-null  bool 
 25  medical_specialty_Cardiology              25000 non-null  bool 
 26  diag_1_428                                25000 non-null  bool 
 27  diag_1_414                                25000 non-null  bool 
 28  diag_1_786                                25000 non-null  bool 
 29  diag_2_276                                25000 non-null  bool 
 30  diag_2_428                                25000 non-null  bool 
 31  diag_2_250                                25000 non-null  bool 
 32  diag_2_427                                25000 non-null  bool 
 33  diag_3_250                                25000 non-null  bool 
 34  diag_3_401                                25000 non-null  bool 
 35  diag_3_276                                25000 non-null  bool 
 36  diag_3_428                                25000 non-null  bool 
 37  max_glu_serum_None                        25000 non-null  bool 
 38  A1Cresult_None                            25000 non-null  bool 
 39  metformin_No                              25000 non-null  bool 
 40  repaglinide_No                            25000 non-null  bool 
 41  nateglinide_No                            25000 non-null  bool 
 42  chlorpropamide_No                         25000 non-null  bool 
 43  glimepiride_No                            25000 non-null  bool 
 44  acetohexamide_No                          25000 non-null  bool 
 45  glipizide_No                              25000 non-null  bool 
 46  glyburide_No                              25000 non-null  bool 
 47  tolbutamide_No                            25000 non-null  bool 
 48  pioglitazone_No                           25000 non-null  bool 
 49  rosiglitazone_No                          25000 non-null  bool 
 50  acarbose_No                               25000 non-null  bool 
 51  miglitol_No                               25000 non-null  bool 
 52  troglitazone_No                           25000 non-null  bool 
 53  tolazamide_No                             25000 non-null  bool 
 54  examide_No                                25000 non-null  bool 
 55  citoglipton_No                            25000 non-null  bool 
 56  insulin_No                                25000 non-null  bool 
 57  glyburide-metformin_No                    25000 non-null  bool 
 58  glipizide-metformin_No                    25000 non-null  bool 
 59  glimepiride-pioglitazone_No               25000 non-null  bool 
 60  metformin-rosiglitazone_No                25000 non-null  bool 
 61  metformin-pioglitazone_No                 25000 non-null  bool 
 62  change_No                                 25000 non-null  bool 
 63  diabetesMed_Yes                           25000 non-null  bool 
 64  readmitted                                25000 non-null  int64
dtypes: bool(56), int64(9)
memory usage: 3.1 MB
```

下面是一些解释字段名称的快速提示： 

- 您的预测目标是 `readmitted`

- 带有单词`diag`的列表示患者入院时所患疾病的诊断代码。例如，`diag_1_428`表示医生表示他们的第一个疾病诊断是编号“428”。428对应什么疾病？你可以在代码本中查找它，但如果没有更多的医学背景，它对你来说也没有任何意义。 

- `glimepiride_No`列名称表示患者没有服用药物格列美脲。如果该特征值为False，则患者确实服用了药物格列美脲(glimepiride) 

- 名称以`medical_speciality`开头的特征描述了医生为患者看病的专业。这些字段中的值均为True或False。

### 您的代码库

当您编写代码来完成这个场景时，前面教程中的这些代码片段可能很有用。您仍然需要修改它们，但我们已将它们复制到此处，以避免您查找它们。

计算并显示排列重要性：

**Calculate and show permutation importance:**

```python
import eli5
from eli5.sklearn import PermutationImportance

perm = PermutationImportance(my_model, random_state=1).fit(val_X, val_y)
eli5.show_weights(perm, feature_names = val_X.columns.tolist())
```

计算并显示部分相关性图：

**Calculate and show partial dependence plot:**

```python
from matplotlib import pyplot as plt
from pdpbox import pdp, get_dataset, info_plots

# Create the data that we will plot
pdp_goals = pdp.pdp_isolate(model=my_model, dataset=val_X, model_features=feature_names, feature='Goal Scored')

# plot it
pdp.pdp_plot(pdp_goals, 'Goal Scored')
plt.show()
```

计算并显示一个预测的`Shap`值：

**Calculate and show Shap Values for One Prediction:**

```python
import shap  # package used to calculate Shap values

data_for_prediction = val_X.iloc[0,:]  # use 1 row of data here. Could use multiple rows if desired

# Create object that can calculate shap values
explainer = shap.TreeExplainer(my_model)
shap_values = explainer.shap_values(data_for_prediction)
shap.initjs()
shap.force_plot(explainer.expected_value[0], shap_values[0], data_for_prediction)
```

#### Step 1

你已经建立了一个简单的模型，但医生说他们不知道如何评估模型，他们希望你向他们展示一些证据，证明模型在做符合他们医学直觉的事情。创建任何图形或表格，让他们快速了解模型在做什么？ 

他们很忙。因此，他们希望您将模型概述压缩为1或2个图形，而不是一长串图形。 

我们将在您构建基本模型之后开始。只需运行下面的单元来构建名为`my_model`的模型。

```python
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

data = pd.read_csv('../input/hospital-readmissions/train.csv')

y = data.readmitted

base_features = [c for c in data.columns if c != "readmitted"]

X = data[base_features]

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)
my_model = RandomForestClassifier(n_estimators=30, random_state=1).fit(train_X, train_y)
```

现在使用以下单元格为医生创建材料。

```python
# Use permutation importance as a succinct model summary
# A measure of model performance on validation data would be useful here too
import eli5
from eli5.sklearn import PermutationImportance

perm = PermutationImportance(my_model, random_state=1).fit(val_X, val_y)
eli5.show_weights(perm, feature_names = val_X.columns.tolist())
```

|          Weight | Feature                                  |
| --------------: | :--------------------------------------- |
| 0.0451 ± 0.0068 | number_inpatient                         |
| 0.0087 ± 0.0046 | number_emergency                         |
| 0.0062 ± 0.0053 | number_outpatient                        |
| 0.0033 ± 0.0016 | payer_code_MC                            |
| 0.0020 ± 0.0016 | diag_3_401                               |
| 0.0016 ± 0.0031 | medical_specialty_Emergency/Trauma       |
| 0.0014 ± 0.0024 | A1Cresult_None                           |
| 0.0014 ± 0.0021 | medical_specialty_Family/GeneralPractice |
| 0.0013 ± 0.0010 | diag_2_427                               |
| 0.0013 ± 0.0011 | diag_2_276                               |
| 0.0011 ± 0.0022 | age_[50-60)                              |
| 0.0010 ± 0.0022 | age_[80-90)                              |
| 0.0007 ± 0.0006 | repaglinide_No                           |
| 0.0006 ± 0.0010 | diag_1_428                               |
| 0.0006 ± 0.0022 | payer_code_SP                            |
| 0.0005 ± 0.0030 | insulin_No                               |
| 0.0004 ± 0.0028 | diabetesMed_Yes                          |
| 0.0004 ± 0.0021 | diag_3_250                               |
| 0.0003 ± 0.0018 | diag_2_250                               |
| 0.0003 ± 0.0015 | glipizide_No                             |
|   *… 44 more …* |                                          |

![image-20221122143339435](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221122143339435.png)

`number_inpatient`特征是最重要的，对再次住院有相对大的影响。

如果你想讨论你的方法或看看其他人做了什么，我们在这里([here](https://www.kaggle.com/learn-forum/66267#latest-390149))有一个讨论论坛。

#### Step 2

看来`number_inpatient`是一个非常重要的功能。医生们想知道更多关于这方面的信息。为他们创建一个图表，显示`num_inpatient`如何影响模型的预测。

```python
# PDP for number_inpatient feature

from matplotlib import pyplot as plt
from pdpbox import pdp, get_dataset, info_plots

feature_name = 'number_inpatient'
# Create the data that we will plot
my_pdp = pdp.pdp_isolate(model=my_model, dataset=val_X, model_features=val_X.columns, feature=feature_name)

# plot it
pdp.pdp_plot(my_pdp, feature_name)
plt.show()
```

![image-20221122143729077](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221122143729077.png)

随着`num_inpatient`的增大，再次住院率也增大。即增加住院手术的数量会导致预测的增加。

#### Step 3

医生们认为这是一个好的迹象，增加住院手术的数量会导致预测的增加。但他们无法从这一情节中判断情节的变化是大还是小。他们希望你为`time_in_hospital`创建类似的东西，看看两者之间的比较。

![image-20221122144008956](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221122144008956.png)

住院时间对再次住院率几乎没有影响。

#### Step 4

哇！看来住院时间一点都不重要。部分依赖图上的最低值与最高值之间的差值约为5%。 

如果这是你的模型得出的结论，医生会相信的。但它似乎很低。数据可能是错误的，或者你的模型做的事情比他们预期的更复杂吗？ 

他们希望您向他们显示`time_in_hospital`的每个值的原始再入院率，看看它与部分依赖图的比较情况。 

- 绘制那张图。 

- 结果相似还是不同？

```python
# A simple pandas groupby showing the average readmission rate for each time_in_hospital.

# Do concat to keep validation data separate, rather than using all original data
all_train = pd.concat([train_X,train_y],axis=1)

all_train.groupby(['time_in_hospital']).mean().readmitted.plot()
plt.show()
```

![image-20221122144803253](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221122144803253.png)

随着住院天数的增加，再次住院率增加，如果住院天数超过12天，再次住院率快速提升至一半。

#### Step 5

现在医生们确信你掌握了正确的数据，模型概述看起来很合理。是时候把它变成他们可以使用的成品了。具体来说，医院希望您创建一个函数`patient_risk_factors`，该函数执行以下操作 

- 用单行患者数据（与原始数据的格式相同）

- 创建一个可视化视图，显示患者的哪些特征增加了再次入院的风险，哪些特征降低了再次入院风险，以及这些特征有多重要。 

显示每个特征对再入院风险的微小影响并不重要。只关注患者最重要的特征是很好的。

```python
import shap  # package used to calculate Shap values

sample_data_for_prediction = val_X.iloc[0].astype(float)  # use 1 row of data here. Could use multiple rows if desired

def patient_risk_factors(model,patient_data):
    # Create object that can calculate shap values
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(patient_data)
    shap.initjs()
    return shap.force_plot(explainer.expected_value[1], shap_values[1], patient_data)
```

```python
patient_risk_factors(my_model,sample_data_for_prediction)
```



![image-20221122150001585](C:\Users\Myste\AppData\Roaming\Typora\typora-user-images\image-20221122150001585.png)

我们预测的再次住院率是0.27，而base_value是0.4538。导致预测增加的特征值是粉红色的，它们的视觉大小显示了特征效果的大小。降低预测的特征值以蓝色表示。最大的影响来自`num_procedures=4`，它会使预测减小。

### 继续前进 

你有一些强大的工具来深入了解模型和个人预测。接下来，您将查看SHAP值的聚合(**[aggregations of SHAP values](https://www.kaggle.com/dansbecker/advanced-uses-of-shap-values)** )，以链接模型级别和预测级别的见解。