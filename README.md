# <center>Final Project Report</center>

# <center>Classifier for Sonar Detection of Naval Mines vs Rocks</center>

## Indroduction

Sonar (short for sound navigation and ranging) is a technique which utilizes sound propagation to develop nautical charts. This tool is most commonly implemented underwater to measure distances and communicate with/detect objects on or below the ocean surface. The focus of this project will be the Solar, Mines vs Rocks data set. The file consists of 111 patterns produced by sonar signals bouncing off a metal cylinder and 97 patterns off of rocks. These signals were obtained via varying aspect angles at 90 degrees (cylinder) and 180 degrees (rock). The broadcasted sonar signal was a frequency-modulated chirp, with increasing frequency. Each pattern is a set of 60 numbers in the range 0.0 to 1.0. The numbers indicate the energy within a specific frequency band, integrated over a certain period of time. The label corresponding to each record contains either the letter “R’ for rock objects or “M” for mines (metal cylinder). The numbers within labels appear in increasing arrangement of aspect angles, but do not encode the angle directly. The problem at hand revolves around the question: “is a specific sonar signal bounced off a metal cylinder or roughly cylindrical rock?”. To address this inquiry, throughout this project, a classification model will be developed to predict metal or rock objects from the sonar return data.

## Methods

First, we loaded our Sonar Data set in directly from the website via its url; read_csv was used to accomplish this. The dataset from the website does not have column names so we set `col_names` to `FALSE` in the argument. This will assign column names as “X” followed by the number of the column. We did not see any purpose in renaming the individual bands, however, the very last column denotes the “type” so is renamed as such. The `mutate()` and `as_factor()` functions are also implemented to convert our `type` column from a vector object to a statistical categorical variable.

```R
library(tidyverse)
library(tidymodels)
```

    ── tidyverse 1.3.1 ──

```R
sonar_url <- "https://archive.ics.uci.edu/ml/machine-learning-databases/undocumented/connectionist-bench/sonar/sonar.all-data"

# does rename successfully but does not show in the red box
sonar_data <- read_csv(sonar_url, col_names = FALSE) |>
    rename(type = X61) |>
    mutate(type = as_factor(type))
```

    Columns: Column specification
    Delimiter: ","
    (1): X61
    (60): X1, X2, X3, X4, X5, X6, X7, X8, X9, X10, X11, X12, X13, X14, X15, ...

### Wrangling data

Our initial plan was to wrangle the dataset to have three columns: **band**(a factor in range from X1 to X60), **value**(0.0 to 1.0) and **type**(will be a factor; rocks(R) or mines(M)). However, after talking to the TA and facing difficulty in choosing appropriate inputs/predictors, we have decided to leave the dataset as is and have **bands** ranging from **X1** to **X60** in separate columns. However, we can increase out accuracy by reducing predictors as seen later.

```R
# shuffling data since it is arranged as R then M
set.seed(9999)
sonar_set <- sonar_data[sample(1:nrow(sonar_data)), ]
```

```R
# splitting the data

sonar_split <- initial_split(sonar_set, prop = 0.75, strata = type)
sonar_train <- training(sonar_split)
sonar_test <- testing(sonar_split)
```

- The `initial_split` function is used to create training and testing sets
- It is specified that `prop = 0.75` so that 75% of our original data set ends up in the training set
- The `strata` argument is set to the categorical label variable (here, type) to ensure that the training and testing subsets contain the right proportions of each category of observation
- The `training()` and `testing()` functions extract the training and testing data sets into two separate data frames

### Exploratory data analysis(tables)

**Summary of exploration below all the code cells**

- Here in some cases we used `sonar_set` to verify the data collected from the web

```R
# verifying if the set matches the description
verify_count <- sonar_set |>
    group_by(type)|>
    summarize(n())

verify_count
```

<table class="dataframe">
<caption>A tibble: 2 × 2</caption>
<thead>
	<tr><th scope=col>type</th><th scope=col>n()</th></tr>
	<tr><th scope=col>&lt;fct&gt;</th><th scope=col>&lt;int&gt;</th></tr>
</thead>
<tbody>
	<tr><td>R</td><td> 97</td></tr>
	<tr><td>M</td><td>111</td></tr>
</tbody>
</table>

- We matched the number of all the rocks and mines with the descriptions; matches

```R
empty_obs <- colSums(is.na(sonar_set))

empty_obs
```

<style>
.dl-inline {width: auto; margin:0; padding: 0}
.dl-inline>dt, .dl-inline>dd {float: none; width: auto; display: inline-block}
.dl-inline>dt::after {content: ":\0020"; padding-right: .5ex}
.dl-inline>dt:not(:first-of-type) {padding-left: .5ex}
</style><dl class=dl-inline><dt>X1</dt><dd>0</dd><dt>X2</dt><dd>0</dd><dt>X3</dt><dd>0</dd><dt>X4</dt><dd>0</dd><dt>X5</dt><dd>0</dd><dt>X6</dt><dd>0</dd><dt>X7</dt><dd>0</dd><dt>X8</dt><dd>0</dd><dt>X9</dt><dd>0</dd><dt>X10</dt><dd>0</dd><dt>X11</dt><dd>0</dd><dt>X12</dt><dd>0</dd><dt>X13</dt><dd>0</dd><dt>X14</dt><dd>0</dd><dt>X15</dt><dd>0</dd><dt>X16</dt><dd>0</dd><dt>X17</dt><dd>0</dd><dt>X18</dt><dd>0</dd><dt>X19</dt><dd>0</dd><dt>X20</dt><dd>0</dd><dt>X21</dt><dd>0</dd><dt>X22</dt><dd>0</dd><dt>X23</dt><dd>0</dd><dt>X24</dt><dd>0</dd><dt>X25</dt><dd>0</dd><dt>X26</dt><dd>0</dd><dt>X27</dt><dd>0</dd><dt>X28</dt><dd>0</dd><dt>X29</dt><dd>0</dd><dt>X30</dt><dd>0</dd><dt>X31</dt><dd>0</dd><dt>X32</dt><dd>0</dd><dt>X33</dt><dd>0</dd><dt>X34</dt><dd>0</dd><dt>X35</dt><dd>0</dd><dt>X36</dt><dd>0</dd><dt>X37</dt><dd>0</dd><dt>X38</dt><dd>0</dd><dt>X39</dt><dd>0</dd><dt>X40</dt><dd>0</dd><dt>X41</dt><dd>0</dd><dt>X42</dt><dd>0</dd><dt>X43</dt><dd>0</dd><dt>X44</dt><dd>0</dd><dt>X45</dt><dd>0</dd><dt>X46</dt><dd>0</dd><dt>X47</dt><dd>0</dd><dt>X48</dt><dd>0</dd><dt>X49</dt><dd>0</dd><dt>X50</dt><dd>0</dd><dt>X51</dt><dd>0</dd><dt>X52</dt><dd>0</dd><dt>X53</dt><dd>0</dd><dt>X54</dt><dd>0</dd><dt>X55</dt><dd>0</dd><dt>X56</dt><dd>0</dd><dt>X57</dt><dd>0</dd><dt>X58</dt><dd>0</dd><dt>X59</dt><dd>0</dd><dt>X60</dt><dd>0</dd><dt>type</dt><dd>0</dd></dl>

- We checked for empty cells in column; no empty cells

```R
rock_10_band_mean <- sonar_train |>
    filter(type == "R")|>
    select(X1:X10) |>
    map(mean) |>
    data.frame()
rock_10_band_mean
```

<table class="dataframe">
<caption>A data.frame: 1 × 10</caption>
<thead>
	<tr><th scope=col>X1</th><th scope=col>X2</th><th scope=col>X3</th><th scope=col>X4</th><th scope=col>X5</th><th scope=col>X6</th><th scope=col>X7</th><th scope=col>X8</th><th scope=col>X9</th><th scope=col>X10</th></tr>
	<tr><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th></tr>
</thead>
<tbody>
	<tr><td>0.02290278</td><td>0.03145833</td><td>0.0371</td><td>0.04174583</td><td>0.06115833</td><td>0.09964444</td><td>0.1164764</td><td>0.12115</td><td>0.1367</td><td>0.1612903</td></tr>
</tbody>
</table>

```R
mine_10_band_mean <- sonar_train |>
    filter(type == "M")|>
    select(X1:X10) |>
    map(mean) |>
    data.frame()
mine_10_band_mean
```

<table class="dataframe">
<caption>A data.frame: 1 × 10</caption>
<thead>
	<tr><th scope=col>X1</th><th scope=col>X2</th><th scope=col>X3</th><th scope=col>X4</th><th scope=col>X5</th><th scope=col>X6</th><th scope=col>X7</th><th scope=col>X8</th><th scope=col>X9</th><th scope=col>X10</th></tr>
	<tr><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th><th scope=col>&lt;dbl&gt;</th></tr>
</thead>
<tbody>
	<tr><td>0.03203976</td><td>0.04141807</td><td>0.04771446</td><td>0.06454699</td><td>0.08704578</td><td>0.1023747</td><td>0.1213422</td><td>0.1337398</td><td>0.197594</td><td>0.2370578</td></tr>
</tbody>
</table>

- Means of the first 10 bands of each type; both have an increasing trend but stronger in mines. Further relation explored ahead.

### Exploratory data analysis (visual)

- We want to see if the two types have distinct distributions, so that we can distinguish between them through knn model analysis.

```R
rock_mean <- sonar_train |>
    filter(type == "R")|>
    select(-type) |>
    map(mean) |>
    data.frame()
rock_mean["type"] <- c("R")

mine_mean <- sonar_train |>
    filter(type == "M")|>
    select(-type) |>
    map(mean) |>
    data.frame()
mine_mean["type"] <- c("M")

plot_data <- rbind(rock_mean, mine_mean) |>
    pivot_longer(X1:X60, names_to = "band", values_to = "value")|>
    mutate(band = as_factor(band))
```

The data frames `rock_mean` and `mine_mean` were created by:

- filtering the `sonar_train` data based on the `type` column ("R" for rock and "M" for mine)
- calculating the mean of each column using the `map(mean)` function
- converting the results into data frames with data.frame()

The `rock_mean` and `mine_mean` data frames were then given a `type` column with values "R" for `rock_mean` and "M" for `mine_mean`.

The `plot_data` data frame was created by combining the `rock_mean` and `mine_mean` data frames using `rbind()` and tidied. The `band` column was also converted into a factor using `mutate`.

```R
options(repr.plot.width = 13)
# plot_data
mean_plot <- plot_data |>
    ggplot(aes(x = band, y = value, color = as_factor(type))) +
    geom_point() +
    facet_grid(cols = vars(type), labeller = as_labeller(c(M = "Mine", R = "Rock"))) +
    labs(x = "Band(in order)", y = "Average Value(0.0 to 1.0)", color = "Legend", title = "Comparison of band means between Mine and Rock for all bands") +
    ylim(0.0, 1.0)+
# scale_x_continuous(limits = c(1, 60))+
    theme(axis.text.x=element_blank(),
        axis.ticks.x=element_blank(),
        text = element_text(size=20))
mean_plot
```

![png](output_26_0.png)

The combined plot_data data frame was visualized as a scatterplot by:

a. Plotting the `band` colomn along the x-axis against its corresponding average value ranging from 0.0 to 1.0 along the y-axis by the `aes()` function

b. Using `color = as_factor(type)` to specify graphic colors according to the factor `type` (i.e. "R" or "M")

c. Creating a multi-axis grid with subplots illustrating the distribution of separate variables ("Mine" and "Rock") of the data set using `facet_grid()`

##### Observation

- The last 11 bands do not show significant difference in pattern or trend of between rocks and mines.
- We can omit these last 11 bands; we can also verify this through accuracy result below

```R
#recipe, spec and cross validation spec
recipe_full <- recipe(type ~ ., data = sonar_train)|>
    step_scale(all_predictors()) |>
    step_center(all_predictors())

recipe_reduced <- recipe(type ~ ., data = sonar_train) |>
    step_rm(X50:X60) |>
    step_scale(all_predictors()) |>
    step_center(all_predictors())

knn_spec <- nearest_neighbor(weight_func = "rectangular", neighbors = tune()) |>
    set_engine("kknn") |>
    set_mode("classification")

# 5 fold validation since the dataset is small
sonar_vfold <- vfold_cv(sonar_train, v = 5, strata = type)
```

- K-nearest neighbors is sensitive to the scale of the predictors, so `step_scale()` and `step_center()` are used to perform some preprocessing and standardize them
- `recipe_full` includes all predictors
- `recipe_reduced` excludes predictors from "X50" to "X60" columns
- `vfold_cv` splits up the training data for cross validation; "v = 5" indicates 5 folds automatically, and we set the `strata` argument to the catagorical label (here, type) to ensure that the training and validation subsets contain the right proportions of each category of observations
- A model specification for K-nearest neighbors classification is created by calling the `nearest_neighbor` function, with `tune()`

```R
#workflow
work_full <- workflow()|> add_recipe(recipe_full)|> add_model(knn_spec)
work_reduced <- workflow()|> add_recipe(recipe_reduced)|> add_model(knn_spec)
```

- Here, we add the recipe and model specification to a `workflow()`

```R
#results
metrics_full <- work_full |> tune_grid(resamples = sonar_vfold, grid = tibble(neighbors = 2:10))|>
    collect_metrics()
metrics_reduced <- work_reduced |> tune_grid(resamples = sonar_vfold, grid = tibble(neighbors = 2:10))|>
    collect_metrics()
```

- The `tune_grid` function is used on the train/validation splits to estimate the classifier accuracy for a range of K values
- The `collect_metrics` function is used to aggregate the mean and standard error of the classifier’s validation accuracy across the folds

```R
options(repr.plot.height = 6, repr.plot.width = 10)
cross_plot_data <- bind_rows(mutate(metrics_full, id = as_factor("f")),
                             mutate(metrics_reduced, id = as_factor("r")))|>
                    filter(.metric == "accuracy")
cross_val_plot <- cross_plot_data |>
    ggplot(aes(x = neighbors, y = mean, color = id)) +
    geom_point() +
    geom_line() +
    labs(x = "Neighbors", y = "Accuracy Estimate", color = "Workflow type", title = "Comparison of accuracy in full and reduced workflow across varying neighbours")+
    scale_x_continuous(breaks = seq(0, 10, by = 1)) +  # adjusting the x-axis
    scale_y_continuous(limits = c(0.4, 1.0)) + # adjusting the y-axis
    theme(text = element_text(size = 20), title = element_text(size = 15)) +
    scale_colour_discrete(labels = c("Full", "Reduced"))
cross_val_plot
# cross_plot_data

```

![png](output_34_0.png)

- Excluding bands "X50" to "X60" (reduced) results in a classifier with higher accuracy compared to full aside from when neighbours is equal to 2 to 2.5(due ot overfitting)
- We will choose the 3 neighbours with the reduced workflow type as it yields the highest accuracy estimate(and use an odd number to avoid draws)

```R
knn_spec_final <- nearest_neighbor(weight_func = "rectangular", neighbors = 3) |>
    set_engine("kknn") |>
    set_mode("classification")

workflow_final <- workflow()|>
    add_recipe(recipe_reduced)|>
    add_model(knn_spec_final)|>
    fit(data = sonar_train)

sonar_predictions <- predict(workflow_final, sonar_test)|>
    bind_cols(sonar_test)

sonar_metrics <- sonar_predictions|>
    metrics(truth = type, estimate = .pred_class)

sonar_conf_mat <- sonar_predictions|>
    conf_mat(truth = type, estimate = .pred_class)

sonar_metrics
sonar_conf_mat
```

<table class="dataframe">
<caption>A tibble: 2 × 3</caption>
<thead>
	<tr><th scope=col>.metric</th><th scope=col>.estimator</th><th scope=col>.estimate</th></tr>
	<tr><th scope=col>&lt;chr&gt;</th><th scope=col>&lt;chr&gt;</th><th scope=col>&lt;dbl&gt;</th></tr>
</thead>
<tbody>
	<tr><td>accuracy</td><td>binary</td><td>0.8490566</td></tr>
	<tr><td>kap     </td><td>binary</td><td>0.6945245</td></tr>
</tbody>
</table>

              Truth
    Prediction  R  M
             R 19  2
             M  6 26

- A new model specification is created with the previously determined best parameter value (i.e., K = 3)
- The recipe and model specification is added to a `workflow()` and we retrain the classifier using the `fit()` function
- The class labels for our test set are predicted using the `predict()` function and `bind_cols()` is applied to add the column of predictions to the original test data creating the predictions data frame
- The estimated accuracy of the classifier is evaluated on the test set using the `metrics()` function
- A confusion matrix is created for the classifier using `conf_mat()` which shows the table of predicted labels and correct labels

## Results

The results of the classification model on the test data are:

#### Accuracy: 84.90%

The confusion matrix shows that out of 47 test instances:

- 19 instances of type "Rock" (labelled as R) were correctly predicted as "Rock" (R)
- 26 instances of type "Mine" (labelled as M) were correctly predicted as "Mine" (M)
- 2 instances of type "Rock" (R) were incorrectly predicted as "Mine" (M)
- 6 instances of type "Mine" (M) were incorrectly predicted as "Rock" (R)

## Discussion

The KNN classification model achieved an accuracy of 84.9% on the test data. The classifier excluded bands X50 to X60 as shown by our EDA we have irrelevant predictors that reduce accuracy, verefied by our accuracy estimate plot. Bands X50 to X60 are similar in pattern across both mines and rocks, so influence from these bands in the classifier does not aid in differentiation between mines and rocks. The classifier correctly predicted the majority of instances for both "Rock" and "Mine" types, but there were some misclassifications with 2 instances of "Rock" being predicted as "Mine" and 6 instances of "Mine" being predicted as "Rock". These accuracies fit within what we expected to find, as both mines and rocks were seen to have different mean measurement values in Fig. 1. Some inaccuracies were also expected as some mean measurement values were similar in value to each other.

A classifier that accurately classifies rocks from naval mines opens possibilities in developing autonomous or A.I. directed applications. With current technology, detecting and clearing naval mines with the use of specialized equipment and human operators analyzing sonar imagery for mine detection can be a resource-intensive costly procedure (Gebhardt et al., 2017). A classifier that can accurately distinguish between rocks and naval mines can potentially lead to reductions in cost and labor. When human operators are responsible for mine detection, there are limiting factors such as human error through “drowsiness or nervousness” (Gebhardt et al., 2017) that may result in missing a naval mine or falsely identifying objects as naval mines; such limitations can be prevented through the development of an accurate classifier that would increase the efficiency of mine clearing operations by reducing the amount of time needed to identify a naval mine without direct mistake-prone human examination. For example, deep learning can be used to classify objects automatically, with little need for human input to extract image features for identification, which is a major drawback for classical image processing (Hożyń, S., 2021). Trends point to the combined use of manpower and deep learning to maximize classification accuracy, whilst minimizing computational costs by allowing deep learning algorithms to focus on parts of sonar images significant towards classification purposes (Hożyń, S., 2021).

## References

Gebhardt, D., Parikh, K., Dzieciuch, I., Walton, M., & Hoang, N. A. V. (2017, September). Hunting for naval mines with deep neural networks. In OCEANS 2017-Anchorage (pp. 1-5). IEEE.

Hożyń, S. (2021). A review of underwater mine detection and classification in sonar imagery. Electronics, 10(23), 2943. https://doi.org/10.3390/electronics10232943

Köhntopp, D., Lehmann, B., Kraus, D., & Birk, A. (2019). Classification and Localization of Naval Mines With Superellipse Active Contours. IEEE Journal of Oceanic Engineering, 44(3), 767–782. https://doi.org/10.1109/JOE.2018.2835218
