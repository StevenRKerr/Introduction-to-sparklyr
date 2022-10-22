library(sparklyr)
library(sparkxgb)
library(dplyr)

library(modeldata)
data("attrition", package = "modeldata")

# Only needs to be run once
# spark_install('3.3')

#df = read.csv('G:/My Drive/Teaching/Big data analytics/Projects/Breast cancer gene expression/METABRIC_RNA_Mutation.csv')

sc = spark_connect(master = "local")

# Read cars into spark
cars = copy_to(sc, mtcars, overwrite = TRUE)

spark_write_csv(collect(cars), 'cars.csv')



cars = cars %>%
  mutate(transmission = ifelse(am == 0, "automatic", "manual"))

car_group = cars %>%
  group_by(cyl) %>%
  summarise(mpg = sum(mpg, na.rm = TRUE)) %>%
  collect()

ggplot(aes(as.factor(cyl), mpg), data = car_group) + 
  geom_col(fill = "#999999") + coord_flip()


# Web management interface
spark_web(sc)

# Plot
select(cars, hp, mpg) %>%
  sample_n(100) %>%
  collect() %>%
  plot()

summarize_all(cars, mean)

# Linear regression
ols_model = ml_linear_regression(cars, mpg ~ hp + disp)

# Logistic regression
lr = ml_logistic_regression(
  cars, am ~ hp + disp
)

validation_summary = ml_evaluate(lr, cars)


cars = mutate(cars, 
                carb = case_when(
                  carb == 1 ~ 0,
                  carb == 2 ~ 1,
                  carb == 3 ~ 2,
                  carb == 4 ~ 3,
                  carb == 5 ~ 4,
                  carb == 6 ~ 5,
                  carb == 8 ~ 6,
                  TRUE ~ carb))

# MLPs
mlp = ml_multilayer_perceptron_classifier(
  cars,
  am ~ hp + disp, 
  layers = c(2, 8, 8, 2)
)

mlp = ml_multilayer_perceptron_classifier(
  cars,
  carb ~ hp + disp,
  layers = c(2, 5, 6)
)

predictions = ml_predict(mlp, cars)

spark_disconnect(sc)

