install.packages("glmnet")
require(glmnet)
install.packages("readxl")
library(readxl)

Affairs<- read.csv("C:/Users/Leo Nian/Downloads/data/Affairs.csv")
Affairs
#查看缺失值
sum(is.na(Affairs))
#查看異常值
library(ggplot2)
boxplot(Affairs$age)
str(Affairs)

table(Affairs$affairs)
Affairs$affairs[Affairs$affairs >= 1] <- 1
a <- sub("female",1,Affairs$gender)
b <- sub("male",0,a)
Affairs$gender <- a                         
Affairs$gender <-b
Affairs$gender <- as.numeric(Affairs$gender)

Affairs$occupation <-as.numeric(Affairs$occupation)
table(Affairs$children)
c <- sub("yes",1,Affairs$children)
d<- sub("no",0,c)
Affairs$children <- c                         
Affairs$children<-d
Affairs$children<- as.numeric(Affairs$children)
Affairs$occupation <- as.character(Affairs$occupation)
set.seed(16)
train.index = sample(x=data.matrix(1:nrow(Affairs)),
                     size=ceiling(0.8*nrow(Affairs)))
train = Affairs[train.index, ]
test = Affairs[-train.index, ]

install.packages("DMwR")
library(DMwR)
train_smote <- SMOTE(affairs ~ ., train,perc.over=600,perc.under=100 )
install.packages("ROSE")
library(ROSE)
train_smote <- ROSE(affairs ~ ., data = train, seed=1)$data
table(train_smote$affairs)
lasso = glmnet(x = as.matrix(Affairs[, 3:10]), 
               y = Affairs[, 2], 
               alpha = 1,
               family = "binomial")

plot(lasso, xvar='lambda', main="Lasso")

#Lasso
cv.lasso = cv.glmnet(x = data.matrix(Affairs[, 3:10]), 
                     y = Affairs[, 2], 
                     alpha = 1,  # lasso
                     family = "binomial")
best.lasso.lambda = cv.lasso$lambda.min
best.lasso.lambda

plot(lasso, xvar='lambda', main="Lasso")
abline(v=log(best.lasso.lambda), col="blue", lty=5.5 )

coef(cv.lasso, s = "lambda.min")

#Forward Stepwise regression
# 1.建立空的線性迴歸(只有截距項)
null = glm(affairs ~ 1, data = train[, 2:10])  
full = glm(affairs ~ ., data = train[, 2:10]) # 建立上界，也就是完整的線性迴歸

# 2.使用step()，一個一個把變數丟進去
forward.glm = step(null, 
                   # 從空模型開始，一個一個丟變數，
                   # 最大不會超過完整的線性迴歸
                   # (一定要加上界 upper=full，不可以不加) 
                   scope=list(lower=null, upper=full), 
                   direction="forward")
summary(forward.glm)
#Backward Stepwise regression
# 1. 先建立一個完整的線性迴歸
full = glm(affairs ~ ., data = train_smote[,2:10])  

# 2. 用`step()`，一個一個把變數移除，看移除哪個變數後 AIC 下降最多
backward.glm = step(full, 
                    scope = list(upper=full), 
                    direction="backward")  
summary(backward.glm)

#logistic regression
fit.forward <- glm(affairs ~ children + religiousness +
                     rating, data=train, family=binomial())
summary(fit.forward)
fit.backward <- glm(affairs ~ age + yearsmarried + children + religiousness +
                      rating, data=train_smote, family=binomial())
summary(fit.backward)
fit.reduced <- glm(affairs ~ age + yearsmarried + religiousness +
                     rating, data=train_smote, family=binomial())
summary(fit.reduced)

anova(fit.reduced,full,test="Chisq")

coef(fit.backward)
#odds
exp(coef(fit.backward))

#ROC Curve
install.packages("pROC")
library(pROC)

pre <- predict(fit.backward,test)
modelroc <- roc(Affairs$affairs,pre)
plot(modelroc, print.auc=TRUE, auc.polygon=TRUE, grid=c(0.1, 0.2),
     grid.col=c("green", "red"), max.auc.polygon=TRUE,
     auc.polygon.col="skyblue", print.thres=TRUE)

test$prob <- predict(fit.backward, 
                     newdata=test,
                     type="response")
test
Affairs$prob <- predict(fit.backward, 
                        newdata=Affairs,
                        type="response")
Affairs$pred[Affairs$prob >= 0.5] <- 1
Affairs$pred[Affairs$prob < 0.5] <- 0
Affairs$pred
cm <- table(Affairs$affairs, Affairs$pred, dnn = c("實際", "預測"))
cm
accuracy <- sum(diag(cm)) / sum(cm)
accuracy

