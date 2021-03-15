####Packages####
library(MASS)
library(ggcorrplot)
library(glmnet) #lasso
library(randomForest)
library(stats)
library(rpart)
install.packages('adabag')
library(adabag)
install.packages('e1071')
library(e1071) 
library(class)
library(ROCR)
library(pROC)
library(caret)
library(ISLR)
data <- read.csv("C:/Users/Leo Nian/Downloads/OnlineNewsPopularity/OnlineNewsPopularity/OnlineNewsPopularity.csv")
data <- data[-1:-2]
str(data)
View(data)

####Preprocessing####
popularity <- ifelse(data$shares>1400,"popular","unpopular")
popularity_glm <- ifelse(data$shares>1400,1,0)
data <- cbind(data,popularity,popularity_glm)
data <- data[-30:-36] #delete weekday
data <- data[-52] #delete share
View(data)
set.seed(2019)
idx <- 1:nrow(data)
idx_s <- sample(idx)
idx_train <- idx_s[1:round(0.7*39644)]
train <- data[idx_train,]
test <- data[-idx_train,]
train_1 <- train[-53]
test_1 <- test[-53]
train_glm <- train[-52]
test_glm <- test[-52]
####EDA####
cont_data <- data[,-c(12:17,30)]
pca_data <- cont_data[-45:-46]
cor_matrix <- cor(cont_data)
ggcorrplot(cor_matrix, hc.order = TRUE, colors = c("#0e02f2", "white", "#f70707"))
cont_data_glm <- cont_data[-45]
train_cont <- cont_data[idx_train,]
train_cont <- train_cont[-46]
test_cont <- cont_data[-idx_train,]
test_cont <- test_cont[-46]
train_cont_glm <- cont_data_glm[idx_train,]
test_cont_glm <- cont_data_glm[-idx_train,]
train_cont_nor <- scale(train_cont[-45], center = TRUE, scale = TRUE)
train_cont_nor <- cbind(train_cont_nor,train_cont[45])
test_cont_nor <- scale(test_cont[-45], center = TRUE, scale = TRUE)
test_cont_nor <- cbind(test_cont_nor,test_cont[45])
train_glm_nor <- cbind(scale(train_glm[-52], center = TRUE, scale = TRUE),train_glm[52])
test_glm_nor <- cbind(scale(test_glm[-52], center = TRUE, scale = TRUE),test_glm[52])
train_cont_glm_nor <- cbind(scale(train_cont_glm[-45], center = TRUE, scale = TRUE),train_cont_glm[45])
test_cont_glm_nor <- cbind(scale(test_cont_glm[-45], center = TRUE, scale = TRUE),test_cont_glm[45])
####PCA####
eigen<-eigen(cor(pca_data)) 
plot(eigen$values,type="h") 
pca<-princomp(scale(pca_data,scale=TRUE,center=TRUE),cor=TRUE) 
loadings(pca) 
summary(pca)
plot(pca, type="line")
abline(a=1,b=0)
####logistic####
all_glm <- glm(formula = popularity_glm ~ ., family = "binomial", data = train_glm_nor)
cont_glm <- glm(formula = popularity_glm ~ ., family = "binomial", data = train_cont_glm)
cont_glm_nor <- glm(formula = popularity_glm ~ ., family = "binomial", data = train_cont_glm_nor)
all_result <- predict(all_glm, test_glm_nor, type = "response")
all_result_Approved <- ifelse(all_result > 0.5, 1, 0)
glm_all <- table(test_glm_nor$popularity_glm,all_result_Approved)
cont_result <- predict(cont_glm,test_cont_glm,type = "response")
cont_result_Approved <- ifelse(cont_result>0.5,1,0)
glm_cont <- table(test_cont_glm$popularity_glm,cont_result_Approved)
cont_result_nor <- predict(cont_glm_nor,test_cont_glm_nor,type = "response")
cont_result_Approved_nor <- ifelse(cont_result_nor>0.5,1,0)
glm_cont_nor <- table(test_cont_glm_nor$popularity_glm,cont_result_Approved_nor)
####Decision Tree####
control<-rpart.control(minisplit=10,minbucket=3,xval=10)
treeorig.all<-rpart(popularity~.,data=train_1,control=control)
treeorig.cont<-rpart(popularity~.,data=train_cont,control=control)
tree.predict <- predict(treeorig.all,test_1)
treecont.predict <- predict(treeorig.cont,test_cont)
rpart_pred_Class <- apply( tree.predict,1,function(one_row) return(colnames(tree.predict)[which(one_row == max(one_row))]))
rpart_pred1_Class <- apply( treecont.predict,1,function(one_row) return(colnames(tree.predict)[which(one_row == max(one_row))]))
treeall<- table(test_1$popularity,rpart_pred_Class)
treecont<- table(test_cont$popularity,rpart_pred1_Class)
####RandomForest####
rf_model <- randomForest(popularity~.,mtry=7,nodesize=3,data = train_1)
rf_model_con <- randomForest(popularity~.,mtry=7,nodesize=3,data=train_cont)
importance(rf_model)
varImpPlot(rf_model)
rfall_predict = predict(rf_model,test_1,type = "prob")
rfcont_predict = predict(rf_model_con,test_cont,type = "prob")
rfall <- table(test_1$popularity,rfall_predict)
rfcont <- table(test_cont$popularity,rfcont_predict)
####bagging####
a.control<-rpart.control(minisplit=10,minbucket=3,cp=0.01) 
bagging_all <- bagging(popularity~., data=train_1,control=a.control) 
bagging_cont <- bagging(popularity~., data=train_cont,control=a.control) 
baggingall_pred <- predict.bagging(bagging_all,test_1)
baggingcont_pred <- predict.bagging(bagging_cont,test_cont)
baggingall <- table(test_1$popularity,baggingall_pred$class)
baggingcont <- table(test_cont$popularity,baggingcont_pred$class)
####boosting####
adaboost_all <- boosting(popularity~., data=train_1, boos=T, mfinal=20)
adaboost_cont <- boosting(popularity~., data=train_cont, boos=T, mfinal=20)
adaboost_all.pred <- predict.boosting(adaboost_all,test_1)
adaboost_cont.pred <- predict.boosting(adaboost_cont,test_cont)
ada_all <- adaboost_all.pred$confusion
ada_cont <- adaboost_cont.pred$confusion
####svm####

c1<-svm(train_cont_nor[,1:44],train_cont_nor[,45],cost=100,gamma=0.2,probability = TRUE )
svm.pred<-predict(c1,test_cont_nor[,1:44],type="prob",probability = TRUE)
table(test_cont$popularity,svm.pred)

####k-NN####
set.seed(2019)
ctrl <- trainControl(method="repeatedcv",repeats = 3) #,classProbs=TRUE,summaryFunction = twoClassSummary)
knnFit <- train(popularity ~ ., data = train_cont, method = "knn", trControl = ctrl, preProcess = c("center","scale"), tuneLength = 20)
#Output of kNN fit
knnFit
plot(knnFit)
knnPredict <- predict(knnFit,newdata = test_cont,type = "prob")
confusionMatrix(knnPredict, test_cont$popularity)

knncv<-knn.cv(train_cont[,1:44],train_cont[,"popularity"], k=1, prob=T)
table(train_cont$popularity,knncv)
table(test_cont$popularity,knncv)
knntest<-knn(train_cont[,1:44],test_cont[,1:44],train_cont[,"popularity"], k=1, prob=T)
knn <- table(test_cont$popularity,knntest)

####evalution####
#pred <- prediction(result, testdata)
#perf <- performance(pred, measure = "tpr", x.measure = "fpr")
#logistic
pred_glm_all <- prediction(all_result,test_glm$popularity_glm)
perf_glm_all <- performance(pred_glm_all,measure = "tpr", x.measure = "fpr")
auc_glm_all <- performance(pred_glm_all, "auc")
pred_glm_cont <- prediction(cont_result,test_cont_glm$popularity_glm)
perf_glm_cont <- performance(pred_glm_cont,measure = "tpr", x.measure = "fpr")
auc_glm_cont <- performance(pred_glm_cont, "auc")
plot(perf_glm_all, col =6, main = "Logistic ROC curve", xlab = "Specificity(FPR)", ylab = "Sensitivity(TPR)")
par(new=T)
plot(perf_glm_cont, col = 4, main = "Logistic ROC curve", xlab = "Specificity(FPR)", ylab = "Sensitivity(TPR)")
abline(0, 1)
text(0.5, 0.5, as.character(auc_glm_all@y.values[[1]]),col=6)
text(0.3, 0.2, as.character(auc_glm_cont@y.values[[1]]),col=4)
legend(0.7, 0.3, c('cont', 'all'), c(4,6))
#decision tree
pred_tree_cont <- prediction(treecont.predict[,2],test_cont$popularity)
perf_tree_cont <- performance(pred_tree_cont,measure = "tpr", x.measure = "fpr")
auc_tree_cont <- performance(pred_tree_cont, "auc")
pred_tree_all <- prediction(tree.predict[,2],test_1$popularity)
perf_tree_all <- performance(pred_tree_all,measure = "tpr", x.measure = "fpr")
auc_tree_all <- performance(pred_tree_all, "auc")
plot(perf_tree_all, col = rainbow(7), main = "Decision Tree ROC curve", xlab = "Specificity(FPR)", ylab = "Sensitivity(TPR)")
par(new=T)
plot(perf_tree_cont, col = 84, main = "Decision Tree ROC curve", xlab = "Specificity(FPR)", ylab = "Sensitivity(TPR)")
abline(0, 1)
text(0.5, 0.5, as.character(auc_tree_all@y.values[[1]]),col=rainbow(7))
text(0.3, 0.2, as.character(auc_tree_cont@y.values[[1]]),col=84)
legend(0.7, 0.3, c('cont', 'all'), c(84,rainbow(7)))
#random forest
pred_rf_cont <- prediction(rfcont_predict[,2],test_cont$popularity)
perf_rf_cont <- performance(pred_rf_cont,measure = "tpr", x.measure = "fpr")
auc_rf_cont <- performance(pred_rf_cont, "auc")
pred_rf_all <- prediction(rfall_predict[,2],test_1$popularity)
perf_rf_all <- performance(pred_rf_all,measure = "tpr", x.measure = "fpr")
auc_rf_all <- performance(pred_rf_all, "auc")
plot(perf_rf_all, col = rainbow(7), main = "Random Forest ROC curve", xlab = "Specificity(FPR)", ylab = "Sensitivity(TPR)")
par(new=T)
plot(perf_rf_cont, col = 84, main = "Random Forest ROC curve", xlab = "Specificity(FPR)", ylab = "Sensitivity(TPR)")
abline(0, 1)
text(0.5, 0.5, as.character(auc_rf_all@y.values[[1]]),col=rainbow(7))
text(0.3, 0.2, as.character(auc_rf_cont@y.values[[1]]),col=84)
legend(0.7, 0.3, c('cont', 'all'), c(84,rainbow(7)))
#bagging
baggingall_pred <- predict.bagging(bagging_all,test_1)
baggingcont_pred <- predict.bagging(bagging_cont,test_cont)
pred_bag_cont <- prediction(baggingcont_pred$prob[,2],test_cont$popularity)
perf_bag_cont <- performance(pred_rf_cont,measure = "tpr", x.measure = "fpr")
auc_bag_cont <- performance(pred_bag_cont, "auc")
pred_bag_all <- prediction(baggingall_pred$prob[,2],test_1$popularity)
perf_bag_all <- performance(pred_bag_all,measure = "tpr", x.measure = "fpr")
auc_bag_all <- performance(pred_bag_all, "auc")
plot(perf_bag_all, col = rainbow(7), main = "Bagging ROC curve", xlab = "Specificity(FPR)", ylab = "Sensitivity(TPR)")
par(new=T)
plot(perf_bag_cont, col = 84, main = "Bagging ROC curve", xlab = "Specificity(FPR)", ylab = "Sensitivity(TPR)")
abline(0, 1)
text(0.5, 0.5, as.character(auc_bag_all@y.values[[1]]),col=rainbow(7))
text(0.3, 0.2, as.character(auc_bag_cont@y.values[[1]]),col=84)
legend(0.7, 0.3, c('cont', 'all'), c(84,rainbow(7)))
#boosting
pred_ada_cont <- prediction(adaboost_cont.pred$prob[,2],test_cont$popularity)
perf_ada_cont <- performance(pred_ada_cont,measure = "tpr", x.measure = "fpr")
auc_ada_cont <- performance(pred_ada_cont, "auc")
pred_ada_all <- prediction(adaboost_all.pred$prob[,2],test_1$popularity)
perf_ada_all <- performance(pred_ada_all,measure = "tpr", x.measure = "fpr")
auc_ada_all <- performance(pred_ada_all, "auc")
plot(perf_ada_all, col = rainbow(7), main = "Boosting ROC curve", xlab = "Specificity(FPR)", ylab = "Sensitivity(TPR)")
par(new=T)
plot(perf_ada_cont, col = 84, main = "Boosting ROC curve", xlab = "Specificity(FPR)", ylab = "Sensitivity(TPR)")
abline(0, 1)
text(0.5, 0.5, as.character(auc_ada_all@y.values[[1]]),col=rainbow(7))
text(0.3, 0.2, as.character(auc_ada_cont@y.values[[1]]),col=84)
legend(0.7, 0.3, c('cont', 'all'), c(84,rainbow(7)))
#svm
pred_svm <- prediction(attr(svm.pred,"probabilities")[,1],test_cont$popularity)
perf_svm <- performance(pred_svm,measure = "tpr", x.measure = "fpr")
auc_svm <- performance(pred_svm, "auc")
plot(perf_svm, col = rainbow(7), main = "SVM ROC curve", xlab = "Specificity(FPR)", ylab = "Sensitivity(TPR)")
abline(0, 1)
text(0.5, 0.5, as.character(auc_svm@y.values[[1]]),col=rainbow(7))
#knn
knntest<-knn(train_cont[,1:44],test_cont[,1:44],train_cont[,"popularity"], k=1, prob=T)
pred_knn <- prediction(knnPredict[,2],test_cont$popularity)
perf_knn <- performance(pred_knn,measure = "tpr", x.measure = "fpr")
auc_knn <- performance(pred_knn, "auc")
plot(perf_knn, col = rainbow(7), main = "KNN ROC curve", xlab = "Specificity(FPR)", ylab = "Sensitivity(TPR)")
abline(0, 1)
text(0.5, 0.5, as.character(auc_knn@y.values[[1]]),col=rainbow(7))

#continuous
plot(perf_glm_cont, col = 4, xlab = "Specificity(FPR)", ylab = "Sensitivity(TPR)")
par(new=T)
plot(perf_tree_cont, col = 5, xlab = "Specificity(FPR)", ylab = "Sensitivity(TPR)")
par(new=T)
plot(perf_rf_cont, col = 6 , xlab = "Specificity(FPR)", ylab = "Sensitivity(TPR)")
par(new=T)
plot(perf_bag_cont, col = 7 , xlab = "Specificity(FPR)", ylab = "Sensitivity(TPR)")
par(new=T)
plot(perf_ada_cont, col = 8, xlab = "Specificity(FPR)", ylab = "Sensitivity(TPR)")
par(new=T)
plot(perf_svm, col = 9, xlab = "Specificity(FPR)", ylab = "Sensitivity(TPR)")
par(new=T)
plot(perf_knn, col = 10, main = "Continuous ROC curve",xlab = "Specificity(FPR)", ylab = "Sensitivity(TPR)")
abline(0, 1)
legend(0.7, 0.7, c('logisitc', 'tree','rf','bag','ada','svm','knn'),4:10) 
#all
plot(perf_glm_all, col = 4, xlab = "Specificity(FPR)", ylab = "Sensitivity(TPR)")
par(new=T)
plot(perf_tree_all, col = 5, xlab = "Specificity(FPR)", ylab = "Sensitivity(TPR)")
par(new=T)
plot(perf_rf_all, col = 6, xlab = "Specificity(FPR)", ylab = "Sensitivity(TPR)")
par(new=T)
plot(perf_bag_all, col = 7, xlab = "Specificity(FPR)", ylab = "Sensitivity(TPR)")
par(new=T)
plot(perf_ada_all, col = 8, main = "all ROC curve",xlab = "Specificity(FPR)", ylab = "Sensitivity(TPR)")
par(new=T)
abline(0, 1)
legend(0.7, 0.7, c('logisitc', 'tree','rf','bag','ada'),4:8) 