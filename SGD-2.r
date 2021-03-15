set.seed(2019)
#View(longley)
require(glmnet)
longley$realGNP =  log(longley$GNP-longley$GNP.deflator)
longley.std=apply(longley, 2,function(x) (x-mean(x))/sqrt(var(x)))



x <- as.matrix(longley.std[, c(3,4,5,8)])
row.names(x)=c()
y <- longley.std[, "Employed"]
m <- length(y)
X<-data.frame(intercept=rep(1,m), x)
X <- as.matrix(X)
#Loss function
compCost<-function(X, y, theta){
  m <- length(y)
  J <- sum((X%*%theta-y)^2)/2*m
  return(J)
}
#SGD for LSE
gradDescent<-function(X, y, init, alpha, num_iters,xtol){
  m <- length(y)
  J_hist <- rep(0, num_iters)
  i = 0
  theta = init
  update = 1
  loss=c()
  while(max(abs(update)>=xtol)&i<num_iters){
    sample=sample(1:nrow(X),size=1,replace=FALSE)
    sam_X=(X[sample,])
    sam_y=y[sample]
    update <- alpha*(1/m)*((sam_X)%*%(sam_X%*%theta -sam_y))
    theta <- theta - update
    loss[i]=compCost(sam_X,sam_y,theta)
    i <- i + 1
  }
  if(i == num_iters){
    print("warning: not converge")
  }
  print(paste('iteration times:',i))
  List <- list(Theta=theta,Loss=loss,inter=i)
  return(List)
}
#SGD for Ridge
ridgeCost<-function(X, y, theta,lambda){
  m <- length(y)
  J <- sum((X%*%theta-y)^2)/2*m+lambda*t(theta)%*%theta
  return(J)
}
R.gradDescent<-function(X, y, init, alpha, num_iters,lambda,xtol){
  m <- length(y)
  i = 0
  theta = init
  update = 1
  cost=rep(0,num_iters)
  while(max(abs(update)>=xtol)&i<num_iters){
    sample=sample(nrow(X),size=1,replace=TRUE)
    sam.X=X[sample,]
    sam.y=y[sample]
    update <- alpha*(1/m)*((sam.X)%*%(sam.X%*%theta - sam.y))+2*lambda*(c(0,theta[-1]))
    theta <- theta - update
    cost[i]=ridgeCost(sam.X, sam.y, theta,lambda)
    i <- i + 1
  }
  if(i == num_iters){
    print("warning: not converge")
  }
  print(paste('iteration times:',i))
  List=list(Theta_Ridge=theta,Cost=cost,inter_Ridge=i)
  return(List)
}
#
cv.ridge = cv.glmnet(x = x, 
                     y = y, 
                     alpha = 0,  
                     nfolds = 3,
                     family = "gaussian",
                     standardize=FALSE,
                     standardize.response=FALSE)
best.lambda = cv.ridge$lambda.min
best.lambda
coef(cv.ridge)
ridge = glmnet(x = x, 
               y = y, 
               alpha = 0,
               lambda = best.lambda,
               family = "gaussian",
               standardize=FALSE,
               standardize.response=FALSE)
coef(ridge)

#parameter
init<-c(0,-0.2,-0.2,0.2,0.9)
init_r <- c(0,-0.2,-0.1,0.4,0.8)
alpha <- .001
num_iters <- 10^6


t(init_r)[-1]%*%init_r[-1]* 0.1214735
#LSE
results <- gradDescent(X, y, init, alpha, num_iters,xtol=1e-6)
print(results)
(coef_lse <- lm(Employed~Unemployed + Armed.Forces + Population  + realGNP
   , data = as.data.frame(longley.std))$coefficient)
results[[1]]
compCost(X,y,results[[1]])
compCost(X,y,coef_lse)# 利用lm()求得的LSE 的 MSE(固定)
#Ridge
results1 <- R.gradDescent(X, y, init=init_r, alpha, num_iters,lambda=best.lambda,xtol=1e-4);results1
results1[1]
ridgeCost(X,y,results1[[1]],lambda=best.lambda)
ridgeCost(X,y,fit1,lambda=best.lambda)
coef(cv.ridge)

fit1=solve(t(X)%*%X+best.lambda*diag(1,5))%*%t(X)%*%y
solve(t(X)%*%X)%*%t(X)%*%y
library(MASS)
(fit <- lm.ridge(Employed~Unemployed + Armed.Forces + Population  + realGNP,data = as.data.frame(longley.std),
                lambda=0.1214735))
as.matrix(fit)
