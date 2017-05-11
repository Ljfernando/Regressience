rm(list = ls())
cat("\014")
library(ISLR)
library(MASS)
library(DAAG)
library(glmnet)
library(leaps)
library(ggplot2)
library(rpart)
library(randomForest)
library(rpart.plot)
library(e1071)
library(FNN)
library(magrittr)
library(boot)
setwd("/Users/lancefernando/Desktop/USFSpring2017/BSDS100/Final")

lm.cv <- function(X, y){
  # Runs linear regression on X using y as the target variable.
  # Uses K-fold CV with different values for K and determines the optimal K
  # 
  # Args:
  #   X: Dataframe of predictors
  #   y: Vector of target values
  #
  # Returns:
  #   List containing the following elements:
  #     plot: ggplot bar-graph of MSE values for different K-folds
  #     errors: vector of MSE values for different K-folds
  #     bestK: integer value of the optimal number of folds
  #     mse: integer MSE CV value using the optimal number of folds
  
  
  cat("Linear Model Formula: y~", paste(names(X), collapse = " + "), "\n")
  
  # Vector containing different values of k-folds
  k <- c(2, 5, 10, 15)
  
  # Empty vector to store MSE values
  k.cv.error <- rep(0, length(k))
  names(k.cv.error) <- k
  
  #all.levels <- sapply(X, levels) %>% rm.null()
  
  #print(all.levels)
  
  for(i in 1:length(k)){
    
    set.seed(1)
    #Creating random sample using k-folds
    folds = sample(1:k[i], nrow(X), replace = TRUE)
    cv.error = rep(0, k[i])
    
    for(j in 1:k[i]){
      train <- which(folds != i)
      
      #Fitting linear model on current training folds
      lm.fit <- glm(y~., data = X, subset = train)
      
      #lm.fit$xlevels <- union.levels(lm.fit$xlevels, all.levels)
      
      #Predicting on current testing fold
      lm.pred <- predict.lm(lm.fit, newdata = X[-train,])
      
      #Calculating MSE and populating the vector
      cv.error[j] <- mean((y[-train] - lm.pred)^2, na.rm = TRUE)
    }
    
    #Calculating the mean MSE for all models using respective k-folds
    k.cv.error[i] <- mean(cv.error, na.rm = TRUE)
  }
  
  #Index of the lowest MSE
  min.cv <- which.min(k.cv.error)
  
  cat(paste("Lowest MSE: ", k.cv.error[min.cv], "\n", sep = ""))
  cat(paste("Number of Folds: ", k[min.cv], "\n\n", sep = ""))
  
  myplot <- ggplot() + 
    geom_col(mapping = aes(x = factor(as.numeric(names(k.cv.error))),
                           y = k.cv.error),
             fill = "red") +
    geom_col(mapping = aes(x = factor(k[min.cv]), y = k.cv.error[min.cv]),
             fill = "green") + 
    geom_text(aes(x = factor(as.numeric(names(k.cv.error))),
                  y = k.cv.error/2),
              label= k.cv.error, color = "black") + 
    labs(x = ("Number of Folds"), y = "MSE",
         title = "MSE Values For Different Values of K in K-Fold Cross Validations") +
    coord_flip()
  
  return(list(plot = myplot,
              errors = k.cv.error,
              bestK = as.numeric(names(min.cv)),
              mse = k.cv.error[min.cv]))
}

variable.selection <- function(X, y, metric, nvmax, nbest){
  # Performs subset selection methods using predictors in X and target y
  # Evaluates the best combination of features based on provided metric
  #
  # Args:
  #   X: Dataframe of predictors
  #   y: Vector of target values
  #   metric: Type of metric used to evaluate the predictor combination
  #           Possible values include rsq, rss, adjr2, cp, bic
  #   nvmax: Maximum size of subsets to examine
  #   nbest: Number of subsets of each size to record
  #
  # Returns:
  #   List containing the following elements:
  #     metricPlot: ggplot scatterplot of (metric value ~ predictor size) for
  #                 subset selection method
  #     msePlot: ggplot bar-graph of the best MSE for each subset selection method
  #     best: vector of predictor names obtained from optimal best subset method
  #     forward: vector of predictor names obtained from optimal forward stepwise method
  #     backward: vector of predictor names obtained from optimal forward stepwise method
  #     best.fit: linear model object using best subset predictors
  #     forward.fit: linear model object using forward stepwise predictors
  #     backward: linear model object using backward stepwise predictors
  
  cat("~~~~~~Algorithmic Model Selection(Best, Forward and Backward Subset)~~~~\n\n")
  
  #Creating regsubset models using each method
  best.fit = regsubsets(y~., data = X,
                        nvmax = nvmax, nbest = nbest)
  forward.fit = regsubsets(y ~., data = X,
                           nvmax = nvmax, nbest = nbest, method = "forward")
  backward.fit = regsubsets(y~., data = X,
                            nvmax = nvmax, nbest = nbest, method = "backward")
  
  #Creating summary info objects to be used later to extract optimal models
  best.summary <- summary(best.fit)
  forward.summary <- summary(forward.fit)
  backward.summary <- summary(backward.fit)
  
  valid.met <- c("rsq", "rss", "adjr2", "cp", "bic")
  
  #These metrics are optimal at small values
  #Other metrics are optimal at larger values
  met.use.min <- c("rss", "cp", "bic")
  
  if(metric %in% valid.met){ #Ensures a valid metric is used
    
    if(metric %in% met.use.min){
      #Notice which.min() is used to find optimal model size based on metric
      best.model <- which.min(eval(parse(text = paste("best.summary$", metric, sep = ""))))
      forward.model <- which.min(eval(parse(text = paste("forward.summary$", metric, sep = ""))))
      backward.model <- which.min(eval(parse(text = paste("backward.summary$", metric, sep = ""))))
    }
    else{
      #Notice which.max() is used to find optimal model size based on metric
      best.model <- which.max(eval(parse(text = paste("best.summary$", metric, sep = ""))))
      forward.model <- which.max(eval(parse(text = paste("forward.summary$", metric, sep = ""))))
      backward.model <- which.max(eval(parse(text = paste("backward.summary$", metric, sep = ""))))
    }
    
    cat(paste("Below are the combinations of variables whose models provided the best ", metric, " value.\n", sep = ""))
    
    
    myplot1 <- ggplot() +
      geom_point(mapping = aes(x = factor(1:length(eval(parse(text = paste("best.summary$", metric, sep = ""))))),
                               y = eval(parse(text = paste("best.summary$", metric, sep = ""))),
                               color = "Best")) +
      geom_point(mapping = aes(x = factor(1:length(eval(parse(text = paste("forward.summary$", metric, sep = ""))))),
                               y = eval(parse(text = paste("forward.summary$", metric, sep = ""))),
                               color = "Forward")) +
      geom_point(mapping = aes(x = factor(1:length(eval(parse(text = paste("backward.summary$", metric, sep = ""))))),
                               y = eval(parse(text = paste("backward.summary$", metric, sep = ""))),
                               color = "Backward")) +
      labs(x = "Number of Predictors", y = toupper(metric),
           colour = "Subset Method",
           title = paste(toupper(metric), " Values for Variable Selection Methods", sep = "")) 
    
    
    cat("\n\nCalculating MSE For Each Combination of Predictors That Gave The Best Metric Value of Each Selection Algorithm\n")
    
    # Running lm.cv() for each optimal model size from each selection method
    cat("Best Subset Selection: \n")
    cat(paste(best.model, " Variables for best subset selection method\n", sep = ""))
    sub.best <- lm.cv(X = X[, names(coef(best.fit, best.model))[ -1]], y = y)
    cat("Forward Subset Selection: \n")
    cat(paste(forward.model, " Variables for forward stepwise selection method.\n", sep = ""))
    sub.forward <- lm.cv(X = X[, names(coef(forward.fit, forward.model))[-1]], y = y)
    cat("Backward Subset Selection: \n")
    cat(paste(backward.model, " Variables for backward stepwise selection method.\n", sep = ""))
    sub.backward <- lm.cv(X = X[, names(coef(backward.fit, backward.model))[-1]], y = y)
    
    #Extracting MSE of linear model
    mses <- c(sub.best$mse, sub.forward$mse, sub.backward$mse)
    names(mses) <- c("Best", "Forward", "Backward")
    
    #Creating data frame to make plotting easier
    subset.frame <- data.frame(Method = names(mses),
                               MSE = mses)
    
    myplot2 <- ggplot() + 
      geom_col(data = subset.frame,
               mapping = aes(x = Method, y = MSE, fill = Method),
               show.legend = FALSE) + 
      geom_text(data = subset.frame,
                mapping = aes(x = Method, y = MSE/2, label = MSE),
                color = "white") +
      labs(title = "MSE Value For Each Subset Selection Method") +
      coord_flip()
    
    
    return(list(metricPlot = myplot1,
                msePlot = myplot2,
                best = names(coef(best.fit, best.model))[ -1],
                forward = names(coef(forward.fit, forward.model))[-1],
                backward = names(coef(backward.fit, backward.model))[-1],
                best.fit = sub.best,
                forward.fit = sub.forward,
                backward.fit = sub.backward))
  }
  else
    #Enters here if metric provided is not valid
    print("Invald Metric.")
}


shrinkage <- function(X, y, k, alpha){
  # Performs shrinkage methods (i.e., Ridge Regression and the Lasso) using 
  # predictors X and target y. 
  #
  # Args:
  #   X: Dataframe of predictors
  #   y: Vector of target values
  #   k: number of folds used for CV
  #   alpha: binary input determinig type of shrinkage method
  #          (0 for Ridge, 1 for Lasso)
  #
  # Returns:
  #   List containing the following elements:
  #     plot: ggplot lineplot mse~lambda portraying optimum lambda value
  #     lambda: numeric value of the tuned lambda parameter
  #     mse: MSE value using optimal lambda parameter
  #     coeffs: Coefficient estimates using optimal lambda parameter
  #     shrink.fit: glmnet object of regularized linear model
  
  set.seed(1)
  
  grid = seq(0, 100, 0.1) # Possible values for lambda
  
  # Printing statements for what type of shrinkage method is executed
  if(alpha == 0)
    cat("\n~~~~~Ridge Regression~~~~~\n")
  else
    cat("\n~~~~~The Lasso~~~~~\n")
  
  cat("lambda grid: seq(0, 100, 0.1)\n")
  cat("Using ", k, "-fold cv\n", sep = "")
  
  # Running shrinkage method using k-folds for all possible lambda values in grid
  cv.out = cv.glmnet(model.matrix(y~., X), y, alpha = alpha,
                     lambda = grid,
                     nfolds = k)
  
  best.lam = cv.out$lambda.min # Lambda value providing best MSE
  
  best.lam.ind <- which(cv.out$lambda == best.lam) # Index of best lambda value
  
  best.mse = cv.out$cvm[best.lam.ind] # MSE of model using best lambda value
  
  cat("Model w/ lowest MSE\n")
  cat("Lambda: ", best.lam, "; MSE: ", best.mse, "\n")
  cat("Coefficients: \n")
  print(coef(cv.out, s = "lambda.min"))
  cat("\n\n")
  
  myplot <- ggplot() + 
    geom_line(mapping = aes(x = cv.out$lambda, y = cv.out$cvm), size = 1, color = "red") + 
    geom_point(mapping = aes(x = cv.out$lambda[best.lam.ind], y = cv.out$cvm[best.lam.ind]), size = 3,  color = "green") +
    labs(x = "Lambda Values", y = "CV MSE", title = "MSE Values Based On Various Lambda Penalization Values")
  
  return(list(plot = myplot,
              lambda = best.lam,
              mse = best.mse,
              coeffs = coef(cv.out, s = "lambda.min"),
              shrink.fit = cv.out))
}
both.shrinkage <- function(X, y, k){
  # Uses shrinkage() and performs both Ridge Regression and the Lasso
  # on X predictors using target y and k folds.
  # Creates a plot comparing both methods
  #
  # Args:
  #   X: Dataframe of predictors
  #   y: Vector of target values
  #   k: number of folds used for CV
  #
  # Returns:
  #   List containing the following elements:
  #     plot: ggplot multi-line plot comparing shrinkage methods
  #     ridge: ridge object returned from running shrinkage() method w/ alpha = 0 
  #     lasso: lasso object returned from running shrinkage() method w/ alpha = 1
  
  cat("\n\n~~~~~~~~Running Shrinkage Methods~~~~~~~~\n\n")
  # Running shrinkage() methods to create ridge and lasso objects
  ridge <- shrinkage(X = X, y = y, k = k, alpha = 0)
  lasso <- shrinkage(X = X, y = y, k = k, alpha = 1)
  
  # Obtaining the glmnet models
  ridge.out <- ridge$shrink.fit
  lasso.out <- lasso$shrink.fit
  
  # Plotting results to compare methods
  myplot <- ggplot() + 
    geom_line(mapping = aes(x = ridge.out$lambda, y = ridge.out$cvm, color = "ridge"), size = 1) + 
    geom_point(mapping = aes(x = ridge$lambda, y = ridge$mse, color = "ridge"), size = 3) +
    geom_line(mapping = aes(x = lasso.out$lambda, y = lasso.out$cvm, color = "lasso"), size = 1) + 
    geom_point(mapping = aes(x = lasso$lambda, y = lasso$mse, color = "lasso"), size = 3) +
    labs(x = "Lambda Values", y = "MSE", title = "MSE Values Based On Various Lambda Penalization Values",
         colour = "Shrinkage Method")
  
  return(list(plot = myplot,
              ridge = ridge,
              lasso = lasso))
}

regressTree <- function(X, y, k, control){
  # Runs a regression tree algorithm using predictor space X on target y
  # and runs k-fold cv to evaluate each model. Utilizes cost-complexity pruning to
  # evaluate subtrees and prune the tree accordingly. Prints information and develops plots
  # for the cross validated model with the lowest mse.
  #
  # Args:
  #   X: Dataframe of predictors
  #   y: Vector of target values
  #   k: number of folds used for CV
  #   control: REGRESSION TREE; list of options passed into rpart.control() that control the rpart
  #             algorithm (default is the default in rpart.control())
  
  # Returns:
  #   List containing the following elements:
  #     mse: average mse from CV
  #     tree: rpart object model with lowest mse from cv 
  #     prunedTree: rpart object pruned version of tree
  
  cat("\n\n~~~~~~~~Running Regression Trees~~~~~~~~\n\n")
  
  set.seed(1)
  #Creating random sample using k-folds
  folds = sample(1:k, nrow(X), replace = TRUE)
  cv.error = rep(0, k)
  
  trees.list <- list()
  # First we create k models and evaluate their test accuracy.
  # We then average all the test accuracies to get an average mse
  for(i in 1:k){
    
    train <- which(folds != i) # creating subset for current cv fold
    
    tree.fit <- rpart(y~., data = X, subset = train, control = control) # Creating tree model
    
    trees.list[[i]] <- tree.fit # Adding to list
    
    # Following code accesses the cp value with the lowest cv error
    best <- tree.fit$cptable[which.min(tree.fit$cptable[,"xerror"]),"CP"] 
    
    prune.tree <- prune(tree.fit, cp = best) # Pruning tree based on above cp value
    
    tree.pred <- predict(prune.tree, newdata = X[-train, ]) # calculating predictions from model
    
    cv.error[i] <-  mean((tree.pred - y[folds == i])^2, na.rm = TRUE)
  }
  
  mse <- mean(cv.error)
  
  cat(paste("MSE of regression tree modeling using K-Fold Cross Validation: ", mse, "\n", sep = ""))
  
  par(mfrow = c(1,3))
  
  tree.fit <- trees.list[[which.min(cv.error)]] # Accessing cv model with lowest cv mse
  
  
  rpart.plot(tree.fit, main = "Full Tree (unpruned)") # Plotting full tree
  
  plotcp(tree.fit) # Plotting CV errors
  
  best <- tree.fit$cptable[which.min(tree.fit$cptable[,"xerror"]),"CP"] # Accessing best cp for pruning
  
  pruned <- prune(tree.fit, cp = best) # Pruning tree based on cp value accessed above
  
  rpart.plot(pruned, main = "Pruned Tree") # Plotting pruned tree
  
  cat("\n\n~~~~~Information For Best Cross Validated Model~~~~~\n")
  printcp(tree.fit)
  cat("\nIn order to prune, we choose the tree size that minimizes cv error(xerror) from the\n")
  cat("above table.")
  cat("\nWe then grab the respective CP value and use that as the parameter for pruning")
  cat("\nIn this case our CP value is: ", best)
  cat("\n\n~~~~~After Pruning Tree~~~~~\n")
  print(pruned)
  
  return (list(mse = mse,
               tree = tree.fit,
               prunedTree = pruned))
}

baggingAndRF <- function(X, y, k, maxnodes){
  # Runs a bagging trees and random forests using predictor space X on target y
  # and runs k-fold cv to evaluate each model. The only difference between bagged trees
  # and the random forest algorithm is the hyperparameter mtry which specifies the number of
  # predictors to use in creating a model. Bagging uses the whole predictor space while random
  # forests uses a smaller subspace to reduce the correlation between trees.
  #
  # Args:
  #   X: Dataframe of predictors
  #   y: Vector of target values
  #   k: number of folds used for CV
  #   maxnodes: Maximum number of terminal nodes a tree in the forest can have 
  #             
  # Returns:
  #   List containing the following elements:
  #     mse.bag: CV mse from bagging trees algorithm
  #     mse.rf: CV mse from random forests algorithm
  
  
  cat("\n\n~~~~~~~~Running Bagging Trees~~~~~~~~\n\n")
  set.seed(1)
  #Creating random sample using k-folds
  folds = sample(1:k, nrow(X), replace = TRUE)
  cv.error = rep(0, k)
  
  # Running k-fold cv
  # Taking average of k models to evaluate cv mse
  
  for(i in 1:k){
    train <- which(folds != i)
    
    # Creating bagging model
    # notice mtry=ncol(X) which makes this a bagging algorithm as opposed to random forest
    bag.fit <- randomForest(y~., data = X, subset = train,
                            mtry = ncol(X), importance = TRUE,
                            maxnodes = maxnodes)
    
    yhat.bag = predict(bag.fit ,newdata = X[-train ,]) #Calculating predictions
    cv.error[i] <- mean((yhat.bag - y[-train])^2) #Populating cv.error array
  }
  
  mse.bag <- mean(cv.error) # Average of k bagging errors
  
  cat(paste("CV MSE of Bagging Model: ", mse.bag, "\n", sep = ""))
  cat(paste("Using ", ncol(X), " Predictors to Construct Each Tree." ))
  
  cat("\n\n~~~~~~~~Running Random Forest~~~~~~~~\n\n")
  
  # Same deal as above
  set.seed(1)
  folds = sample(1:k, nrow(X), replace = TRUE)
  cv.error = rep(0, k)
  
  # Optimal amount of predictors to use for random forests is the square root of
  # the amount of predictors used, rounded up. This allows for lowering the correlation 
  # between models, thus improving the variance.
  numPred = ceiling(sqrt(ncol(X)))
  for(i in 1:5){
    train <- which(folds != i)
    
    # Notice mtry = numPred
    bag.fit <- randomForest(y~., data = X, subset = train,
                            mtry = numPred, importance = TRUE,
                            maxnodes = maxnodes)
    
    yhat.bag = predict(bag.fit ,newdata = X[-train ,])
    cv.error[i] <- mean((yhat.bag - y[-train])^2)
  }
  
  mse.rf <- mean(cv.error)
  cat(paste("CV MSE of Random Forest Model: ",(mean(cv.error)), "\n" , sep = ""))
  cat(paste("Using ", numPred, " Predictors to Construct Each Tree.\n" ))
  return(list(mse.bag = mse.bag,
              mse.rf = mse.rf))
}

knn.regression <- function(X, y, algorithm){
  # Run K nearest neighbors algorithm that predicts the target value y
  # given the predictor space X where each prediction is the average target value
  # of the K nearest neighbors based on euclidean distance.
  #
  # Args:
  #   X: Dataframe of predictors
  #   y: Vector of target values
  #   algorithm: KNN; Specific nearest neighbor search algorithm. 
  #             (default is the same as the knn.reg() default)
  #
  # Returns:
  #   List containing the following elements:
  #     plot: ggplot scatterplot of mse values for different values of k
  #     mse: CV mse of best knn model 
  #     knn.fit: knn model object with best value k
  
  cat("\n\n~~~~~~~~Running K Nearest Neighbors~~~~~~~~\n\n")
  k.vals <- c(1, 3, 5, 10, 25, nrow(X) - 1) # Vector of different values of k to test
  
  cat("Testing the following values for k: \n")
  cat("\t", k.vals, "\n", sep = "   ")
  
  loocv.errors <- rep(0, 5)# Empty vector which will hold the mse errors for each model
  knn.models <- list()
  #Creating a model for each value K and storing its mse
  
  for(i in 1:6){
    knn.fit <- knn.reg(train = X, y = y, k = k.vals[i], algorithm = algorithm)
    
    mse.loocv <- knn.fit$PRESS/knn.fit$n # Calculates the mse of respective model
    
    loocv.errors[i] <- mse.loocv
    
    knn.models[[i]] <- knn.fit
  }
  
  
  mse <- loocv.errors[which.min(loocv.errors)] # Accessing MSE of best KNN model
  
  best.knn <- knn.models[which.min(loocv.errors)]# Accessing best KNN model
  
  best.k <- k.vals[which.min(loocv.errors)]# Optimal K value
  
  cat("\nLOOCV MSE of best model evaluated by ", best.k, 
      " Nearest Neighbors: ", mse, "\n", sep = "")
  
  myplot <- ggplot(mapping = aes(x = as.factor(k.vals), y = loocv.errors)) + 
    geom_col(fill = "#F8766D") + 
    geom_col(mapping = aes(x = as.factor(best.k), y = mse), fill = "#00BFC4") +
    geom_text(mapping = aes(y = mse/2, label = loocv.errors), color = "white") +
    labs(title = "KNN LOOCV MSE for Different Values K",
         x = "Nearest Neighbors",
         y = "LOOCV MSE") +
    coord_flip()
  
  print(myplot)
  return(list(plot = myplot,
              mse = mse,
              knn.fit <- best.knn))
}

multiplot <- function(..., plotlist=NULL, file, cols=1, layout=NULL) {
  # Multiple plot function
  # Function obtained from Cookbook for R by Winston Chang
  # http://www.cookbook-r.com
  #
  # ggplot objects can be passed in ..., or to plotlist (as a list of ggplot objects)
  # - cols:   Number of columns in layout
  # - layout: A matrix specifying the layout. If present, 'cols' is ignored.
  #
  # If the layout is something like matrix(c(1,2,3,3), nrow=2, byrow=TRUE),
  # then plot 1 will go in the upper left, 2 will go in the upper right, and
  # 3 will go all the way across the bottom.
  #
  library(grid)
  
  # Make a list from the ... arguments and plotlist
  plots <- c(list(...), plotlist)
  
  numPlots = length(plots)
  
  # If layout is NULL, then use 'cols' to determine layout
  if (is.null(layout)) {
    # Make the panel
    # ncol: Number of columns of plots
    # nrow: Number of rows needed, calculated from # of cols
    layout <- matrix(seq(1, cols * ceiling(numPlots/cols)),
                     ncol = cols, nrow = ceiling(numPlots/cols))
  }
  
  if (numPlots==1) {
    print(plots[[1]])
    
  } else {
    # Set up the page
    grid.newpage()
    pushViewport(viewport(layout = grid.layout(nrow(layout), ncol(layout))))
    
    # Make each plot, in the correct location
    for (i in 1:numPlots) {
      # Get the i,j matrix positions of the regions that contain this subplot
      matchidx <- as.data.frame(which(layout == i, arr.ind = TRUE))
      
      print(plots[[i]], vp = viewport(layout.pos.row = matchidx$row,
                                      layout.pos.col = matchidx$col))
    }
  }
}



regressience <- function(X,
                         y,
                         metric = "bic",
                         nvmax = ncol(X) - 2,
                         nbest = 1,
                         control = rpart.control(),
                         maxnodes = NULL,
                         algorithm=c("kd_tree", "cover_tree", "brute")){
  # Regressience is a "one-stop-shop" regression function that conducts various regression
  # algorithms to predict the response variable y based on the predictor space X.
  # Those algorithms include linear regression (with subset selection methods), shrinkage
  # methods, regression tree ensembles and K nearest neighbors(KNN). Resulting information     
  # regarding each modeling fit are printed to the console in addition to supplemental 
  # explanatory plots.
  # 
  #
  # Args:
  #   X: Dataframe of predictors
  #   y: Vector of target values
  #   sub.metric: SUBSET SELECTION; optional string providing the subset selection metric
  #       (cp, bic, adjr2, r2)
  #   metric: SUBSET SELECTION; Type of metric used to evaluate the predictor combination
  #           Possible values include rsq, rss, adjr2, cp or the default bic
  #   nvmax: SUBSET SELECTION; Maximum size of subsets to examine (default is predictor space - 2)
  #   nbest: SUBSET SELECTION; Number of subsets of each size to record (default is 1)
  #   control: REGRESSION TREE; list of options passed into rpart.control() that control the rpart
  #             algorithm (default is the default in rpart.control())
  #   maxnodes: TREE ENSEMBLE; Maximum number of terminal nodes a tree in the forest can have 
  #             (if NULL, default is trees are grown to maximum possible size)
  #   algorithm: KNN; Specific nearest neighbor search algorithm. 
  #             (default is the same as the knn.reg() default)
  #               
  
  #Ensuring proper data frame and response variables are inputted.
  stopifnot(is.data.frame(X) | is.matrix(X),
            dim(X)[1] == length(y),
            is.vector(y),
            is.numeric(y),
            !anyNA(y) & !anyNA(X))
  #Creating working dataframe with dummy variables
  X <- as.data.frame(model.matrix(y~., X)[, -1])
  
  #Running linear regression on full dataset
  lm.fit <- lm.cv(X = X, y = y)
  
  #Running subset selection methods
  sub.select <- variable.selection(X = X, y = y,  metric = metric, nvmax = nvmax, nbest = nbest)
  
  #Running shrinkage methods
  shrinkage <- both.shrinkage(X = X, y = y, k = lm.fit$bestK)
  
  #Plotting all ggplot objects from the above calls
  multiplot(lm.fit$plot, sub.select$metricPlot, sub.select$msePlot, shrinkage$plot, cols = 2)
  
  #Running regression tree method
  trees <- regressTree(X = X, y = y, k = lm.fit$bestK, control = control)
  
  #Running baggin and random forest methods
  bagAndRF <- baggingAndRF(X = X, y = y, k = lm.fit$bestK, maxnodes = maxnodes)
  
  #Running knn methods
  knn.regression <- knn.regression(X = X, y = y, algorithm = algorithm)
  
  #Printing the MSE for each model
  cat("\n\n~~~~~~~~~~~~~Each Different Model's CV MSE~~~~~~~~~~~~~\n\n")
  cat(paste("Full Linear Regression MSE: ", lm.fit$mse, "\n", sep = ""))
  cat("Variable Selection Algorithm Models: \n")
  cat(paste("\t\tBest Subset Selection MSE: ", sub.select$best.fit$mse, "\n", sep = ""))
  cat(paste("\t\tForward Stepwise Selection MSE: ", sub.select$forward.fit$mse, "\n", sep = ""))
  cat(paste("\t\tBackward Stepwise Selection MSE: ", sub.select$backward.fit$mse, "\n", sep = ""))
  cat("Shrinkage Methods \n")
  cat(paste("\t\tRidge Regression MSE: ", shrinkage$ridge$mse, "\n", sep = ""))
  cat(paste("\t\tThe Lasso MSE: ", shrinkage$lasso$mse, "\n", sep = ""))
  cat("Regression Tree Methods: \n")
  cat(paste("\t\tSingle Regression Tree MSE: ", trees$mse, "\n", sep = ""))
  cat(paste("\t\tBagging Trees MSE: ", bagAndRF$mse.bag, "\n", sep = ""))
  cat(paste("\t\tRandom Forests MSE: ", bagAndRF$mse.rf, "\n", sep = ""))
  cat(paste("K Nearest Neighbors Model MSE: ", knn.regression$mse, "\n", sep = ""))
  
  # Creating dataframe of models and their respective mse values
  models <- c("Linear Regression",
              "Best Subset",
              "Forward Subset",
              "Backward Subset",
              "Ridge",
              "Lasso",
              "Regression Tree",
              "Bagging Trees",
              "Random Forests",
              "KNN")
  models.mse <- c(lm.fit$mse,
                  sub.select$best.fit$mse,
                  sub.select$forward.fit$mse,
                  sub.select$backward.fit$mse,
                  shrinkage$ridge$mse,
                  shrinkage$lasso$mse,
                  trees$mse,
                  bagAndRF$mse.bag,
                  bagAndRF$mse.rf,
                  knn.regression$mse)
  
  mse.df <- data.frame(models = models,
                       mse = models.mse)
  
  
  #Creating bar plot to compare the all models
  ggplot(data = mse.df, mapping = aes(x = models, y = mse)) + 
    geom_col(mapping = aes(fill = mse)) + 
    geom_text(mapping = aes(x = models, y = mse/2, label = mse), color = "white") + 
    coord_flip() + 
    scale_fill_gradient(low="#00BFC4", high="purple") + 
    labs(title = "MSE Values Of Each Model")
}



Hitters <- Hitters

Hitters <- Hitters[which(!is.na(Hitters$Salary)),]

regressience(X = Hitters[, -19], y = Hitters$Salary, sub.metric = "cp")

regressience(X = Boston[, -14], y = Boston$medv, sub.metric = "bic")

train <- read.csv("final.csv", header = TRUE)

regressience(X = train[, -21], y = train$SalePrice)

