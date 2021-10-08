### load the data
setwd("C:/Users/Clifford/Desktop/masters/IST 707 - Data Mining/investment company project")
invest <- read.csv("investment data.csv")

### munging
# simplify data frame for neural network processing
invest <- invest[-1:-2,1:19]
colnames(invest) <- c("sleeve", "minimum", "fee", "yr1.ret", "yr1.rank", "yr1.ws",  
                          "yr3.ret", "yr3.rank", "yr3.ws", "yr5.ret", "yr5.rank", "yr5.ws",
                          "yr7.ret", "yr7.rank", "yr7.ws", "yr10.ret", "yr10.rank", "yr10.ws",
                          "score")

# convert returns into numerical form
invest$yr1.ret <- as.numeric(invest$yr1.ret)
invest$yr3.ret <- as.numeric(invest$yr3.ret)
invest$yr5.ret <- as.numeric(invest$yr5.ret)
invest$yr7.ret <- as.numeric(invest$yr7.ret)
invest$yr10.ret <- as.numeric(invest$yr10.ret)
# convert returns into decimal form
invest$yr1.ret <- invest$yr1.ret/100
invest$yr3.ret <- invest$yr3.ret/100
invest$yr5.ret <- invest$yr5.ret/100
invest$yr7.ret <- invest$yr7.ret/100
invest$yr10.ret <- invest$yr10.ret/100

# NA removal
summary(invest)
invest <- na.omit(invest)
# invest$yr5.ret[is.na(invest$yr5.ret)] <- mean(invest$yr5.ret, na.rm=T)
# invest$yr7.ret[is.na(invest$yr7.ret)] <- mean(invest$yr7.ret, na.rm=T)
# invest$yr10.ret[is.na(invest$yr10.ret)] <- mean(invest$yr10.ret, na.rm=T)

### load the libraries
library(dplyr)
library(neuralnet)

### prepare the data
for(i in c(1,2,3,5,8,13, 21, 34, 55, 89, 100)){
set.seed(i)

# use 75% for training
# use 25% for testing

train <- slice_sample(invest, prop=0.75)
test <- setdiff(invest, train)

### create neural networks
# train the model to use 10 yr, 7 yr, 5 yr, and 3 yr returns to predict 1 yr returns
# using hyperbolic tangent activation function
tanmodel <- neuralnet(yr1.ret ~ yr3.ret+yr5.ret+yr7.ret+yr10.ret, data=train, hidden=5, act.fct="tanh", linear.output=F)
plot(tanmodel)

# using logistic model activation function
logmodel <- neuralnet(yr1.ret ~ yr3.ret+yr5.ret+yr7.ret+yr10.ret, data=train, hidden=5, act.fct="logistic")
plot(logmodel)

# use neural network to predict testing data and view results
tan.results <- predict(tanmodel, test)
tan.results <- data.frame(test$yr1.ret, tan.results)
plot(tan.results)

log.results <- predict(logmodel, test)
log.results <- data.frame(test$yr1.ret, log.results)
plot(log.results)

# create a linear model to compare
linemodel <- lm(yr1.ret ~ yr3.ret+yr5.ret+yr7.ret+yr10.ret, data=train)

line.results <- predict(linemodel, test)
line.results <- data.frame(test$yr1.ret, line.results)
plot(line.results)


# use correlations to quantify success in prediction models
print(paste("htan seed: ",i, " ",cor(tan.results$test.yr1.ret, tan.results$tan.results)))
print(paste("log seed: ",i, " ",cor(log.results$test.yr1.ret, log.results$log.results)))
print(paste("line seed: ",i, " ",cor(line.results$test.yr1.ret, line.results$line.results)))
}

set.seed(100)
### deep learning with two hidden layers
# using hyperbolic tangent activation function
tanmodel2 <- neuralnet(yr1.ret ~ yr3.ret+yr5.ret+yr7.ret+yr10.ret, data=train, hidden=c(4,2), act.fct="tanh")
plot(tanmodel2)

# using logistic model activation function
logmodel2 <- neuralnet(yr1.ret ~ yr3.ret+yr5.ret+yr7.ret+yr10.ret, data=train, hidden=c(4,2), act.fct="logistic")
plot(logmodel2)


tan.results2 <- predict(tanmodel2, test)
tan.results2 <- data.frame(test$yr1.ret, tan.results2)
plot(tan.results2)

log.results2 <- predict(logmodel2, test)
log.results2 <- data.frame(test$yr1.ret, log.results2)
plot(log.results2)

# use correlations to quantify success in prediction models
cor(tan.results2$test.yr1.ret, tan.results2$tan.results2)
cor(log.results2$test.yr1.ret, log.results2$log.results2)


### use for loops to identify optimal number of nodes
# single layer neural network
for(nodes in c(1,2,3,4,5,6,7,8,9,10)){
  tanmodel <- neuralnet(yr1.ret ~ yr3.ret+yr5.ret+yr7.ret+yr10.ret, data=train, hidden=nodes, act.fct="tanh", linear.output=F)
  tan.results <- predict(tanmodel, test)
  tan.results <- data.frame(test$yr1.ret, tan.results)
  corr<- cor(tan.results$test.yr1.ret, tan.results$tan.results)
  print(paste("Nodes: ", nodes, "   Correlation: ", corr))
}

for(nodes in c(1,2,3,4,5,6,7,8,9,10)){
  logmodel <- neuralnet(yr1.ret ~ yr3.ret+yr5.ret+yr7.ret+yr10.ret, data=train, hidden=nodes)
  log.results <- predict(logmodel, test)
  log.results <- data.frame(test$yr1.ret, log.results)
  corr<- cor(log.results$test.yr1.ret, log.results$log.results)
  print(paste("Nodes: ", nodes, "   Correlation: ", corr))
}

# double layer neural network
for(nodes1 in c(1,2,3,4,5)){
  for(nodes2 in c(1,2,3,4,5)){
    tanmodel <- neuralnet(yr1.ret ~ yr3.ret+yr5.ret+yr7.ret+yr10.ret, data=train, hidden=c(nodes1, nodes2), act.fct="tanh")
    tan.results <- predict(tanmodel, test)
    tan.results <- data.frame(test$yr1.ret, tan.results)
    corr<- cor(tan.results$test.yr1.ret, tan.results$tan.results)
    print(paste("Nodes: ", nodes1, " ", nodes2, "   Correlation: ", corr))
  }
}

for(nodes1 in c(1,2,3,4,5)){
  for(nodes2 in c(1,2,3,4,5)){
    logmodel <- neuralnet(yr1.ret ~ yr3.ret+yr5.ret+yr7.ret+yr10.ret, data=train, hidden=c(nodes1, nodes2), act.fct="logistic")
    log.results <- predict(logmodel, test)
    log.results <- data.frame(test$yr1.ret, log.results)
    corr<- cor(log.results$test.yr1.ret, log.results$log.results)
    print(paste("Nodes: ", nodes1, " ", nodes2, "   Correlation: ", corr))
  }
}


### use for loops to try different weights for each predictor
# preparing data
weights <- c(1, 1, 1, 1)
time <- c("yr3.wt", "yr5.wt", "yr7.wt", "yr10.wt")
weights <- data.frame(time, weights)
# choosing weights for calibration
x <- c(2.1, 1.9, 1.7, 1.5, 1.3, 0.9, 0.7, 0.5, 0.3, 0.1)
# creating blank vectors for results data frame
description <- vector()
values <- vector()

# running nested loops to check all combinations
for(wt3 in x){
  for(wt5 in x){
    for(wt7 in x){
      for(wt10 in x){
        #creating description 
        description <- append(description, paste("wt3: ",wt3, " wt5: ",wt5, " wt7: ",wt7, " wt10: ",wt10))
        #adjusting weights
        weights[1,2] <- wt3
        weights[2,2] <- wt5
        weights[3,2] <- wt7
        weights[4,2] <- wt10
        #calculating score
        invest$yr3.ws <- invest$yr3.ret * weights[1,2]
        invest$yr5.ws <- invest$yr5.ret * weights[2,2]
        invest$yr7.ws <- invest$yr7.ret * weights[3,2]
        invest$yr10.ws <- invest$yr10.ret * weights[4,2]
        invest$score <- invest$yr3.ws+invest$yr5.ws+invest$yr7.ws+invest$yr10.ws
        #calculating correlation
        value <- cor(invest$score, invest$yr1.ret)
        #creating results
        values <- append(values, value)
      }
    }
  }
}

# using correlation to determine best weights configuration
results <- data.frame(description, values)
summary(results$values)
which.max(results$values)
results[9100,]

results <- results[order(-results$values),]
head(results, 25)


cor(invest$yr1.ret, invest$yr5.ret)
cor(invest$yr1.ret, invest$yr3.ret)
cor(invest$yr1.ret, invest$yr10.ret)
