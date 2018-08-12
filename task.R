#####################################################
# Ringaile Placakyte, 12.08.2018
#####################################################
# required libraries
library(ggplot2)
library(ggpubr)
library(dplyr)
library(data.table)
library(rpart)
library(Amelia)  # to check missing values in data (missing vs observed)
library(e1071)   # SVM
library(gmodels)
library(caret)

require(randomForest)
require(MASS)
par(mfrow = c(1, 1))
# options for margins
par(mar = c(7, 6, 3, 3) + 0.1)

# 1. read-in data
cat("====================================================================\n")
cat("Reading data..........\n")
d <- read.csv("~/a/Auto1-DS-TestData.csv", dec = ",", 
                na.strings=c("NA", "<NA>"), stringsAsFactors = FALSE)


# 2. check data
cat("====================================================================\n")
cat("Checking data..........\n")
dim(d)
str(d)
#sum(is.na(d$normalized.losses))

# 3. cleaning data
cat("====================================================================\n")
cat("Cleaning data..........\n")
# ------------------------------------------
#table(d$symboling,exclude=NULL)
# contains negative ranking numbers, add largest negative to get all positive
d$symboling <- d[, 1] - min(d$symboling[d$symboling < 0])

#table(d$normalized.losses,exclude=NULL)
d$normalized.losses<-as.numeric(d$normalized.losses)

# list of variables to convert to numeric
nl <- c("wheel.base",
        "length",
        "width",
        "height",
        "city.mpg",
        "highway.mpg",
        "stroke",
        "bore",
        "compression.ratio",
        "horsepower",
        "peak.rpm",
        "price")
d[nl] <- lapply(d[nl], as.numeric) 


# bit special case with numbers written out as characters:
# can use factor but can also be converted directly to numbers:
d$num.of.cylinders <- gsub("two", "2", d$num.of.cylinders)
d$num.of.cylinders <- gsub("three", "3", d$num.of.cylinders)
d$num.of.cylinders <- gsub("four", "4", d$num.of.cylinders)
d$num.of.cylinders <- gsub("five", "5", d$num.of.cylinders)
d$num.of.cylinders <- gsub("six", "6", d$num.of.cylinders)
d$num.of.cylinders <- gsub("eight", "8", d$num.of.cylinders)
d$num.of.cylinders <- gsub("twelve", "12", d$num.of.cylinders)
d$num.of.cylinders<-as.numeric(d$num.of.cylinders)


# all non-numerical values make factors
d$make=factor(d$make)
d$fuel.type=factor(d$fuel.type)
d$aspiration=factor(d$aspiration)
d$num.of.doors=factor(d$num.of.doors)
d$body.style=factor(d$body.style)
d$drive.wheels=factor(d$drive.wheels)
d$engine.location=factor(d$engine.location)
d$engine.type=factor(d$engine.type)
d$fuel.system=factor(d$fuel.system)

cat("====================================================================\n")
cat("Drawing map with missing data..........\n")
# check the map for missing variables and remove if they are (basically) empty
missmap(d, main = "Missing values vs observed")
# can be seen that:
# d$stroke and bore have 4 NAs
# d$horsepower 2 NA
# d$peak  2 NA
# d$price  4 NA
# will remove them (will be in total 12 removed data points but can also be replace 
# by mean or average like mean(d$horsepower,na.rm=T) )


############################### do exta clean-ups:
d <- subset(d,select = -c(engine.location)) # as only 3 rows have different values
# num.of.doors has one ?, lets check that row:
d[d$num.of.doors %like% "\\?", ]
# as it is only one row we could remove it, however for larger set we may want to 
# insert most likely data here... let see if what crossTable tells us:
CrossTable(d$num.of.doors,d$body.style, prop.chisq = FALSE)
CrossTable(d$num.of.doors,d$aspiration, prop.chisq = FALSE)
# ok, pretty clear it should be 4 doors :) lets replace ? with that:
d$num.of.doors[which(d$num.of.doors=="?")] <- "four"
d$num.of.doors=factor(d$num.of.doors)
##############################################

# first, create separate dataframewith missing 'normalized.losses' 
# (will want to predict values for them)
d_pred_losses <- subset(d, is.na(d$normalized.losses))
#str(d_pred_losses)

# now remove with NAs:
d<-na.omit(d)
d_pred_losses$normalized.losses <-as.numeric(1)  # as NAs cannto be handled by algorithms
d_pred_losses<-na.omit(d_pred_losses)

###########################################################
# 4. correlations
cat("====================================================================\n")
cat("Checking correlations..........\n")

# quick check with PCA (using only numerical val, categorial could be vectorised):
d_pca <- subset(d,select = -c(make,fuel.type,aspiration,num.of.doors,
                              body.style,drive.wheels,
                              engine.type,fuel.system))
pr_comp <- prcomp(d_pca, center = TRUE, scale. = TRUE)
head(pr_comp$x)  # this prints all components
#print(pr_comp)

pairs(d_pca,pch = 1)
# let's see how much prices are correlated wtih losses
cor.test(d$price,d$normalized.losses)
# corr is not high, i.e. ~20%, thus will use price in the main training sample 
# to obtain missing losses, however in general one should be carefull for such 
# circular correlations in the real data

# calculate correlation matrix (valid only for numerical values))
correlationMatrix <- cor(d_pca[,1:ncol(d_pca)])
# summarize the correlation matrix
print(correlationMatrix)
# find attributes that are highly corrected (ideally >0.75)
highlyCorrelated <- findCorrelation(correlationMatrix, cutoff=0.75)
# print indexes of highly correlated attributes
print(highlyCorrelated)

##############################################
# 5. modeling
cat("====================================================================\n")
cat("Starting prediction of modeling data ..........\n")

## split sample in training and testing 
cat("Splitting data to train and test ..........\n")
set.seed(101011)
# selecting 70% of data as sample from total 'n' rows of the data  
sample_split <- sample(2, nrow(d), replace = TRUE, prob = c(0.7,0.3))
train <- d[sample_split==1,]
test  <- d[sample_split==2, ]
nrow(train) # check rows split


### for fun, let's see predictions for aspiration with radom forest:
#rf_model=randomForest(aspiration~., data = train, importance = TRUE)
#rf_model
#plot(rf_model)
#
#importance(rf_model)
#imp_rf <- as.data.frame(varImp(rf_model))
#imp_rf <- data.frame(overall = imp_rf$Overall,names   = rownames(imp_rf))
#imp_rf[order(imp_rf$overall,decreasing = T),]
#
#pred_rf <- predict(rf_model, data = train)
#conf_pred_rf<-table(Predicted=pred_rf,Reference=train$price)
#confusionMatrix(conf_pred_rf)

#####################################################
# will use SVM to predict missing loss values
cat("SVM linear model training ..........\n")
svm_linear <- svm(normalized.losses~., data=train, cost=50, gamma=0.5)
summary(svm_linear)
# now predict:
pred <- predict(svm_linear, newdata = test)
plot(pred~test$normalized.losses,
     data=test,
     pch = 1,
     xlim=c(55,265), ylim=c(55,265),
     xlab="Actual losses",
     ylab="Predicted losess")

ggplot(data = test, aes(x = test$normalized.losses, y = pred)) + geom_point()  +
  stat_smooth(method = "lm", col = "dodgerblue3") +
  theme(panel.background = element_rect(fill = "white"),
        axis.line.x=element_line(),
        axis.line.y=element_line()) +
  ggtitle("Linear SVM Model Fitted to Data")

sum(pred)
sum(test$normalized.losses)

# tuning hyper-parameters
#tune_svnparam = tune.svm(normalized.losses~.,data=train,cost=50:100,gamma=seq(0,3,1))
#summary(tune_svnparam) # ok, got gamma=0.5, cost = 50, put them back

# not prefect but (most likely) better than simple mean... 
# so now we can predict losses for missing values in subsample we created before:
cat("Getting predictions with SVM model  ..........\n")
predi <- predict(svm_linear, newdata = d_pred_losses)
sum(predi)
str(predi)

# now let's append to original test sample
cat("Appending predictions and creating one dataframe  ..........\n")
d_pred_losses$normalized.losses <-as.integer(predi)
#table(d_pred_losses$normalized.losses)

# add two sets to one which we can use now to predict e.g. prices 
d_tot <- rbind(d, d_pred_losses)
str(d_tot)

# fuel.system is fully collinear (correlated) with others and causing 
# singularities in liner regression, so we remove it (no data loss)
d_tot <- subset(d_tot,select = -c(fuel.system)) 
###############################################################
# now let's use GLM to predict prices (from correlation plots we saw that
# this approach should work quite well, however other models can be used

# first -  split sample in training and testing 
cat("Splitting data to train and test (now using whole data sample)..........\n")
set.seed(101010)
# selecting 70% of data as sample from total 'n' rows of the data  
sample_split <- sample(2, nrow(d_tot), replace = TRUE, prob = c(0.7,0.3))
dtrain <- d_tot[sample_split==1,]
dtest  <- d_tot[sample_split==2, ]
nrow(dtrain) # check rows split

### linear regression model:
cat("Starting the GLM model  ..........\n")
dm <- glm(price~., data=dtrain, family=gaussian())
summary(dm)

#plot(dm) # there are few outliers and data is not fully normal, 
# however we ignore it for this test
#AICc(dm)

#library(Matrix)
#cat(rankMatrix(dtrain), "\n")
#cat(rankMatrix(dtest), "\n") 

cat("Getting predictions with the GLM model  ..........\n")
dpred <- predict(dm, newdata = dtest)
plot(dpred~dtest$price,
     data=dtest,
     pch = 1,
     xlim=c(5000,25000), ylim=c(5000,25000),
     xlab="Actual prices",
     ylab="Predicted prices")

# gray area is confidence interval of 0.95 (default for the stat_smooth() function)
ggplot(data = dtest, aes(x = dtest$price, y = dpred)) + geom_point()  +
  stat_smooth(method = "lm", col = "dodgerblue") +
  theme(panel.background = element_rect(fill = "white"),
        axis.line.x=element_line(),
        axis.line.y=element_line()) +
  ggtitle("GLM (Linear Model) Fitted to Data")

sum(dtest$price)
sum(dpred)

head(dtest$price,nrow(dtest))
head(dpred,nrow(dtest))

# got LM (gaussian) AIC: 2341.5

# let's try random forest for comparison:
cat("As alternative, using Random Forest to predict prices..........\n")
set.seed(1010)
rf_out <- randomForest(price ~ . ,data = dtrain, importance = T)
print(rf_out)
plot(rf_out)
#importance(rf_out)
# better to have sorted view of importance:
imp_rf <- as.data.frame(varImp(rf_out))
imp_rf <- data.frame(overall = imp_rf$Overall,names   = rownames(imp_rf))
imp_rf[order(imp_rf$overall,decreasing = T),]

fit_rf <-predict(rf_out, newdata = dtest)
plot(fit_rf~dtest$price,
     data=dtest,
     pch = 1,
     xlim=c(5000,25000), ylim=c(5000,25000),
     xlab="Actual price",
     ylab="Predicted price")

ggplot(data = dtest, aes(x = dtest$price, y = fit_rf)) + geom_point()  +
  stat_smooth(method = "lm", col = "red") +
  theme(panel.background = element_rect(fill = "white"),
        axis.line.x=element_line(),
        axis.line.y=element_line()) +
  ggtitle("Linear Random Forest Model Fitted to Data")

sum(fit_rf)
sum(dtest$price)

cat("GLM preforms slightly better than random forest in this case (less underestimated prices)")
cat("The End\n")
cat("====================================================================\n")


