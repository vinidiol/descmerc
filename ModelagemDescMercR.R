# Set seed for reproducible results
set.seed(100)

# Packages
library(readxl)
library(tm)         # Text mining: Corpus and Document Term Matrix
library(class)      # KNN model
library(SnowballC)  # Stemming words
library(h2o)        # Pacote H2O
h2o.init()

# Ler Arquivo csv com duas colunas: Text e Category
df <- read.csv("~/Documents/AmostraDescMerc.csv")

# Create corpus
docs <- Corpus(VectorSource(df$Text))

# Clean corpus
docs <- tm_map(docs, content_transformer(tolower))
docs <- tm_map(docs, removeNumbers)
docs <- tm_map(docs, removeWords, stopwords("pt"))
docs <- tm_map(docs, removePunctuation)
docs <- tm_map(docs, stripWhitespace)

# Create dtm
dtm <- DocumentTermMatrix(docs)

# Transform dtm to matrix to data frame - df is easier to work with
mat.df <- as.data.frame(data.matrix(dtm), stringsAsfactors = FALSE)


# Column bind category (known classification)
mat.df <- cbind(mat.df, df$Category)

# Change name of new column to "category"
colnames(mat.df)[ncol(mat.df)] <- "category"

mat.df.h <- as.h2o(mat.df)
data.split <- h2o.splitFrame(data = mat.df.h, ratios = c(0.7, 0.2), seed = 1234)
data.train <- data.split[[1]]
data.valid <- data.split[[2]]
data.test <- data.split[[3]]
myY <- "category"
myX <- setdiff(names(data.train), c(myY, "ID"))


# Modelo GBM - Gradient Boosting Machine
gbm.model <- h2o.gbm(myX, myY,
                     training_frame = data.train,
                     validation_frame = data.valid, ntrees = 1000, max_depth = 3,
                     model_id = "gbm_xprod_5mil")
h2o.confusionMatrix(gbm.model@model$validation_metrics)
conf.mat <- h2o.confusionMatrix(gbm.model@model$validation_metrics)
write.table(conf.mat, file="confmat_gbm_5mil.csv", sep=";")
r2.gbm.model.5mil <- gbm.model@model$validation_metrics@metrics$r2


#DeepLearning (MLP - Multi Layer Perceptron)
dl.model <- h2o.deeplearning(myX, myY, 
                             training_frame = data.train ,
                             hidden = c(100,200,100), 
                             epochs = 20,
                             validation_frame = data.valid,
                             model_id = "dl_xprod_5mil")
h2o.confusionMatrix(dl.model@model$validation_metrics)
conf.mat.dl5mil <- h2o.confusionMatrix(dl.model@model$validation_metrics)
write.table(conf.mat.dl5mil, file="confmat_dl_5mil.csv", sep=";")
r2.dl.model.5mil <- dl.model@model$validation_metrics@metrics$r2