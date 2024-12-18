# Carregar pacotes necessários
library(dplyr)
library(caret)
library(e1071) # Para Naïve Bayes
library(rpart) # Para Árvore de Decisão
library(randomForest) # Para Random Forest
library(kernlab) # Para SVM
library(h2o) # Para Redes Neurais

# Ler os arquivos CSV
student_mat <- read.csv("student-mat.csv", sep = ";")
student_por <- read.csv("student-por.csv", sep = ";")

# Juntar as bases de dados
students <- bind_rows(student_mat, student_por)

# Converter variáveis categóricas em numéricas
students$school <- as.numeric(as.factor(students$school))
students$sex <- as.numeric(as.factor(students$sex))
students$address <- as.numeric(as.factor(students$address))
students$famsize <- as.numeric(as.factor(students$famsize))
students$Pstatus <- as.numeric(as.factor(students$Pstatus))
students$Mjob <- as.numeric(as.factor(students$Mjob))
students$Fjob <- as.numeric(as.factor(students$Fjob))
students$reason <- as.numeric(as.factor(students$reason))
students$guardian <- as.numeric(as.factor(students$guardian))
students$schoolsup <- as.numeric(as.factor(students$schoolsup))
students$famsup <- as.numeric(as.factor(students$famsup))
students$paid <- as.numeric(as.factor(students$paid))
students$activities <- as.numeric(as.factor(students$activities))
students$nursery <- as.numeric(as.factor(students$nursery))
students$higher <- as.numeric(as.factor(students$higher))
students$internet <- as.numeric(as.factor(students$internet))
students$romantic <- as.numeric(as.factor(students$romantic))

# Categorizar as notas em classes
categorize_grades <- function(g) {
  if (g >= 0 && g <= 3) return("D")
  if (g >= 4 && g <= 10) return("C")
  if (g >= 11 && g <= 15) return("B")
  if (g >= 16 && g <= 20) return("A")
}

students$G3_cat <- sapply(students$G3, categorize_grades)
students$target <- as.factor(students$G3_cat)

# Remover colunas desnecessárias
students_cleaned <- students %>%
  select(-G1, -G2, -G3, -G3_cat)

# Dividir os dados em conjuntos de treinamento e teste
set.seed(123)
library(caTools)

divisao <- sample.split(students_cleaned$target, SplitRatio = 0.7)
train_data <- subset(students_cleaned, divisao == TRUE)
test_data <- subset(students_cleaned, divisao == FALSE)

# Treinamento dos Modelos

# Naïve Bayes
nb_model <- naiveBayes(target ~ ., data = train_data)
nb_pred <- predict(nb_model, test_data)

# Árvore de Decisão
tree_model <- rpart(target ~ ., data = train_data)
tree_pred <- predict(tree_model, test_data, type = "class")

# Random Forest
rf_model <- randomForest(target ~ ., data = train_data)
rf_pred <- predict(rf_model, test_data)

# SVM
svm_model <- svm(target ~ ., data = train_data)
svm_pred <- predict(svm_model, test_data)

# Redes Neurais (H2O)
h2o.init()
train_h2o <- as.h2o(train_data)
test_h2o <- as.h2o(test_data)

nn_model <- h2o.deeplearning(
  x = names(train_h2o)[-which(names(train_h2o) == "target")],
  y = "target",
  training_frame = train_h2o,
  hidden = c(5),
  epochs = 100,
  activation = "Rectifier"
)

nn_pred <- h2o.predict(nn_model, test_h2o)
nn_pred_class <- as.vector(nn_pred$predict)

# Função para calcular métricas usando confusionMatrix
get_metrics <- function(true_labels, predictions) {
  matriz_confusao <- table(true_labels, predictions)
  
  print(matriz_confusao) # Exibir matriz de confusão
  
  cm <- confusionMatrix(matriz_confusao)
  
  accuracy <- cm$overall['Accuracy']
  
  # Intervalo de Confiança (95%)
  ci_lower <- cm$overall['95% CI'][1]
  ci_upper <- cm$overall['95% CI'][2]
  
  # Calcular a média do intervalo de confiança
  ci_mean <- (ci_lower + ci_upper) / 2
  
  # Taxa de Não Informação (NIR)
  nir <- (sum(matriz_confusao) - sum(apply(matriz_confusao, 1, max))) / sum(matriz_confusao)
  
  p_value <- cm$overall['P-value']
  kappa <- cm$overall['Kappa']
  
  return(c(accuracy = accuracy, ci_mean = ci_mean, nir = nir, p_value = p_value, kappa = kappa))
}

# Calcular métricas para cada modelo
metrics_nb <- get_metrics(test_data$target, nb_pred)
metrics_tree <- get_metrics(test_data$target, tree_pred)
metrics_rf <- get_metrics(test_data$target, rf_pred)
metrics_svm <- get_metrics(test_data$target, svm_pred)
metrics_nn <- get_metrics(test_data$target, nn_pred_class)

# Criar tabela com resultados
results_table <- data.frame(
  Model = c("Naïve Bayes", "Árvore de Decisão", "Random Forest", "SVM", "Redes Neurais"),
  Accuracy = c(metrics_nb[1], metrics_tree[1], metrics_rf[1], metrics_svm[1], metrics_nn[1]),
  CI_Mean = c(metrics_nb[2], metrics_tree[2], metrics_rf[2], metrics_svm[2], metrics_nn[2]),
  NIR = c(metrics_nb[3], metrics_tree[3], metrics_rf[3], metrics_svm[3], metrics_nn[3]),
  P_Value = c(metrics_nb[4], metrics_tree[4], metrics_rf[4], metrics_svm[4], metrics_nn[4]),
  Kappa = c(metrics_nb[5], metrics_tree[5], metrics_rf[5], metrics_svm[5], metrics_nn[5])
)

print(results_table)


#Calculo das importantes variaveis 
importance_values <- svm_model$coefs
#ERRO AQUI no levantamento do data frame
importance_df <- data.frame(Variable = names(importance_values), Importance = importance_values)
importance_df <- importance_df[order(-abs(importance_df$Importance)), ]
print(importance_df)
# Finalize H2O
h2o.shutdown(prompt = FALSE)