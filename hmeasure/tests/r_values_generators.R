generate_test_case <- function(
  case_export_path = ".", 
  case_export_name = "hmeasure_r_case_1.json",
  seed = 1,
  sev_ratio = 1) {
  
  case_full_file_path <- file.path(case_export_path, case_export_name)
  
  data_collection <- list()

  library(MASS)
  library(hmeasure)
  library(jsonlite)
  library(assertthat)

  set.seed(seed)
  data_collection[['sev_ratio']] <- sev_ratio

  pima.train <- MASS::Pima.tr
  pima.test <- MASS::Pima.te
  true.labels <- pima.test$type
  true.labels.01 <- ifelse(true.labels=="Yes", 1, 0)
  lda.pima <- lda(formula=type~., data=pima.train)
  out.lda = predict(lda.pima, newdata=pima.test)
  scores.lda <- out.lda$posterior[,2]  

  ### collect model_data ####
  s <- scores.lda
  y <- true.labels.01
  data_collection[['y_pred']] <- unname(s)
  data_collection[['y_true']] <- y==1
  
  ### collect n ####
  n <- length(s)
  n1 <- sum(y)
  n0 <- length(y) - n1
  pi0 <- n0/n
  pi1 <- n1/n
  
  data_collection[['n0']] <- n0
  data_collection[['n1']] <- n1
  
  ### collect shapes ####
  severity.ratio <- sev_ratio
  if (is.na(severity.ratio)) {
    severity.ratio <- pi1/pi0
  }
  if (severity.ratio > 0) {
    shape1 <- 2
    shape2 <- 1 + (shape1 - 1) * 1/severity.ratio
  }
  if (severity.ratio < 0) {
    shape1 <- pi1 + 1
    shape2 <- pi0 + 1
  }
  
  data_collection[['a']] <- shape1
  data_collection[['b']] <- shape2
  
  ## collect invF ####
  Get.Score.Distributions <- function(y, s, n1, n0) {
    s1 <- unname(tapply(y, s, sum))/n1
    s1 <- c(0, s1, 1 - sum(s1))
    s0 <- unname(tapply(1 - y, s, sum))/n0
    s0 <- c(0, s0, 1 - sum(s0))
    S <- length(s1)
    F1 <- cumsum(s1)
    F0 <- cumsum(s0)
    return(list(F1 = F1, F0 = F0, s1 = s1, s0 = s0, S = S))
  }
  out.scores <- Get.Score.Distributions(y = y, s = s, n1 = n1, 
                                        n0 = n0)
  F1 <- out.scores$F1
  F0 <- out.scores$F0
  invF1 = 1 - F1
  invF0 = 1 - F0

  data_collection[['invF0']] <- invF0[-length(invF0)]
  data_collection[['invF1']] <- invF1[-length(invF1)]
  
  ## collect G ####
  chull.points <- chull(invF0, pmax(invF1, invF0))
  G0 <- invF0[chull.points]
  G1 <- invF1[chull.points]
  hc <- length(chull.points)

  data_collection[['G0']] <- G0
  data_collection[['G1']] <- G1
  
  ## collect cost ####
  cost <- c(1:(hc + 1))
  cost[1] <- 0
  cost[hc+1] <- 1
  for (i in 2:hc) {
    cost[i] <- pi1 * (G1[i] - G1[i - 1])/(pi0 * (G0[i] -
                                                   G0[i - 1]) + pi1 * (G1[i] - G1[i - 1]))
  }
  
  data_collection[['cost']] <- cost
  
  ## collect b_vecs ####
  b0 <- c(1:hc + 1)
  b1 <- c(1:hc + 1)
  
  b00 <- beta(shape1, shape2)
  b10 <- beta(1 + shape1, shape2)
  b01 <- beta(shape1, 1 + shape2)
  b0[1] <- pbeta(cost[1], shape1 = (1 + shape1), shape2 = shape2) *
    b10/b00
  b1[1] <- pbeta(cost[1], shape1 = shape1, shape2 = (1 + shape2)) *
    b01/b00
  b0[hc + 1] <- pbeta(cost[hc + 1], shape1 = (1 + shape1),
                      shape2 = shape2) * b10/b00
  b1[hc + 1] <- pbeta(cost[hc + 1], shape1 = shape1, shape2 = (1 +
                                                                 shape2)) * b01/b00
  for (i in 2:hc) {
    b0[i] <- pbeta(cost[i], shape1 = (1 + shape1), shape2 = shape2) *
      b10/b00
    b1[i] <- pbeta(cost[i], shape1 = shape1, shape2 = (1 +
                                                         shape2)) * b01/b00
  }

  data_collection[['b0']] <- b0
  data_collection[['b1']] <- b1
  
  ## collect LH ####
  LHshape1 <- 0
  for (i in 1:hc) {
    LHshape1 <- LHshape1 + pi0 * (1 - G0[i]) * (b0[(i + 
                                                      1)] - b0[i]) + pi1 * G1[i] * (b1[(i + 1)] - b1[i])
  }
  
  data_collection[['LH']] <- LHshape1
  
  ## collect B_coefs ####
  B0 <- pbeta(pi1, shape1 = (1 + shape1), shape2 = shape2) * 
    b10/b00
  B1 <- pbeta(1, shape1 = shape1, shape2 = (1 + shape2)) * 
    b01/b00 - pbeta(pi1, shape1 = shape1, shape2 = (1 + 
                                                      shape2)) * b01/b00
  
  data_collection[['B0']] <- B0
  data_collection[['B1']] <- B1
  
  ## collect H ####
  H <- 1 - LHshape1/(pi0 * B0 + pi1 * B1)
  data_collection[['H']] <- H
  
  ## validate H ####
  original_H_output <- hmeasure::HMeasure(true.class = true.labels.01, scores = scores.lda, severity.ratio = sev_ratio)
  original_H <- original_H_output[['metrics']][,'H']
  final_assertion <- assertthat::are_equal(H, original_H)
  if (!final_assertion) {
    stop("Internally generated H-measure differs from hmeasure::HMeasure")
  }
  
  ## save output ####
  jsonlite::write_json(x = data_collection, path = case_full_file_path, digits = NA, auto_unbox=TRUE, pretty=TRUE)
  return(normalizePath(case_full_file_path))

}


### Script ####
generate_test_case(case_export_path = "./resources/", 
                   case_export_name = "hmeasure_r_case_0.json", 
                   seed = 10, 
                   sev_ratio = 1)

generate_test_case(case_export_path = "./resources/", 
                   case_export_name = "hmeasure_r_case_1.json", 
                   seed = 1, 
                   sev_ratio = NA)


generate_test_case(case_export_path = "./resources/", 
                   case_export_name = "hmeasure_r_case_2.json", 
                   seed = 20, 
                   sev_ratio = 2)

generate_test_case(case_export_path = "./resources/", 
                   case_export_name = "hmeasure_r_case_3.json", 
                   seed = 40, 
                   sev_ratio = 4)

generate_test_case(case_export_path = "./resources/", 
                   case_export_name = "hmeasure_r_case_4.json", 
                   seed = 1, 
                   sev_ratio = 0.1)

generate_test_case(case_export_path = "./resources/", 
                   case_export_name = "hmeasure_r_case_5.json", 
                   seed = 3, 
                   sev_ratio = 0.3)

generate_test_case(case_export_path = "./resources/", 
                   case_export_name = "hmeasure_r_case_6.json", 
                   seed = 13, 
                   sev_ratio = -1.3)






