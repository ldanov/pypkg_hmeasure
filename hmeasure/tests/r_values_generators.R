library(MASS)
library(hmeasure)
data(Pima.te)

set.seed(1)
severity.ratio <- 1
n <- dim(Pima.te)[1]
pima.train <- Pima.te[seq(1,n,3),]
pima.test <- Pima.te[-seq(1,n,3),]
true.labels <- pima.test[,8]
lda.pima <- lda(formula=type~., data=pima.train)
out.lda = predict(lda.pima, newdata=pima.test)
true.labels.01 <- relabel(true.labels)
lda.labels.01 <- relabel(out.lda$class)
scores.lda <- out.lda$posterior[,2]


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

n <-length(s)
y = true.labels.01
s = scores.lda
n1 = sum(y)
n0 = length(y) - n1
pi0 <- n0/n
pi1 <- n1/n

paste0(ifelse(y==1, "True", "False"), collapse = ", ")
paste0(s, collapse = ", ")

out.scores <- Get.Score.Distributions(y = y, s = s, n1 = n1, 
                                      n0 = n0)
AUC <- 1 - sum(out.scores$s0 * (out.scores$F1 - 0.5 * out.scores$s1))
F1 <- out.scores$F1
F0 <- out.scores$F0

paste0(F0, collapse = ", ")
paste0(F1, collapse = ", ")
invF1 = 1 - F1
invF0 = 1 - F0
paste0(invF0[-length(invF0)], collapse = ", ")
paste0(invF1[-length(invF1)], collapse = ", ")
chull.points <- chull(invF0, pmax(invF1, invF0))
paste0(pmax(invF1, invF0), collapse = ", ")
G0 <- invF0[chull.points]
G1 <- invF1[chull.points]
hc <- length(chull.points)


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


cost <- c(1:(hc + 1))
cost[1] <- 0
cost[hc+1] <- 1
for (i in 2:hc) {
  cost[i] <- pi1 * (G1[i] - G1[i - 1])/(pi0 * (G0[i] -
                                                  G0[i - 1]) + pi1 * (G1[i] - G1[i - 1]))
}

#################
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
paste0(format(b0, scientific = FALSE), collapse = ", ")
paste0(format(b1, scientific = FALSE), collapse = ", ")
##################

LHshape1 <- 0
for (i in 1:hc) {
  LHshape1 <- LHshape1 + pi0 * (1 - G0[i]) * (b0[(i + 
                                                    1)] - b0[i]) + pi1 * G1[i] * (b1[(i + 1)] - b1[i])
}

####################
B0 <- pbeta(pi1, shape1 = (1 + shape1), shape2 = shape2) * 
  b10/b00
B1 <- pbeta(1, shape1 = shape1, shape2 = (1 + shape2)) * 
  b01/b00 - pbeta(pi1, shape1 = shape1, shape2 = (1 + 
                                                    shape2)) * b01/b00

#######################
H <- 1 - LHshape1/(pi0 * B0 + pi1 * B1)
H
