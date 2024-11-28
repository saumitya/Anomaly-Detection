
library(tidyverse)

load("C:/Users/saumi/OneDrive/Desktop/control/TEP_FaultFree_training.RData")
load("C:/Users/saumi/OneDrive/Desktop/control/TEP_FaultFree_testing.RData")
load("C:/Users/saumi/OneDrive/Desktop/control/TEP_Faulty_training.RData")
load("C:/Users/saumi/OneDrive/Desktop/control/TEP_Faulty_testing.RData")


write.csv(fault_free_training, "C:/Users/saumi/OneDrive/Desktop/control/fault_free_training.csv", row.names = FALSE)
write.csv(fault_free_testing, "C:/Users/saumi/OneDrive/Desktop/control/fault_free_testing.csv", row.names = FALSE)
write.csv(faulty_training, "C:/Users/saumi/OneDrive/Desktop/control/faulty_training.csv", row.names = FALSE)
write.csv(faulty_testing, "C:/Users/saumi/OneDrive/Desktop/control/faulty_testing.csv", row.names = FALSE)

