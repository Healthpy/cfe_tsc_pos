library(dplyr)
library(ggplot2)
library(tidyverse)
library(xtable)

setwd("C:/Users/20200059/OneDrive - TU Eindhoven/Documents/Emmanuel/Chukwu, Emmanuel's files - TSInterpret/results/")
dataA <- read.csv("AB_CF/ECG200_cf_results.csv", sep=",")
dataC <- read.csv("COMTE/ECG200_cf_results.csv", sep=",")
dataN <- read.csv("NG_CF/ECG200_cf_results.csv", sep=",")
dataT <- read.csv("TSEVO/ECG200_cf_results.csv", sep=",")

dataA <- read.csv("AB_CF/Epilepsy_cf_results.csv", sep=",")
dataC <- read.csv("COMTE/Epilepsy_cf_results.csv", sep=",")
#dataN <- read.csv("NG_CF/Epilepsy_cf_results.csv", sep=",")
dataT <- read.csv("TSEVO/Epilepsy_cf_results.csv", sep=",")

?read_csv

data200 <- bind_rows(dataA, dataC, dataN, dataT)

data200 %>% group_by(Method) %>%
  summarize(n = n()) %>%
  mutate(prop_found = n / 100)
# not all methods find CFEs for all instances within the max number of iterations


data200 %>% group_by(Method, Original_Class, Counterfactual_Class) %>%
  summarize(n = n()) 
# sometimes original class and counterfactual class are the same, 
# this is because the original instance was predicted wrong by the pred method.

names(data200)
data200$Orig_Conf

data200filt <- data200 %>%
  filter(Original_Class != Counterfactual_Class) %>%
  mutate(yhatcfyhatorig = CF_Conf_Gauss_eps0.00 - Orig_Conf_Gauss_eps0.00,
         ytildeyhatorig = Orig_Conf_Gauss_eps0.20 - Orig_Conf_Gauss_eps0.00,
         ytildeyhatcf = CF_Conf_Gauss_eps0.20 - CF_Conf_Gauss_eps0.00)
         
    #Orig_Gauss = Orig_Conf_Gauss - Orig_Conf,
    #     #Orig_Adv = Orig_Conf_Adv - Orig_Conf,
    #     CF_Conf = CF_Conf - Orig_Conf, 
    #     CF_Orig_Gauss = Conf_Gauss - Orig_Conf_Gauss,
    #     #CF_Adv = Conf_Adv - Orig_Conf_Adv, 
    #     CF_Gauss = Conf_Gauss - CF_Conf
    #     )

ECG200_ResultConf <- data200filt %>% group_by(Method, Original_Class) %>%
  summarise(n=n(),
            across(c("yhatcfyhatorig", "ytildeyhatorig", "ytildeyhatcf"), 
                   list(mean=mean, sd=sd)))
            
xtable(ECG200_ResultConf)

# New facet label names for supp variable
metric.labs <- c("CFE minus Original", "CFE minus CFE with noise","Original minus Original with noise")
names(metric.labs) <- c("yhatcfyhatorig", "ytildeyhatcf", "ytildeyhatorig")

data200filt %>%
  pivot_longer(
    cols = c("yhatcfyhatorig", "ytildeyhatcf", "ytildeyhatorig"),
    names_to = "metric",
    values_to = "value") %>%
  ggplot(aes(x = Method, y = value)) + 
  geom_boxplot(aes(fill = as.factor(Original_Class)), outlier.shape = NA) + 
  facet_grid( ~ metric, labeller = labeller(metric = metric.labs)) + 
  #facet_grid( ~ metric, labeller = label_both) + 
  ggtitle(label = "") + 
  xlab("") + 
  ylab("Distance in average confidence") + 
  scale_fill_manual(values = c("#fdb462", "#bebada"), 
                     labels = c("Class 0 to 1", "Class 1 to 0"),
                     name = "") +
  theme_bw(base_size=7) + 
  theme(legend.position="top",
        legend.justification="right",
        #legend.direction = "vertical",
        #plot.title = element_text(hjust = 0, vjust=-3), 
        legend.box.margin = margin(0,0,-0.2,0, "line"),
        #axis.title.y = element_text(),
        panel.grid.major.x = element_blank(),
        panel.grid.minor.x = element_blank(),
        #panel.grid.major.y = element_blank(),
        panel.grid.minor.y = element_blank(),
        legend.text  = element_text(size = 7),
        legend.key.width = unit(0.2,"cm"),
        legend.key.size = unit(0.2,"cm"))
        #plot.margin = unit(x = c(-2, 1, -2, -2), units = "mm"))

name <- paste("C:/Users/20200059/OneDrive - TU Eindhoven/Documents/Emmanuel/Rianne's analysis/ECG200Dif.pdf", sep = "", collapse = NULL)
ggsave(name, width = 12, height = 8, units = "cm")

?geom_boxplot

# proximity

data200 %>%
  filter(Original_Class != Counterfactual_Class) %>%
  select(c(Method, Original_Class, Counterfactual_Class,L1,L2,DTW,Tp_Sparsity,Seg_Sparsity)) %>%
  pivot_longer(
    cols = c("L1", "L2", "DTW", "Tp_Sparsity", "Seg_Sparsity"),
    names_to = "metric",
    values_to = "value") %>%
  ggplot(aes(x = Method, y = value)) + 
  geom_boxplot(aes(fill = as.factor(Original_Class)), outlier.shape = NA) + 
  facet_grid(metric ~ ., labeller = label_both, scales='free_y') 
# results for l1 and l2 are similar

dataprox <- data200 %>%
  filter(Original_Class != Counterfactual_Class) %>%
  select(c(Method, Original_Class, Counterfactual_Class,L1,L2,DTW,Tp_Sparsity,Seg_Sparsity)) %>%
  group_by(Method, Original_Class) %>%
  summarise(across(c("L1", "L2", "DTW", "Tp_Sparsity", "Seg_Sparsity"), 
         list(mean=mean, sd=sd))) 

dataprox %>%
  ggplot(aes(x=L1_mean,y=DTW_mean,color=Method)) + 
  geom_point(aes(shape = as.factor(Original_Class))) + 
  ggtitle(label = "") + 
  xlab("L1 norm") + 
  ylab("Dynamic Time Warping distance") + 
  scale_shape_manual(values = c(15,19), 
                    labels = c("Class 0 to 1", "Class 1 to 0"),
                    name = "") +
  theme_bw(base_size=9) + 
  theme(legend.position="top",
        legend.justification="right",
        #legend.direction = "vertical",
        #plot.title = element_text(hjust = 0, vjust=-3), 
        legend.box.margin = margin(0,0,-0.2,0, "line"),
        #axis.title.y = element_text(),
        panel.grid.major.x = element_blank(),
        panel.grid.minor.x = element_blank(),
        #panel.grid.major.y = element_blank(),
        panel.grid.minor.y = element_blank(),
        legend.text  = element_text(size = 7),
        legend.key.width = unit(0.2,"cm"),
        legend.key.size = unit(0.2,"cm"))

name <- paste("C:/Users/20200059/OneDrive - TU Eindhoven/Documents/Emmanuel/Rianne's analysis/ECG200Prox.pdf", sep = "", collapse = NULL)
ggsave(name, width = 10, height = 10, units = "cm")

dataprox %>%
  ggplot(aes(x=Tp_Sparsity_mean,y=Seg_Sparsity_mean,color=Method)) + 
  geom_point(aes(shape = as.factor(Original_Class))) + 
  xlab("Timepoint sparsity") + 
  ylab("Segment sparsity") + 
  scale_shape_manual(values = c(15,19), 
                     labels = c("Class 0 to 1", "Class 1 to 0"),
                     name = "") +
  theme_bw(base_size=9) + 
  theme(legend.position="top",
        legend.justification="right",
        #legend.direction = "vertical",
        #plot.title = element_text(hjust = 0, vjust=-3), 
        legend.box.margin = margin(0,0,-0.2,0, "line"),
        #axis.title.y = element_text(),
        panel.grid.major.x = element_blank(),
        panel.grid.minor.x = element_blank(),
        #panel.grid.major.y = element_blank(),
        panel.grid.minor.y = element_blank(),
        legend.text  = element_text(size = 7),
        legend.key.width = unit(0.2,"cm"),
        legend.key.size = unit(0.2,"cm"))

name <- paste("C:/Users/20200059/OneDrive - TU Eindhoven/Documents/Emmanuel/Rianne's analysis/ECG200Sparsity.pdf", sep = "", collapse = NULL)
ggsave(name, width = 10, height = 10, units = "cm")
  
xtable(dataprox)
  
