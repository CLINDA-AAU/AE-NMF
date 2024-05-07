# Load necessary libraries
library(readr)
library(tidyverse)
library(ggplot2)
library(lsa)
library(stringr)
library(RcppHungarian)
library(MutationalPatterns)
library(GDAtools)
library(cluster)

# Specify the index for diagnosis and relevant details
i = 3
diag <- c("Ovary", "Prostate", "Uterus")
sigs <- c(4,5,8)
diagnosis = diag[i]
nsigs = sigs[i]
nruns = 30

# Read the COSMIC data
COSMIC <- read_table("external_data/COSMIC_v3.4_SBS_GRCh37.txt")

# To arrange the GEL data with the same trinucleotide ordering as COSMIC
tri_order <- data.frame(tri = COSMIC$Type, mut = str_sub(COSMIC$Type, 3, 5))%>%arrange(mut)%>%select(tri)

# Set directory and extract signature matrices
dir <- paste0("generated_data/",diagnosis)
file_list <- list.files(dir)#"generated_data/2023-09-18Ovary/matrices")  


NMF_sig_files <- file_list[str_sub(file_list, 1, 3) == "nmf" & str_sub(file_list, -8, -5) == "sigs"]
AE_sig_files <- file_list[str_sub(file_list, 1, 7) == "AEFPabs" & str_sub(file_list, -8, -5) == "sigs"] 

NMF_path <- paste0(dir,"/", NMF_sig_files)
AE_path <- paste0(dir,"/", AE_sig_files)

print(NMF_sig_files)

NMF_COSMIC_matches <- matrix(rep(0,nruns*nsigs), nrow = nruns)
NMF_COSMIC_sim <- matrix(rep(0,nruns*nsigs), nrow = nruns)
NMF_all_sigs <- matrix(rep(0, 96*nruns*nsigs), nrow = 96)

# Loop over signature files to find COSMIC mathces for all runs
for (i in 1:length(NMF_path)){
  sig_matNMF = read_csv(NMF_path[i],col_types = cols(...1 = col_skip()))%>%mutate(order = tri_order)%>%arrange(order)%>%select(-order)
  NMF_all_sigs[,(nsigs*i-(nsigs-1)):(nsigs*i)] <- as.matrix(sig_matNMF)
  total_sigs = ncol(sig_matNMF) + ncol(COSMIC) - 1
  cosine_mat <-  cosine(as.matrix(cbind(sig_matNMF,COSMIC[,2:ncol(COSMIC)])))[1:nsigs,(nsigs+1):total_sigs]
  NMF_COSMIC_hung <- HungarianSolver(1-cosine_mat)
  NMF_COSMIC_matches[i,] <- colnames(COSMIC[,2:ncol(COSMIC)])[(NMF_COSMIC_hung$pairs)[,2]]
  NMF_COSMIC_sim[i,] <- 1- NMF_COSMIC_hung$cost/nsigs
}

print(dim(NMF_all_sigs))

AE_COSMIC_matches <- matrix(rep(0,nruns*nsigs), nrow = nruns)
AE_COSMIC_sim <- matrix(rep(0,nruns*nsigs), nrow = nruns)
AE_all_sigs <- matrix(rep(0, 96*nruns*nsigs), nrow = 96)
for (i in 1:length(AE_path)){
  sig_matAE = read_csv(AE_path[i],col_types = cols(...1 = col_skip()))%>%mutate(order = tri_order)%>%arrange(order)%>%select(-order)
  AE_all_sigs[,(nsigs*i-(nsigs-1)):(nsigs*i)] <- as.matrix(sig_matAE)
  total_sigs = ncol(sig_matAE) + ncol(COSMIC) - 1
  cosine_mat <- cosine(as.matrix(cbind(sig_matAE,COSMIC[,2:ncol(COSMIC)])))[1:nsigs,(nsigs+1):total_sigs]
  AE_COSMIC_hung <- HungarianSolver(1-cosine_mat)
  AE_COSMIC_matches[i,] <- colnames(COSMIC[,2:ncol(COSMIC)])[(AE_COSMIC_hung$pairs)[,2]]
  AE_COSMIC_sim[i,] <- 1- AE_COSMIC_hung$cost/nsigs
}

# Perform PAM clustering
NMF_pam_res <- pam(1-cosine(NMF_all_sigs), nsigs, diss = TRUE)
NMF_cons_idx <- NMF_pam_res$medoids
NMF_clust_sigs <- as.data.frame(NMF_all_sigs[,NMF_cons_idx])
rownames(NMF_clust_sigs) <- COSMIC$Type
colnames(NMF_clust_sigs) <- paste0("NMF-SBS",1:nsigs)

AE_pam_res <- pam(1-cosine(AE_all_sigs), nsigs, diss = TRUE)
AE_cons_idx <- AE_pam_res$medoids
AE_clust_sigs <- as.data.frame(AE_all_sigs[,AE_cons_idx])
rownames(AE_clust_sigs) <- COSMIC$Type

# Hungarian algorithm to match clusters between NMF and AE
NMFAE_hung <- HungarianSolver(1-cosine(as.matrix(cbind(NMF_clust_sigs, AE_clust_sigs)))[1:nsigs,(nsigs+1):(2*nsigs)])
AE_clust_sigs <- AE_clust_sigs[,NMFAE_hung$pairs[,2]]
colnames(AE_clust_sigs) <- paste0("AE-SBS",1:nsigs)

# Print results
plot_96_profile(NMF_clust_sigs)
plot_96_profile(AE_clust_sigs)

#write.csv(AE_clust_sigs, paste0("generated_data/", diagnosis, "_AE_cons_sigs.csv"))
#write.csv(NMF_clust_sigs, paste0("generated_data/", diagnosis, "_NMF_cons_sigs.csv"))

# Calulate COSMIC matches for the clustered signatures
cosine_mat <- cosine(as.matrix(cbind(AE_clust_sigs,COSMIC[,2:ncol(COSMIC)])))[1:nsigs,(nsigs+1):total_sigs]
AE_COSMIC_hung <- HungarianSolver(1-cosine_mat)
AE_cons_COSMIC_matches <- colnames(COSMIC[,2:ncol(COSMIC)])[(AE_COSMIC_hung$pairs)[,2]]
AE_cons_COSMIC_sim <- c(diag(cosine(as.matrix(cbind(AE_clust_sigs,COSMIC[,AE_cons_COSMIC_matches])))[1:nsigs, (nsigs+1):(2*nsigs)]))

cosine_mat <- cosine(as.matrix(cbind(NMF_clust_sigs,COSMIC[,2:ncol(COSMIC)])))[1:nsigs,(nsigs+1):total_sigs]
NMF_COSMIC_hung <- HungarianSolver(1-cosine_mat)
NMF_cons_COSMIC_matches <- colnames(COSMIC[,2:ncol(COSMIC)])[(NMF_COSMIC_hung$pairs)[,2]]
NMF_cons_COSMIC_sim <- c(diag(cosine(as.matrix(cbind(NMF_clust_sigs,COSMIC[,NMF_cons_COSMIC_matches])))[1:nsigs, (nsigs+1):(2*nsigs)]))

print("AE")
print(AE_cons_COSMIC_matches)
print(AE_cons_COSMIC_sim)
print(sum(AE_cons_COSMIC_sim)/nsigs)

print("NMF")
print(NMF_cons_COSMIC_matches)
print(NMF_cons_COSMIC_sim)
print(sum(NMF_cons_COSMIC_sim)/nsigs)

AE_matches <- data.frame("Match" = AE_cons_COSMIC_matches, "cosine" = AE_cons_COSMIC_sim)
AE_matches$Method <- "AE-NMF"
AE_matches$Signature <- paste0("SBS",1:nsigs)

NMF_matches <- data.frame("Match" = NMF_cons_COSMIC_matches, "cosine" = NMF_cons_COSMIC_sim)
NMF_matches$Method <- "NMF"
NMF_matches$Signature <- paste0("SBS",1:nsigs)

cons_matches <- full_join(AE_matches,NMF_matches)


## For each run
#Instead of looking at the clustered signatures, we look at the similarity to COSMIC for the signatures in each of the 30 splits:

AE_COSMIC_matches <- cbind(AE_COSMIC_matches, "AE-NMF", 1:nruns)
NMF_COSMIC_matches <- cbind(NMF_COSMIC_matches, "NMF", 1:nruns)

all_matches <-  data.frame(rbind(NMF_COSMIC_matches, AE_COSMIC_matches))
colnames(all_matches) <-  c(paste0("SBS", 1:nsigs), "Method", "Split")
all_matches <- all_matches%>%pivot_longer(SBS1:paste0("SBS",nsigs), names_to = "Signature", values_to = "Match")

NMF_COSMIC_sim <- as.data.frame(cbind(NMF_COSMIC_sim, "NMF", 1:nruns))
colnames(NMF_COSMIC_sim) <- c(paste0("SBS", 1:nsigs),"Method", "Split")

AE_COSMIC_sim <- as.data.frame(cbind(AE_COSMIC_sim, "AE-NMF", 1:nruns))
colnames(AE_COSMIC_sim) <- c(paste0("SBS", 1:nsigs), "Method", "Split")

all_sim <- data.frame(rbind(NMF_COSMIC_sim, AE_COSMIC_sim))
all_sim <- all_sim%>%pivot_longer(SBS1:paste0("SBS",nsigs), names_to = "Signature", values_to = "cosine")



all_total <- full_join(all_sim,all_matches)%>%mutate(Split = factor(Split, levels = 1:nruns))

# Plot the cosine similarity with COSMIC match for each run
ggplot(data = cbind(all_total%>%
                      group_by(Split)%>%
                      arrange(Method,Split,Match),
                    pos = rep(nsigs:1, 2*nruns)), 
       aes(x = Split, fill = Match, y = 1)) + 
  geom_bar( stat = "identity") + 
  facet_grid(rows = vars(Method)) + 
  geom_text(aes( y = pos-0.5,label=round(as.numeric(cosine),2)), size = 6.5) + 
  #scale_fill_manual(values = color)  +
  theme_bw() + 
  theme(text = element_text(size = 35), 
        axis.title = element_text(size = 40), 
        legend.position = "bottom", 
        axis.title.y = element_blank(), 
        axis.ticks.y = element_blank() , 
        axis.text.y = element_blank(),
        axis.title.x = element_blank(),
        strip.background = element_blank()) 



print(all_total%>%group_by(Method)%>%summarise(cos = mean(as.numeric(cosine))))