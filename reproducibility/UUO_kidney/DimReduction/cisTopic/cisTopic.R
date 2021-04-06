library(cisTopic)
library(methods)
library(stringr)

#data_folder <- "../../ArchR/filtered_peak_bc_matrix"
#metrics <- "../../ArchR/meta_data.csv"

#cisTopicObject <- createcisTopicObjectFrom10Xmatrix(data_folder, 
#                                                    metrics,  
#                                                    project.name='UUO')

counts <- readRDS("../../ArchR/PeakMatrix.Rds")
region <- as.data.frame(t(as.data.frame(str_split(rownames(counts), "_"))))
colnames(region) <- c("chrom", "p1", "p2")
rownames(counts) <- paste0(region$chrom, ":", region$p1, "-", region$p2)
cisTopicObject <- createcisTopicObject(count.matrix = counts)


cisTopicObject <- runWarpLDAModels(cisTopicObject, 
                                   topic = seq(from = 10, to = 40, by = 2), 
                                   seed = 987, 
                                   nCores = 1, 
                                   iterations = 500,
                                   addModels=FALSE)

saveRDS(cisTopicObject, file = "cisTopicObject.Rds")

cisTopicObject <- selectModel(cisTopicObject)
t1 <- as.data.frame(cisTopicObject@selected.model$document_expects)
colnames(t1) <- colnames(cisTopicObject@count.matrix)
write.table(t1, file = "./cisTopic.txt", quote = FALSE, sep = "\t")
