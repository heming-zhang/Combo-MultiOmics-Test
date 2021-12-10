getwd()
library(plyr)

### 1. FACETS PLOT
testpred <- read.csv('./datainfo/result/epoch_50/combine_testpred.csv')
testpred <- testpred[testpred$Cell.Line.Name == c('786-0', 'A498', 'A549/ATCC', 'ACHN', 'BT-549'), ]
facet <- ggplot(data = testpred,
                aes(x = Score, y = Pred.Score,
                    color = Cell.Line.Name)) +
        geom_point(aes(shape = Cell.Line.Name), size = 1.5) +
        geom_smooth(method = "lm") +
        xlab("Score") +
        ylab("Prediction Score") +
        ggtitle("Faceting of Drug Score and Predicted Score Accross Cell Line")
# Along columns
facet + facet_grid(Cell.Line.Name ~ .)

### 2. KDE PLOT
gene_edge_weight = read.csv('./datainfo/analysis_data/gene_edge_weight.csv')
gene_edge_weight_filter = filter(gene_edge_weight, weight > 0.1)
max <- which.max(density(gene_edge_weight_filter$weight)$y)
density(gene_edge_weight_filter$weight)$x[max]

ggplot(gene_edge_weight_filter, aes(x=weight))+ 
  geom_density(color="darkblue", fill="lightblue")+
  geom_vline(xintercept = density(gene_edge_weight_filter$weight)$weight[max])+
  xlab("weight")+
  scale_x_continuous(breaks=c(0.1, 0.12, 0.14, 0.16, 0.18, 0.2, 0.3, 1), trans='log10')+
  geom_vline(aes(xintercept =0.15, color='edge_weight_threshold'), linetype='dashed')+
  geom_vline(aes(xintercept =0.1, color='threshold=0.1'), linetype='dashed')+
  scale_color_manual(values = c("edge_weight_threshold" = "red", 'threshold=0.1'='black'))+
  xlab('Log10 Edge Weight')+
  ylab('Density')+
  ggtitle('KDE Plot of Gene-Gene Interaction (Start from 0.1)')+
  theme(plot.title = element_text(hjust = 0.5, size = 20, face = "bold"))