library(shiny)
library(dplyr)
library(igraph)
library(networkD3)
library(kdensity)
library(ggplot2)

### 1. READ GRAPH [edge_index, node] FROM FILES
net_edge_weight = read.csv('./datainfo/analysis_data/network_edge_weight_cmeth.csv')
net_edge = net_edge_weight[, c('src', 'dest', 'weight', 'edge_type')]
net_node = read.csv('./datainfo/analysis_data/node_num_dict.csv')

gene_edge_weight_filter = read.csv('./datainfo/analysis_data/gene_edge_weight_cmeth.csv')
# gene_edge_weight_filter = filter(gene_edge_weight_filter, weight > 0.1)

ui <- fluidPage(
  titlePanel('Whole Network Interaction'),
  sidebarLayout(
    sidebarPanel(
      sliderInput('threshold',
                  'Select the threshold of edge weight to plot',
                  min = 0.005, max = 0.2,
                  value = 0.015),
      selectInput('community', label = 'Selection of community detection', 
                  choices = list('No Community Detection' = 0, 
                                 'Greedy Community Detection' = 1, 
                                 'Spectral Community Detection' = 2,
                                 'Betweenness Community Detection' = 3), 
                  selected = 0),
      sliderInput('edge_width',
                  'Select the coefficient edge width to plot',
                  min = 5, max = 50,
                  value = 20),
      sliderInput('drug_edge_width',
                  'Select the drug-gene edge width',
                  min = 0, max = 0.5,
                  value = 0.3),
      sliderInput('drug_label_size',
                  'Select the label size of drug nodes',
                  min = 0.1, max = 2.0,
                  value = 0.7),
      selectInput('drug_node_color', 
                  'Select the drug node color', 
                  choices = c('tomato', 'gold', 'skyblue','green', 'white'),
                  selected='white'),
      sliderInput('gene_label_size',
                  'Select the label size of gene nodes',
                  min = 0.1, max = 1.5,
                  value = 0.5),
      sliderInput('imgene_label_size',
                  'Select the label size of important genes',
                  min = 0.1, max = 2.5,
                  value = 0.7),
      selectInput('gene_node_color', 
                  'Select the common gene color', 
                  choices = c('lightblue', 'gold', 'purple','green', 'white', 'tomato'),
                  selected='lightblue'),
      selectInput('impgene_node_color', 
                  'Select the important gene color', 
                  choices = c('skyblue', 'gold', 'purple','green', 'white', 'tomato'),
                  selected='tomato')
    ),
    mainPanel(
      plotOutput(outputId = 'kde', height = 120, width = 1150),
      plotOutput(outputId = 'network', height = 880, width = 1150)
    )
  )
)

server <- function(input, output) {
  filter_net_edge <- reactive({
    filter_net_edge = filter(net_edge, weight > input$threshold)
  })
  output$kde <- renderPlot({
    ggplot(gene_edge_weight_filter, aes(x=weight))+ 
      geom_density(color="darkblue", fill="lightblue")+
      geom_vline(xintercept = density(gene_edge_weight_filter$weight)$weight[max])+
      xlab("weight")+
      # scale_x_continuous(breaks=c(0.1, 0.12, 0.14, 0.16, 0.18, 0.2, 0.3, 1), trans='log10')+
      geom_vline(aes(xintercept =input$threshold, color='edge_weight_threshold'), linetype='dashed')+
      # geom_vline(aes(xintercept =0.1, color='threshold=0.1'), linetype='dashed')+
      # scale_color_manual(values = c("edge_weight_threshold" = "red", 'threshold=0.1'='black'))+
      xlab('Log10 Edge Weight')+
      ylab('Density')+
      ggtitle('KDE Plot of Gene-Gene Interaction (Start from 0.1)')+
      theme(plot.title = element_text(hjust = 0.5, size = 20, face = "bold"))
  })
  output$network <- renderPlot({
    ### 2. FILTER EDGE BY [edge_weight]
    filter_net_edge_node = unique(c(filter_net_edge()$src, filter_net_edge()$dest))
    filter_net_node = net_node[net_node$node_num %in% filter_net_edge_node,]
    ### 2.1 FILTER WITH GIANT COMPONENT
    tmp_net = graph_from_data_frame(d=filter_net_edge(), vertices=filter_net_node, directed=F)
    small_deg_node = as.numeric(V(tmp_net)$name[degree(tmp_net)<=1])
    refilter_net_edge<-subset(filter_net_edge(), !(src %in% small_deg_node & dest %in% small_deg_node))
    refilter_net_edge_node = unique(c(refilter_net_edge$src, refilter_net_edge$dest))
    refilter_net_node = filter_net_node[filter_net_node$node_num %in% refilter_net_edge_node,]
    ### 3. BUILD UP GRAPH
    net = graph_from_data_frame(d=refilter_net_edge, vertices=refilter_net_node, directed=F)
    ### 4. NETWORK PARAMETERS SETTINGS
    # vertex frame color
    vertex_fcol = rep('black', vcount(net))
    # vertex_fcol[degree(net)>5] = 'tomato'
    vertex_fcol[V(net)$node_type=='drug'] = 'black'
    # vertex color
    vertex_col = rep(input$gene_node_color, vcount(net))
    vertex_col[degree(net)>=5] = input$impgene_node_color
    vertex_col[V(net)$node_type=='drug'] = input$drug_node_color
    # vertex size
    vertex_size = rep(5, vcount(net))
    # vertex_size[degree(net)>5] = degree(net)[degree(net)>5]*1.5
    vertex_size[degree(net)>=5] = 10
    vertex_size[V(net)$node_type=='drug'] = 7
    # vertex cex
    vertex_cex = c(input$drug_label_size, input$gene_label_size)[1+(V(net)$node_type=='gene')]
    vertex_cex[degree(net)>=5] = input$imgene_label_size
    # edge with
    edge_width = (E(net)$weight-input$threshold+0.2)*(input$edge_width)
    edge_width[E(net)$edge_type=='drug-gene'] = (input$drug_edge_width)*(input$edge_width)
    ### 5. PLOT GRAPH
    if (input$community == 0){
      set.seed(7778)
      plot(net,
           vertex.frame.width=25,
           vertex.frame.color = vertex_fcol,
           vertex.color = vertex_col,
           vertex.size = vertex_size,
           vertex.shape = c('square', 'circle')[1+(V(net)$node_type=='gene')],
           vertex.label = V(net)$node_name,
           vertex.label.color = 'black',
           vertex.label.font = c(4, 2)[1+(V(net)$node_type=='gene')],
           vertex.label.cex = vertex_cex,
           edge.width = edge_width,
           edge.color = c('gold', 'gray')[1+(E(net)$edge_type=='gene-gene')],
           edge.alpha = c(input$edge_opacity, 1.0)[1+(E(net)$edge_type=='gene-gene')],
           edge.curved = 0.2,
           layout=layout_nicely)
      ### 6. ADD LEGEND
      legend(x=-0.95, y=-0.94, legend=c('Genes','Drugs', 'Important Genes'), pch=c(21,22,21), 
             pt.bg=c(input$gene_node_color, input$drug_node_color, input$impgene_node_color), pt.cex=2, cex=1.2, bty='n')
      legend(x=-0.95, y=-1.1, 
             legend=c('Drug-Gene', 'Gene-Gene'),
             col=c('gold','gray'), lwd=c(5,5), cex=1.2, bty='n')
    }else if (input$community == 1){
      community = cluster_fast_greedy(net)
    }else if (input$community == 2){
      community = cluster_leading_eigen(net)
    }else if (input$community == 3){
      community = cluster_edge_betweenness(net)
    }
    if (input$community != 0){
      set.seed(7778)
      plot(community, net,
           vertex.frame.width=25,
           vertex.frame.color = vertex_fcol,
           vertex.color = vertex_col,
           vertex.size = vertex_size,
           vertex.shape = c('square', 'circle')[1+(V(net)$node_type=='gene')],
           vertex.label = V(net)$node_name,
           vertex.label.color = 'black',
           vertex.label.font = c(4, 2)[1+(V(net)$node_type=='gene')],
           vertex.label.cex = vertex_cex,
           edge.width = (E(net)$weight-input$threshold+0.2)*(input$edge_width),
           edge.color = c('gold', 'gray')[1+(E(net)$edge_type=='gene-gene')],
           edge.curved = 0.2,
           layout=layout_nicely)
      ### 6. ADD LEGEND
      legend(x=-0.95, y=-0.94, legend=c('Genes','Drugs'), pch=c(21,22), 
             pt.bg=c(input$gene_node_color, input$drug_node_color), pt.cex=2, cex=1.2, bty='n')
      legend(x=-0.95, y=-1.1, 
             legend=c('Drug-Gene', 'Gene-Gene'),
             col=c('gold','gray'), lwd=c(5,5), cex=1.2, bty='n')
    }
    title(main = 'Whole Network Interaction', cex.main = 1.75)
  })
}

shinyApp(ui = ui, server = server)









