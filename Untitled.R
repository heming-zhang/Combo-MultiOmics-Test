library(shiny)
library(dplyr)
library(igraph)
library(networkD3)
library(kdensity)
library(ggplot2)

### 1. READ GRAPH [edge_index, node] FROM FILES
net_edge_weight = read.csv('./datainfo/analysis_data/webgnn_2/network_edge_weight.csv')
net_edge = net_edge_weight[, c('src', 'dest', 'weight', 'edge_type')]
net_node = read.csv('./datainfo/analysis_data/webgnn_2/node_num_dict.csv')
gene_edge_weight_filter = read.csv('./datainfo/analysis_data/webgnn_2/gene_edge_weight.csv')

filter_net_edge = filter(net_edge, weight > 0.32)
filter_net_edge_node = unique(c(filter_net_edge$src, filter_net_edge$dest))
filter_net_node = net_node[net_node$node_num %in% filter_net_edge_node,]
### 2.1 FILTER WITH GIANT COMPONENT
tmp_net = graph_from_data_frame(d=filter_net_edge, vertices=filter_net_node, directed=F)
small_deg_node = as.numeric(V(tmp_net)$name[degree(tmp_net)<=1])
refilter_net_edge<-subset(filter_net_edge, !(src %in% small_deg_node & dest %in% small_deg_node))
refilter_net_edge_node = unique(c(refilter_net_edge$src, refilter_net_edge$dest))
refilter_net_node = filter_net_node[filter_net_node$node_num %in% refilter_net_edge_node,]
### 3. BUILD UP GRAPH
net = graph_from_data_frame(d=refilter_net_edge, vertices=refilter_net_node, directed=F)
deg_node = V(net)[degree(net)>=5]
union_net = make_ego_graph(net, order=10, as.character(deg_node$name))
net = union_net[[1]]
for (subnet in union_net){
  net = net %u% subnet
}
sub_refilter_net_edge<-subset(refilter_net_edge, 
          (src %in%  as.character(V(net)$name) | dest %in% as.character(V(net)$name)))
sub_refilter_net_edge_node = unique(c(sub_refilter_net_edge$src, sub_refilter_net_edge$dest))
sub_refilter_net_node = refilter_net_node[refilter_net_node$node_num %in% sub_refilter_net_edge_node,]
sub_net = graph_from_data_frame(d=sub_refilter_net_edge, vertices=sub_refilter_net_node, directed=F)
net = sub_net

### 4. NETWORK PARAMETERS SETTINGS
# vertex frame color
vertex_fcol = rep('black', vcount(net))
# vertex_fcol[degree(net)>5] = 'tomato'
vertex_fcol[V(net)$node_type=='drug'] = 'black'
# vertex color
vertex_col = rep('lightblue', vcount(net))
vertex_col[degree(net)>=5] = 'tomato'
vertex_col[V(net)$node_type=='drug'] = 'white'
# vertex size
vertex_size = rep(5, vcount(net))
# vertex_size[degree(net)>5] = degree(net)[degree(net)>5]*1.5
vertex_size[degree(net)>=5] = 7
vertex_size[V(net)$node_type=='drug'] = 7
# vertex cex
vertex_cex = rep(0.6, vcount(net))
vertex_cex[degree(net)>=5] = 0.8
vertex_cex[V(net)$node_type=='drug'] = 0.6
# edge with
edge_width = (E(net)$weight)*(20)
edge_width[E(net)$edge_type=='drug-gene'] = 8
### 5. PLOT GRAPH
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
     edge.curved = 0.2,
     layout=layout_nicely)




