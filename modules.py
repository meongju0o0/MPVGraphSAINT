import dgl.function as fn
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import dgl.nn.pytorch as dglnn


class SAGEConvSUM(dglnn.SAGEConv):
    def __init__(self, in_feats, n_classes):
        super().__init__(in_feats, n_classes,
                         aggregator_type="mean", feat_drop=0, bias=False, norm=None, activation=None)

    def reset_parameters(self):
        """
        Reset weight parameters as a one
        """
        nn.init.ones_(self.fc_neigh.weight)

    def forward(self, graph, feat, edge_weight=None):
        with graph.local_scope():
            if isinstance(feat, tuple):
                feat_src = self.feat_drop(feat[0])
                feat_dst = self.feat_drop(feat[1])
            else:
                feat_src = feat_dst = self.feat_drop(feat)
                if graph.is_block:
                    feat_dst = feat_src[: graph.number_of_dst_nodes()]
            msg_fn = fn.copy_u("h", "m")
            if edge_weight is not None:
                assert edge_weight.shape[0] == graph.num_edges()
                graph.edata["_edge_weight"] = edge_weight
                msg_fn = fn.u_mul_e("h", "_edge_weight", "m")

            h_self = feat_dst

            # Handle the case of graphs without edges
            if graph.num_edges() == 0:
                graph.dstdata["neigh"] = th.zeros(
                    feat_dst.shape[0], self._in_src_feats
                ).to(feat_dst)

            # Determine whether to apply linear transformation before message passing A(XW)
            lin_before_mp = self._in_src_feats > self._out_feats

            # Message Passing
            graph.srcdata["h"] = (
                self.fc_neigh(feat_src) if lin_before_mp else feat_src
            )
            graph.update_all(msg_fn, fn.sum("m", "neigh"))
            h_neigh = graph.dstdata["neigh"]
            if not lin_before_mp:
                h_neigh = self.fc_neigh(h_neigh)

        rst = self.fc_self(h_self) + h_neigh

        return rst


class SimpleAGG(nn.Module):
    """
    Simple Aggregation Model to Calculate ego-graph's changing rate

    Parameters
    ----------
    num_hop : int
        Depth of Aggregation
    """

    def __init__(self, num_hop, in_feats=1, n_classes=1, dropout=0):
        super().__init__()
        self.layers = nn.ModuleList()

        for _ in range(num_hop):
            self.layers.append(SAGEConvSUM(in_feats, n_classes))

        self.dropout = nn.Dropout(dropout)

    def forward(self, blocks, x):
        """
        Forward function.

        Parameters
        ----------
        blocks : List[DGLBlock]
            Sampled blocks.
        x : DistTensor
            Feature data.
        """
        h = x
        for i, (layer, block) in enumerate(zip(self.layers, blocks)):
            h = layer(block, h)
            if i != len(self.layers) - 1:
                h = self.dropout(h)
        return h


class GCNLayer(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        order=1,
        act=None,
        dropout=0,
        batch_norm=False,
        aggr="concat",
    ):
        super(GCNLayer, self).__init__()
        self.lins = nn.ModuleList()
        self.bias = nn.ParameterList()
        for _ in range(order + 1):
            self.lins.append(nn.Linear(in_dim, out_dim, bias=False))
            self.bias.append(nn.Parameter(th.zeros(out_dim)))

        self.order = order
        self.act = act
        self.dropout = nn.Dropout(dropout)

        self.batch_norm = batch_norm
        if batch_norm:
            self.offset, self.scale = nn.ParameterList(), nn.ParameterList()
            for _ in range(order + 1):
                self.offset.append(nn.Parameter(th.zeros(out_dim)))
                self.scale.append(nn.Parameter(th.ones(out_dim)))

        self.aggr = aggr
        self.reset_parameters()

    def reset_parameters(self):
        for lin in self.lins:
            nn.init.xavier_normal_(lin.weight)

    def feat_trans(
        self, features, idx
    ):  # linear transformation + activation + batch normalization
        h = self.lins[idx](features) + self.bias[idx]

        if self.act is not None:
            h = self.act(h)

        if self.batch_norm:
            mean = h.mean(dim=1).view(h.shape[0], 1)
            var = h.var(dim=1, unbiased=False).view(h.shape[0], 1) + 1e-9
            h = (h - mean) * self.scale[idx] * th.rsqrt(var) + self.offset[idx]

        return h

    def forward(self, graph, features):
        g = graph.local_var()
        h_in = self.dropout(features)
        h_hop = [h_in]

        D_norm = (
            g.ndata["train_D_norm"]
            if "train_D_norm" in g.ndata
            else g.ndata["full_D_norm"]
        )
        for _ in range(self.order):  # forward propagation
            g.ndata["h"] = h_hop[-1]
            if "w" not in g.edata:
                g.edata["w"] = th.ones((g.num_edges(),)).to(features.device)
            g.update_all(fn.u_mul_e("h", "w", "m"), fn.sum("m", "h"))
            h = g.ndata.pop("h")
            h = h * D_norm
            h_hop.append(h)

        h_part = [self.feat_trans(ft, idx) for idx, ft in enumerate(h_hop)]
        if self.aggr == "mean":
            h_out = h_part[0]
            for i in range(len(h_part) - 1):
                h_out = h_out + h_part[i + 1]
        elif self.aggr == "concat":
            h_out = th.cat(h_part, 1)
        else:
            raise NotImplementedError

        return h_out


class GCNNet(nn.Module):
    def __init__(
        self,
        in_dim,
        hid_dim,
        out_dim,
        arch="1-1-0",
        act=F.relu,
        dropout=0,
        batch_norm=False,
        aggr="concat",
    ):
        super(GCNNet, self).__init__()
        self.gcn = nn.ModuleList()

        orders = list(map(int, arch.split("-")))
        self.gcn.append(
            GCNLayer(
                in_dim=in_dim,
                out_dim=hid_dim,
                order=orders[0],
                act=act,
                dropout=dropout,
                batch_norm=batch_norm,
                aggr=aggr,
            )
        )
        pre_out = ((aggr == "concat") * orders[0] + 1) * hid_dim

        for i in range(1, len(orders) - 1):
            self.gcn.append(
                GCNLayer(
                    in_dim=pre_out,
                    out_dim=hid_dim,
                    order=orders[i],
                    act=act,
                    dropout=dropout,
                    batch_norm=batch_norm,
                    aggr=aggr,
                )
            )
            pre_out = ((aggr == "concat") * orders[i] + 1) * hid_dim

        self.gcn.append(
            GCNLayer(
                in_dim=pre_out,
                out_dim=hid_dim,
                order=orders[-1],
                act=act,
                dropout=dropout,
                batch_norm=batch_norm,
                aggr=aggr,
            )
        )
        pre_out = ((aggr == "concat") * orders[-1] + 1) * hid_dim

        self.out_layer = GCNLayer(
            in_dim=pre_out,
            out_dim=out_dim,
            order=0,
            act=None,
            dropout=dropout,
            batch_norm=False,
            aggr=aggr,
        )

    def forward(self, graph):
        h = graph.ndata["feat"]

        for layer in self.gcn:
            h = layer(graph, h)

        h = F.normalize(h, p=2, dim=1)
        h = self.out_layer(graph, h)

        return h
